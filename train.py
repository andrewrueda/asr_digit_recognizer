import torch
import torch.functional as F
import json
import os
import argparse
import inspect
import random
from typing import List, Dict, Tuple, Set, Union
from data import DataManager, FeatureExtractor
from model import HMM, HMMState, GMM
from eval import HMMTest
import numpy as np


def parse_args():
    # Load configs as default
    with open("configs/config.json", "r", encoding="utf-8") as config_file:
        configs = json.load(config_file)
    parser = argparse.ArgumentParser(description='Training')

    # parse data
    data_configs = configs["data"]
    parser.add_argument('--data_dir', type=str, default=data_configs['data_dir'])
    parser.add_argument('--output_dir', type=str, default=data_configs['output_dir'])
    parser.add_argument('--log_dir', type=str, default=data_configs['log_dir'])
    parser.add_argument('--test_indx', type=int, default=data_configs['test_indx'])
    parser.add_argument('--vocab', type=dict, default=data_configs['vocab'])

    # parse audio features
    features_configs = configs["features"]
    parser.add_argument('--fs', type=int, default=features_configs['fs'])
    parser.add_argument('--n_mels', type=int, default=features_configs['n_mels'])
    parser.add_argument('--hop_length', type=int, default=features_configs['hop_length'])
    parser.add_argument('--win_length', type=int, default=features_configs['win_length'])
    parser.add_argument('--fmin', type=int, default=features_configs['fmin'])
    parser.add_argument('--fmax', type=int, default=features_configs['fmax'])
    parser.add_argument('--frontend_type', type=str, default=features_configs['frontend_type'])
    parser.add_argument('--use_specaug', type=bool, default=features_configs['use_specaug'])

    # parse model
    model_configs = configs["model"]
    parser.add_argument('--n_states', type=int, default=model_configs['n_states'])
    parser.add_argument('--n_components', type=int, default=model_configs['n_components'])
    parser.add_argument('--inner_epochs', type=int, default=model_configs['inner_epochs'])
    parser.add_argument('--device', type=str, default=model_configs['device'])

    # parse training
    training_configs = configs["training"]
    parser.add_argument('--seed', type=int, default=training_configs['seed'])
    parser.add_argument('--epochs', type=int, default=training_configs['epochs'])

    return parser.parse_args()


def prepare_data(args) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and prepare data"""

    # get only relevant kwargs for feature extractor
    feature_params = inspect.signature(FeatureExtractor.__init__).parameters
    feature_kwargs = {key: getattr(args, key) for key in feature_params
                      if key != "self" and hasattr(args, key)}
    
    feature_extractor = FeatureExtractor(**feature_kwargs)
    data_manager = DataManager(args.data_dir)

    # gather and load data kaldi-style for train and test
    train_files = data_manager.gather_by_word(test=False, test_indx=args.test_indx, vocab=args.vocab)
    data_manager.kaldi_prepare(train_files, output_dir=os.path.join(args.output_dir, "train"))

    test_files = data_manager.gather_by_word(test=True, test_indx=args.test_indx, vocab=args.vocab)
    data_manager.kaldi_prepare(test_files, output_dir=os.path.join(args.output_dir, "test"))

    train_data = data_manager.load_data(split="train")
    test_data = data_manager.load_data(split="test")

    # extract and load tensors
    print(f"Loading training tensors...")
    training_tensors = feature_extractor.target_tensors(train_data)
    print(f"Loading test tensors...")
    test_tensors = feature_extractor.target_tensors(test_data)

    return training_tensors, test_tensors


def load_model(args) -> Dict[str, HMM]:
    print(f"Loading model...")
    word_models = dict()

    for digit in range(10):
        states = [HMMState(id=i, n_components=args.n_components, n_features=args.n_mels, inner_epochs=args.inner_epochs)
                  for i in range(args.n_states)]

        for i in range(args.n_states):
            if i < args.n_states - 1:
                states[i].add_transition(i, np.log(0.5))      # self-loop
                states[i].add_transition(i + 1, np.log(0.5))  # forward
            else:
                states[i].add_transition(i, np.log(1.0))      # final state self-loop
        
        word_models[args.vocab[str(digit)]] = HMM(states)

    print(f"number of HMMs: {len(word_models)}")
    return word_models


def forward_backward(args, model: HMM, observations: torch.Tensor,
                     state_responsibilities: torch.Tensor, observation_lengths: torch.Tensor) -> torch.Tensor:
    """E step of Baum-Welch (Forward Backward Algorithm)"""

    N, T, F = observations.size()
    S = state_responsibilities.size(2)
    
    # build lattice
    lattice = torch.zeros(N, T, S)

    for i in range(S):
        emissions = model.states[i].emission_probs(observations) # (N, T)
        lattice[:, :, i] = emissions

    # get transitions
    transitions = model.transitions.unsqueeze(0) # (1, S, S)

    # fill alpha tensor
    alpha = torch.zeros(N, T, S)
    alpha[:, 0, 0] = lattice[:, 0, 0]
    alpha[:, 0, 1:] = float('-inf')

    for t in range(1, T):
        # add last alpha, last emissions, and logsumexped transitions
        alpha_t = alpha[:, t-1, :].detach().clone().unsqueeze(-1) # (N, S, 1)
        emissions_t = lattice[:, t-1, :].detach().clone().unsqueeze(-1) # (N, S, 1)

        alpha_t = alpha_t + emissions_t + transitions # (N, S, S)

        alpha_t = torch.logsumexp(alpha_t, dim=1) # (N, 1, S)
        alpha[:, t, :] = alpha_t


    # fill beta tensor
    beta = torch.zeros(state_responsibilities.shape) # (N, T, S)

    last_indices = torch.tensor(observation_lengths) - 1 # (N,)
    N_range = torch.arange(N)

    # set final state value from lattice, neg inf for other states
    beta[N_range, last_indices, S-1] = lattice[N_range, last_indices, S-1]
    beta[N_range, last_indices, :S-1] = float('-inf')

    # transpose transitions
    transitions_back = transitions.transpose(1, 2)

    for i in range(N):
        for t in range(last_indices[i]-1, 0, -1):

            beta_t = beta[i, t+1, :].detach().clone().unsqueeze(-1) # (1, S, 1)

            emissions_t = lattice[i, t, :].detach().clone().unsqueeze(-1) # (1, S, 1)

            beta_t = beta_t + transitions_back + emissions_t # (1, S, S)
            
            beta_t = torch.logsumexp(beta_t, dim=1) # (1, 1, S)

            beta[i, t, :] = beta_t


    # initialize new log state responsibilities
    new_state_responsibilities = torch.full((N, T, S), float('-inf'))
    new_state_responsibilities[:, 0, :] = torch.tensor([1., 0., 0.]).log()

    gamma = alpha[:, 1:, :] + beta[:, 1:, :] # (N, T-1, S)

    # ???
    log_likelihood = torch.logsumexp(alpha[N_range, last_indices, :], dim=-1)  # (N,) sum over final states
    gamma -= log_likelihood.unsqueeze(-1).unsqueeze(-1)

    new_state_responsibilities[:, 1:, :] = gamma


    # use mask to convert to -inf when t >= seq_len
    T_range = torch.arange(T).unsqueeze(0)  # (1, T)
    mask = T_range >= last_indices.unsqueeze(1)  # (N, T)

    new_state_responsibilities[mask.unsqueeze(-1).expand(-1, -1, S)] = float('-inf')
    new_state_responsibilities = torch.softmax(new_state_responsibilities, dim=2)

    new_state_responsibilities[mask.unsqueeze(-1).expand(-1, -1, S)] = 0.

    # force align end
    new_state_responsibilities[N_range, last_indices, S-1] = 1.

    return new_state_responsibilities.log()
    

def _uniform_segmentation(seq_len: int, n_states: int) -> List[int]:
    assert(seq_len >= n_states)

    part = seq_len // n_states
    modulus = seq_len % n_states

    weights = [part] * n_states
    small = set([x for x in range(n_states)])

    for _ in range(modulus):
        extra = random.choice(list(small))
        small.remove(extra)
        weights[extra] += 1

    return weights


def main():
    """Baum-Welch training for HMM-GMM."""
    args = parse_args()
    training_tensors, test_tensors = prepare_data(args)
    word_models = load_model(args)

    # Training
    for target, tensor in training_tensors.items():
        print(f"training word {target}!")
        hmm = word_models[target]

        # initialize state responsibilities
        state_responsibilties = torch.ones(tensor.shape[0], tensor.shape[1], args.n_states) # (N, T, S)

        # find observation lengths
        mask = (tensor != 0).any(dim=-1) # (N, T)
        observation_lengths = mask.sum(dim=1)

        # initialize with uniform segmentation
        for i in range(state_responsibilties.size(0)):
            seq_len = observation_lengths[i].item()
            weights = _uniform_segmentation(seq_len, args.n_states)

            for j in range(args.n_states):
                new_row = []
                for k in range(len(weights)):

                    z = ((1. - (0.05 * (args.n_states-1))) if j==k else 0.05)

                    # z = (1. if j==k else 0.)

                    new_row.extend([z] * weights[k])

                unused = [0.] * (state_responsibilties.size(1) - seq_len)
                new_row.extend(unused)

                state_responsibilties[i, :, j] = torch.tensor(new_row)


            # force align start and end
            force_start = torch.tensor([1.] + [0.] * (args.n_states-1))
            state_responsibilties[i, 0, :] = force_start

            force_end = torch.tensor([0.] * (args.n_states-1) + [1.])
            state_responsibilties[i, seq_len-1, :] = force_end

        # take log
        state_responsibilties = state_responsibilties.log()

        # initial M step (fit gaussians)
        for i in range(hmm.n_states):
            state = hmm.states[i]
            state.gmm.fit(tensor, state_responsibilties[:,:, i])

        for epoch in range(1, args.epochs):
            # E step: update state responsibilities
            state_responsibilties = forward_backward(args, hmm, tensor, state_responsibilties, observation_lengths)

            # M step: update gaussians
            print(f"M step:")
            for i in range(hmm.n_states):
                state = hmm.states[i]
                print(f"state {i}...")
                state.gmm.fit(tensor, state_responsibilties[:,:, i])



    # temp eval
    vocab_list = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    results = {}

    for target in vocab_list:

        tester = HMMTest(word_models, test_tensor = test_tensors[target])

        predictions = tester.batch_inference()

        correct = 0
        total = 0
        for pred, value in predictions:
            if pred == target:
                correct += 1
            total += 1

        results[target] = correct/total

    final = sum(results.values())
    print(final / 10)
    print()
    print(results)



if __name__ == "__main__":
    main()