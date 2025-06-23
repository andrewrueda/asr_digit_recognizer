import torch
import torch.functional as F
import json
import os
import argparse
from data import DataManager, FeatureExtractor
from model import HMM, HMMState, GMM
import numpy as np


def parse_args():
    pass


def setup_configs():
    with open("configs/config.json", "r", encoding="utf-8") as config_file:
        configs = json.load(config_file)
        data_dir = configs["data_dir"]
        return data_dir


def prepare_data(data_dir):
    """Load and prepare data"""
    feature_extractor = FeatureExtractor()
    data_manager = DataManager(data_dir)

    vocab_list = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    vocab = { str(i) : vocab_list[i] for i in range(10) }

    if not os.path.isdir("data"):
        os.mkdir("data")
        os.mkdir("data/train")
        os.mkdir("data/test")

        train_files = data_manager.gather_by_word(test=False, vocab=vocab)
        data_manager.kaldi_prepare(train_files, output_dir="data/train")

        test_files = data_manager.gather_by_word(test=True, vocab=vocab)
        data_manager.kaldi_prepare(test_files, output_dir="data/test")


    train_data = data_manager.load_data(split="train")
    test_data = data_manager.load_data(split="test")

    print(f"Loading training tensors...")
    training_tensors = feature_extractor.target_tensors(train_data)


def load_model():
    word_models = dict()
    n_states = configs["n_states"]

    for word in vocab_list:
        states = [HMMState(id=i) for i in range(n_states)]

        for i in range(n_states):
            if i < n_states - 1:
                states[i].add_transition(i, np.log(0.5))      # self-loop
                states[i].add_transition(i + 1, np.log(0.5))  # forward
            else:
                states[i].add_transition(i, np.log(1.0))      # final state self-loop
        
        word_models[word] = HMM(states, start_id=0)


def train_epoch():
    pass


def main():
    args = parse_args()



if __name__ == "__main__":
    main()


    # digit = word_models['zero']
    # digit_tensors = training_tensors['zero']
    # digit_tensor_T = len(digit_tensors[1])

    # a = digit.states[0]
    # b = digit.states[1]
    # c = digit.states[2]

    # gmm_a = a.emissions

    # state_responsibilities = torch.randn(240, digit_tensor_T) ** 2
    # state_responsibilities = state_responsibilities / state_responsibilities.sum(dim=1, keepdim=True)

    # gmm_a.fit(digit_tensors, state_responsibilities)

    # print(gmm_a.mixture_weights)
    # print(gmm_a.means[0])
    # print(gmm_a.log_vars[0])