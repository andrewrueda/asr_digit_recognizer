import torch
import torch.functional as F
import numpy as np
import argparse
import inspect
import json
import os
from data import FeatureExtractor, DataManager
from model import WordRecognizer, HMM, HMMState, GMM
from typing import List, Dict, Tuple, Set, Union
import logging



class HMMTest:
    """Test class for Word Recognizers"""
    def __init__(self, word_recognizer: WordRecognizer, vocab: List[str]):
        self.word_recognizer = word_recognizer
        self.vocab = vocab


    def evaluate(self, labeled_inputs: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        correct = 0
        total = 0
        by_word = {}

        for target, tensor in labeled_inputs.items():
            preds = self.predict(tensor)

            correct_i = len([x for x in preds if x == target])
            total_i = len(preds)

            by_word[target] = round(correct_i / total_i, 4) if total_i else 0.

            correct += correct_i
            total += total_i

        final = (correct/total) if total else 0.

        logging.info(f"Accuracy: {final:4f}\n")

        logging.info(f"by digit:")
        by_word = [(self.vocab[i], by_word[self.vocab[i]]) for i in range(len(self.vocab))]
        for word, acc in by_word:
            logging.info(f"{word}: {acc}")
        

    def predict(self, x: torch.Tensor) -> List[str]:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        N = x.size(0)
        T = x.size(1)

        words = []
        log_probs = []

        for word, model in self.word_recognizer.models.items():
            words.append(word)
            S = model.n_states

            # get unpadded seq_lens
            mask = (x != 0).any(dim=-1) # (N, T)
            seq_lens = mask.sum(dim=1) - 1

            # Viterbi decoding

            # get emissions and transitions
            emissions = torch.zeros(N, T, S)
            for i in range(S):
                emissions[:, :, i] = model.states[i].gmm(x)

            transitions = model.transitions.unsqueeze(0) # (N, S, S)

            # build lattice
            lattice = torch.zeros(N, T, S) # (N, T, S)

            lattice[:, 0, :] = torch.Tensor([1.] + [0.] * (S-1)).log() + emissions[:, 0, :]
            # backpointers = np.full((N, T, S), -1, dtype=int) # (N, T, S)

            for t in range(1, T):
                new_paths = (emissions[:, t, :] + lattice[:, t-1, :]).unsqueeze(-1) # (N, S, 1)
                new_paths = new_paths + transitions # (N, S, S)
                
                # backpointers[:, t, :] = torch.argmax(new_paths, dim=-1)
                lattice[:, t, :] = torch.max(new_paths, dim=1).values

            # go to each final unpadded observation and find best path log probs
            N_range = torch.arange(N)
            best_paths = torch.max(lattice[N_range, seq_lens, :], dim=1).values # (N,)
            log_probs.append(best_paths)

        log_probs = torch.stack(log_probs, dim=0)

        preds = torch.argmax(log_probs, dim=0)
        return [words[i] for i in preds]


def set_logging(args):
    log_subdir = os.path.join(args.log_dir, args.model_id)
    n_files = sum(1 for entry in os.scandir(log_subdir) if entry.is_file())
    log_path = os.path.join(log_subdir, f"{args.model_id}-{n_files+1}.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode='w'
    )

    # Add console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(console)


def prepare_test_data(args) -> Dict[str, torch.Tensor]:
    """Load and prepare test data"""

    # get only relevant kwargs for feature extractor
    feature_params = inspect.signature(FeatureExtractor.__init__).parameters
    feature_kwargs = {key: getattr(args, key) for key in feature_params
                      if key != "self" and hasattr(args, key)}
    

    feature_extractor = FeatureExtractor(**feature_kwargs)
    data_manager = DataManager(args.data_dir)

    # gather and load data kaldi-style for test
    test_files = data_manager.gather_by_word(test=True, test_indx=args.test_indx, vocab=args.vocab)
    data_manager.kaldi_prepare(test_files, output_dir=os.path.join(args.output_dir, "test"))
    test_data = data_manager.load_data(split="test")


    logging.info(f"Loading test tensors...")
    test_tensors = feature_extractor.target_tensors(test_data)

    return test_tensors


def prepare_one_inference(args) -> torch.Tensor:
    """Get features for one sample"""
    feature_params = inspect.signature(FeatureExtractor.__init__).parameters
    feature_kwargs = {key: getattr(args, key) for key in feature_params
                      if key != "self" and hasattr(args, key)}
    
    feature_extractor = FeatureExtractor(**feature_kwargs)
    return feature_extractor.extract_features(args.audio_file)



def main():
    # get model settings
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('model_id', type=str, help='model name')
    parser.add_argument('--audio_file', type=str, default=None)
    args = parser.parse_args()
    one_sample = args.audio_file

    with open(f"saved/{args.model_id}/settings.json", "r", encoding="utf-8") as settings_file:
        settings = json.load(settings_file)

        for key, value in settings.items():
            setattr(args, key, value)


    recognizer = torch.load(f"saved/{args.model_id}/{args.model_id}.pt", weights_only=False)
    tester = HMMTest(recognizer, vocab=args.vocab)

    if one_sample:
        x = prepare_one_inference(args)
        y = tester.predict(x)
        print(y[0])
        
    else:
        set_logging(args)
        test_tensors = prepare_test_data(args)
        tester.evaluate(test_tensors)


if __name__ == "__main__":
    main()
