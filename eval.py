import torch
import torch.functional as F
import numpy as np
from model import WordRecognizer, HMM, HMMState, GMM
from typing import List, Dict, Tuple, Set, Union


class HMMTest:
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

            by_word[target] = round(correct_i / total_i, 4)

            correct += correct_i
            total += total_i

        final = correct/total

        print(f"Accuracy: {final:4f}\n")

        print(f"by digit:")
        by_word = [(self.vocab[i], by_word[self.vocab[i]]) for i in range(len(self.vocab))]
        for word, acc in by_word:
            print(f"{word}: {acc}")
        

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


if __name__ == "__main__":
    pass


