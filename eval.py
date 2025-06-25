import torch
import numpy as np
from model import HMM, HMMState, GMM
from typing import List, Dict, Tuple, Set, Union


class HMMTest:
    def __init__(self, hmm: Dict[str, HMM], test_tensor: torch.Tensor = None):
        self.hmm = hmm
        self.test_tensor = test_tensor

    def batch_inference(self):
        preds = []
        for i in range(self.test_tensor.size(0)):
            pred, pred_value = self.predict(self.test_tensor[i, :, :])
            preds.append((pred, pred_value))
            
        return preds


    def predict(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        N = x.size(0)
        T = x.size(1)
        S = self.hmm['one'].n_states


        log_probs = torch.zeros(10)
        word_predictions = {}
        vocab_list = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        vocab = {vocab_list[i]: i for i in range(len(vocab_list))}


        for word, model in self.hmm.items():

            # get unpadded seq_len
            mask = (x != 0).any(dim=-1) # (N, T)
            seq_len = mask.sum(dim=1)

            # Viterbi decoding

            # get emissions and transitions
            emissions = torch.zeros(N, T, S)
            for i in range(S):
                emissions[:, :, i] = model.states[i].gmm(x)

            transitions = model.transitions.unsqueeze(0) # (N, S, S)

            # build lattice
            lattice = torch.zeros(seq_len, S).unsqueeze(0) # (N, T, S)
            lattice[:, 0, :] = torch.Tensor([1.] + [0.] * (S-1)).log() + emissions[:, 0, :]
            # backpointers = np.full((1, seq_len, S), -1, dtype=int) # (N, T, S)

            for t in range(1, seq_len):
                new_paths = (emissions[:, t, :] + lattice[:, t-1, :]).unsqueeze(-1) # (N, S, 1)
                new_paths = new_paths + transitions # (N, S, S)
                
                # backpointers[:, t, :] = torch.argmax(new_paths, dim=-1)
                lattice[:, t, :] = torch.max(new_paths, dim=1).values

            best_path = torch.max(lattice[:, seq_len-1, :], dim=-1).values # (N,)

            log_probs[vocab[word]] = best_path
            word_predictions[word] = best_path
            
        prediction = max(word_predictions, key=word_predictions.get)
        pred_value = word_predictions[prediction]
        return prediction, pred_value
    

if __name__ == "__main__":
    w = {'one': HMM([HMMState(0), HMMState(1), HMMState(2)])}
    y = HMMTest(w)

    x = torch.Tensor([.1, .3, .5, .2, -.1])
    y.predict(x)


