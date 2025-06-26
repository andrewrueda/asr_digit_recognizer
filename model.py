import torch
from torch import nn
import numpy as np
import os
from typing import List, Dict, Tuple, Set, Union
import torch.nn.functional as F
import adjectiveanimalnumber

# from torch.distributions import Categorical, Normal, MixtureSameFamily


class GMM(torch.nn.Module):
    def __init__(self, n_components: int, n_features: int, inner_epochs: int = 10):
        super().__init__()
        self.n_components = n_components
        self.n_features= n_features
        self.inner_epochs = inner_epochs

        # initialize
        self.mixture_weights = nn.Parameter(torch.ones(n_components) / n_components) # 1d, Uniform
        self.means = nn.Parameter(torch.randn(n_components, n_features) * 0.1) # 2d, Random
        self.log_vars = nn.Parameter(torch.zeros(n_components, n_features)) # 2d, Zeroes


    def fit(self, observations: torch.Tensor, state_responsibilities: torch.Tensor,
            eps: float = 1e-6):
        """Fit GMM with inner EM algorithm"""

        prev_log_likelihood = float('-inf')

        for epoch in range(self.inner_epochs):
            component_responsibilities = self._e_step(observations)

            self._m_step(observations, component_responsibilities, state_responsibilities)

            emissions = self(observations)
            current_log_likelihood = torch.logsumexp(emissions, dim=tuple(range(emissions.ndim))).item()

            if abs(current_log_likelihood - prev_log_likelihood) < eps:
                break

            prev_log_likelihood = current_log_likelihood

        return current_log_likelihood


    def _e_step(self, observations):
        """Compute component responsibilities"""
        # create mask
        mask = (observations != 0).any(dim=-1) # (N, T)

        # find log_probs and re-apply mask
        log_probs = self._compute_log_probs(observations) # (N, T, K)

        log_responsibilities = F.log_softmax(log_probs, dim=-1)

        return log_responsibilities.masked_fill(~mask.unsqueeze(-1), float('-inf'))


    def _m_step(self, observations, component_responsibilities, state_responsibilities):
        """Update parameters given component responsibilities, as well as GMM state occupancy responsibilities:
        N = batch_size, T = observation length, K = n_components, F = n_features"""          

        # combine responsibilities
        responsibilities = component_responsibilities + state_responsibilities.unsqueeze(-1) # (N, T, K)

        # update mixing coefficients
        N_k = torch.exp(torch.logsumexp(responsibilities, dim=(0, 1))) # (K,)
        self.mixture_weights.data = N_k / N_k.sum() # (K,)

        # update means
        responsibilities = torch.exp(responsibilities).unsqueeze(-1) # (N, T, K, 1)
        observations = observations.unsqueeze(2) # (N, T, 1, F)

        weighted_observations = responsibilities * observations # (N, T, K, F)
        means_numerators = weighted_observations.sum(dim=(0, 1)) # (K, F)

        self.means.data = means_numerators / N_k.unsqueeze(-1) # (K, F)


        # update log variances
        centered = observations - self.means.unsqueeze(0).unsqueeze(0) # (N, T, K, F)
        centered = centered ** 2
        weighted_centered = responsibilities * centered # (N, T, K, F)

        vars_numerators = weighted_centered.sum(dim=(0, 1)) # (K, F)

        variances = vars_numerators / N_k.unsqueeze(-1) # (K, F)
        self.log_vars.data = variances.clamp(min=1e-6).log() # (K, F)       


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """emit log-likelihood of observations"""
        # if not batched, add batch dim
        if x.dim() == 1:
            x = x.unsqueeze(0)

        log_probs = self._compute_log_probs(x) # (N, T, K)
        return torch.logsumexp(log_probs, dim=2) # (N, T)

    
    def _compute_log_probs(self, data: torch.Tensor) -> torch.Tensor:
        """given tensor (N, T, F),
        compute log probabilities for each component"""
        log_probs = []

        # create mask
        mask = (data != 0).any(dim=-1)

        for k in range(self.n_components):
            diff = data - self.means[k]
            vars = torch.exp(self.log_vars[k]) + 1e-8
            
            # multivariate gaussian formula in log space
            mahalanobis_distance = torch.sum(diff**2 / vars, dim=-1)
            covariance_determinant = torch.sum(self.log_vars[k])
            normalization_constant = self.n_features * np.log(2 * np.pi)
            
            log_prob = -0.5 * (mahalanobis_distance + covariance_determinant +
                               normalization_constant)
            
            log_prob += torch.log(self.mixture_weights[k] + 1e-8)
            
            # apply mask
            log_prob = log_prob.masked_fill(~mask, float('-inf'))
            
            log_probs.append(log_prob)

        return torch.stack(log_probs, dim=-1) # (N, T, K)


class HMMState:
    def __init__(self, id: int, n_components: int = 3, n_features: int = 80,
                 inner_epochs: int = 10):
        self.id = id
        self.transitions = {}
        self.gmm = GMM(n_components, n_features, inner_epochs)

    def __repr__(self):
        return f"(State {self.id})"

    def add_transition(self, next_state: int, log_prob: float):
        self.transitions[next_state] = log_prob

    def emission_probs(self, observations: torch.Tensor) -> torch.Tensor:
        return self.gmm(observations) # (N, T)


class HMM:
    def __init__(self, states: List[HMMState]):
        self.states = states
        self.n_states = len(self.states)
        self._set_transitions()

    def __repr__(self) -> str:
        s = f"["
        for _, state in self.states.items():
            s += f"{state} > "
        if s:
            s = s[:-3] + f"]"
        return s
    
    def _set_transitions(self):
        transitions = []
        for i in range(self.n_states):
            row = torch.full((self.n_states,), float('-inf'))

            for id, log_prob in self.states[i].transitions.items():
                row[id] = log_prob
            
            transitions.append(row)
        self.transitions = torch.stack(transitions, dim=0)


    def fit(self, observations: torch.Tensor, state_responsibilities: torch.Tensor):
        log_likelihoods = []
        for i in range(self.n_states):
            log_likelihood = self.states[i].gmm.fit(observations, state_responsibilities[:, :, i])
            log_likelihoods.append(log_likelihood)
        return f"state log likelihoods: {tuple([round(x, 3) for x in log_likelihoods])}"


class WordRecognizer:
    def __init__(self, from_saved: bool, id: str = None,
                 models: Dict[str, HMM] = None,  path: str = "saved"):
        self.path = path
        
        if from_saved:
            self.load_saved(f"{path}/{id}.pt")

        else:
            self.models = models
            self.generate_id()


    def load_saved(self, id: str):
        self.models = {}
        self.id = id

    def save_model(self):
        pass

    def generate_id(self):
        self.id = adjectiveanimalnumber.generate()

        while os.path.isdir(os.path.join(self.path, self.id)):
            self.id = adjectiveanimalnumber.generate()


if __name__ == "__main__":
    recognizers = []

    import torch
    torch.manual_seed(0)

    for _ in range(5):
        recognizers.append(WordRecognizer(from_saved=False))
    print([recognizer.id for recognizer in recognizers])