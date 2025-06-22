import torch
from torch import nn
import numpy as np
from typing import List, Dict, Tuple, Set
# from torch.distributions import Categorical, Normal, MixtureSameFamily


class GMM(nn.Module):
    def __init__(self, n_components: int, n_features: int):
        super().__init__()
        self.n_components = n_components
        self.n_features= n_features

        # initialize
        self.mixture_weights = nn.Parameter(torch.ones(n_components) / n_components) # 1d, Uniform
        self.means = nn.Parameter(torch.randn(n_components, n_features) * 0.1) # 2d, Random
        self.log_variances = nn.Parameter(torch.zeros(n_components, n_features)) # 2d, Zero

    def fit(self, observations: torch.Tensor, responsibilities: torch.Tensor, 
            inner_iterations: int = 10, eps: float = 1e-6):
        """Fit GMM with inner EM algorithm"""
        prev_log_likelihood = float('-inf')

        for iteration in range(inner_iterations):
            self._e_step(observations, responsibilities)

            self._m_step()

            current_log_likelihood = self(observations).sum.item()
            
            if abs(current_log_likelihood - prev_log_likelihood) < eps:
                # print(f"converged at iteration {iteration+1}")
                break

    def _e_step(self, observations, responsibilities):
        pass
    def _m_step(self):
        """Update parameters"""
        pass

    def forward(self, x) -> torch.Tensor:
        """Compute log-likelihood of observations"""
        return None



class HMMState:
    def __init__(self, id: int, n_components: int = 3, n_features: int = 39):
        self.id = id
        self.transitions = {}
        self.emissions = GMM(n_components, n_features)

    def add_transition(self, next_state: int, log_prob: float):
        self.transitions[next_state] = log_prob

    def emission_prob(self, observation: torch.Tensor) -> float:
        return self.emissions(observation).item()


class HMM:
    def __init__(self, states: List[HMMState], start_id: int = 0):
        self.states = {state.id: state for state in states}
        self.start = self.states[start_id]
        self.n_states = len(self.states)