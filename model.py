import torch
from torch import nn
import numpy as np
from typing import List, Dict, Tuple, Set, Union
# from torch.distributions import Categorical, Normal, MixtureSameFamily


class GMM(torch.nn.Module):
    def __init__(self, n_components: int, n_features: int):
        super().__init__()
        self.n_components = n_components
        self.n_features= n_features

        # initialize
        self.mixture_weights = nn.Parameter(torch.ones(n_components) / n_components) # 1d, Uniform
        self.means = nn.Parameter(torch.randn(n_components, n_features) * 0.1) # 2d, Random
        self.log_vars = nn.Parameter(torch.zeros(n_components, n_features)) # 2d, Zero

    def fit(self, observations: torch.Tensor, state_responsibilities: torch.Tensor, 
            inner_iterations: int = 10, eps: float = 1e-6):
        """Fit GMM with inner EM algorithm"""
        prev_log_likelihood = float('-inf')

        for iteration in range(inner_iterations):
            component_responsibilities = self._e_step(observations)

            self._m_step(component_responsibilities, state_responsibilities)

            current_log_likelihood = self(observations).sum.item()
            
            if abs(current_log_likelihood - prev_log_likelihood) < eps:
                # print(f"converged at iteration {iteration+1}")
                break

    def _e_step(self, observations):
        """Compute component responsibilities"""
        log_probs = self._compute_log_probs(observations)
        return torch.softmax(log_probs, dim=-1)

        
    def _m_step(self, component_responsibilities, state_responsibilities):
        """Update parameters"""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """emit log-likelihood of observations"""
        # if not batched, add batch dim
        if x.dim() == 1:
            x = x.unsqueeze(0)

        log_probs = self._compute_log_probs(x)
        return torch.logsumexp(log_probs, dim=-1)

    
    def _compute_log_probs(self, data: torch.Tensor) -> torch.Tensor:
        """given tensor [batch_size, n_features],
        compute log probabilities for each component"""
        log_probs = []
        for k in range(self.n_components):
            diff = data - self.means[k]
            vars = torch.exp(self.log_vars[k]) + 1e-8
            
            # multivariate gaussian formula in log space
            mahalanobis_distance = torch.sum(diff**2 / vars, dim=-1)
            covariance_determinant = torch.sum(self.log_vars[k])
            normalization_constant = self.n_features * np.log(2 * np.pi)
            
            log_prob = -0.5 * (mahalanobis_distance + covariance_determinant +
                               normalization_constant)
            
            log_probs.append(log_prob + self.mixture_weights[k] + 1e-8)

        return torch.stack(log_probs, dim=-1)



class HMMState:
    def __init__(self, id: int, n_components: int = 3, n_features: int = 80):
        self.id = id
        self.transitions = {}
        self.emissions = GMM(n_components, n_features)

    def __repr__(self):
        return f"(State {self.id})"

    def add_transition(self, next_state: int, log_prob: float):
        self.transitions[next_state] = log_prob

    def emission_prob(self, observation: torch.Tensor) -> float:
        return self.emissions(observation).item()



class HMM:
    def __init__(self, states: List[HMMState], start_id: int = 0):
        self.states = {state.id: state for state in states}
        self.start = self.states[start_id]
        self.n_states = len(self.states)
    def __repr__(self) -> str:
        s = f"["
        for id, state in self.states.items():
            s += f"{state} > "
        if s:
            s = s[:-2] + f"]"
        return s

