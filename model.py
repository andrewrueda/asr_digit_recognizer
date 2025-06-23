import torch
from torch import nn
import numpy as np
from typing import List, Dict, Tuple, Set, Union
import torch.nn.functional as F
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

            self._m_step(observations, component_responsibilities, state_responsibilities)

            current_log_likelihood = self(observations).item()

            print(f"Iteration {iteration+1}: Log Likelihood = {current_log_likelihood:.2f}")

            if abs(current_log_likelihood - prev_log_likelihood) < eps:
                print(f"converged at iteration {iteration+1}")
                break

    def _e_step(self, observations):
        """Compute component responsibilities"""
        # create mask
        mask = (observations != 0).any(dim=-1)

        # find log_probs and re-apply mask
        log_probs = self._compute_log_probs(observations)
        soft = F.log_softmax(log_probs, dim=-1)
        return soft.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        
    def _m_step(self, observations, component_responsibilities, state_responsibilities):
        """Update parameters given component responsibilities, as well as GMM state occupancy responsibilities:
        N = batch_size, T = observation length, K = n_components, F = n_features"""          

        # combine responsibilities
        component_responsibilities = torch.exp(component_responsibilities) # (N, T, K)
        responsibilities = component_responsibilities * state_responsibilities.unsqueeze(-1) # (N, T, K)

        # update mixing coefficients
        N_k = responsibilities.sum(dim=(0, 1)) # (K,)
        self.mixture_weights.data = N_k / N_k.sum() # (K,)

        # update means
        responsibilities = responsibilities.unsqueeze(-1) # (N, T, K, 1)
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


        # # ensure non-negativity
        # self.mixture_weights.data =  F.softmax(self.mixture_weights, dim=0)

        # self.means.data = torch.clamp(self.means, min=-10.0, max=10.0)
        # self.log_vars.data = torch.clamp(self.log_vars, min=-10.0, max=10.0)          


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """emit log-likelihood of observations"""
        # if not batched, add batch dim
        if x.dim() == 1:
            x = x.unsqueeze(0)

        log_probs = self._compute_log_probs(x)
        return torch.logsumexp(log_probs, dim=(0, 1, 2))

    
    def _compute_log_probs(self, data: torch.Tensor) -> torch.Tensor:
        """given tensor [batch_size, T, n_features],
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
            
            log_prob += self.mixture_weights[k] + 1e-8
            
            # apply mask
            log_prob = log_prob.masked_fill(~mask, float('-inf'))
            
            log_probs.append(log_prob)

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
        for _, state in self.states.items():
            s += f"{state} > "
        if s:
            s = s[:-3] + f"]"
        return s


if __name__ == "__main__":
    gmm = GMM(n_components = 3, n_features = 80)

    import random
    from torch.nn.utils.rnn import pad_sequence

    observations = []
    for _ in range(20):
        n = random.randint(26, 30)
        observations.append(torch.randn(n, 80))

    observations = sorted(observations, key=lambda x: x.shape[0]) 
    observations = pad_sequence(observations, batch_first = True)

    state_responsibilities = torch.randn(20, 30) ** 2
    state_responsibilities = state_responsibilities / state_responsibilities.sum(dim=1, keepdim=True)

    gmm.fit(observations=observations, state_responsibilities=state_responsibilities)

