#!/usr/bin/env python3
"""
HMM/GMM ASR System Framework
A foundational implementation for learning ASR fundamentals

Written by Claude
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import scipy.io.wavfile as wav
from typing import List, Tuple, Dict, Optional
import pickle
import json
from pathlib import Path

class FeatureExtractor:
    """Extract MFCC features with delta and delta-delta coefficients"""
    
    def __init__(self, n_mfcc=13, n_fft=512, hop_length=160, win_length=400, 
                 sample_rate=16000, n_mels=40):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.n_mels = n_mels
    
    def extract_mfcc(self, audio_path: str) -> torch.Tensor:
        """Extract MFCC features from audio file"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length,
            n_mels=self.n_mels
        )
        
        # Compute deltas
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack features (39-dimensional: 13 MFCC + 13 delta + 13 delta-delta)
        features = np.vstack([mfcc, delta, delta2])
        
        return torch.FloatTensor(features.T)  # Shape: [time, features]
    
    def apply_cmn(self, features: torch.Tensor) -> torch.Tensor:
        """Apply Cepstral Mean Normalization"""
        return features - features.mean(dim=0, keepdim=True)

class GMM(nn.Module):
    """Gaussian Mixture Model implementation in PyTorch"""
    
    def __init__(self, n_components: int, n_features: int):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        
        # Parameters
        self.weights = nn.Parameter(torch.ones(n_components) / n_components)
        self.means = nn.Parameter(torch.randn(n_components, n_features))
        self.log_vars = nn.Parameter(torch.zeros(n_components, n_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood of observations
        Args:
            x: [batch_size, n_features] or [n_features]
        Returns:
            log_likelihood: scalar or [batch_size]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Compute log probabilities for each component
        log_probs = []
        for k in range(self.n_components):
            # Multivariate Gaussian log probability (diagonal covariance)
            diff = x - self.means[k]
            vars = torch.exp(self.log_vars[k])
            log_prob = -0.5 * (
                torch.sum(diff**2 / vars, dim=-1) +
                torch.sum(self.log_vars[k]) +
                self.n_features * np.log(2 * np.pi)
            )
            log_probs.append(log_prob + torch.log(self.weights[k]))
        
        # Stack and use logsumexp for numerical stability
        log_probs = torch.stack(log_probs, dim=-1)  # [batch_size, n_components]
        return torch.logsumexp(log_probs, dim=-1)
    
    def fit(self, data: torch.Tensor, n_iter: int = 100, tol: float = 1e-6):
        """Fit GMM using EM algorithm"""
        for iteration in range(n_iter):
            old_log_likelihood = self.forward(data).sum()
            
            # E-step: compute responsibilities
            with torch.no_grad():
                responsibilities = self._e_step(data)
            
            # M-step: update parameters
            self._m_step(data, responsibilities)
            
            # Check convergence
            new_log_likelihood = self.forward(data).sum()
            if abs(new_log_likelihood - old_log_likelihood) < tol:
                print(f"Converged after {iteration + 1} iterations")
                break
    
    def _e_step(self, data: torch.Tensor) -> torch.Tensor:
        """E-step: compute responsibilities"""
        log_probs = []
        for k in range(self.n_components):
            diff = data - self.means[k]
            vars = torch.exp(self.log_vars[k])
            log_prob = -0.5 * (
                torch.sum(diff**2 / vars, dim=-1) +
                torch.sum(self.log_vars[k]) +
                self.n_features * np.log(2 * np.pi)
            )
            log_probs.append(log_prob + torch.log(self.weights[k]))
        
        log_probs = torch.stack(log_probs, dim=-1)
        # Convert to responsibilities using softmax
        responsibilities = torch.softmax(log_probs, dim=-1)
        return responsibilities
    
    def _m_step(self, data: torch.Tensor, responsibilities: torch.Tensor):
        """M-step: update parameters"""
        N_k = responsibilities.sum(dim=0)  # [n_components]
        
        # Update weights
        self.weights.data = N_k / data.shape[0]
        
        # Update means
        for k in range(self.n_components):
            self.means[k].data = (responsibilities[:, k:k+1] * data).sum(dim=0) / N_k[k]
        
        # Update variances
        for k in range(self.n_components):
            diff = data - self.means[k]
            weighted_var = (responsibilities[:, k:k+1] * diff**2).sum(dim=0) / N_k[k]
            self.log_vars[k].data = torch.log(weighted_var + 1e-6)  # Add small epsilon

class HMMState:
    """Single HMM state with GMM emission model"""
    
    def __init__(self, state_id: int, n_components: int, n_features: int):
        self.state_id = state_id
        self.gmm = GMM(n_components, n_features)
        self.transitions = {}  # {next_state_id: log_prob}
    
    def add_transition(self, next_state: int, log_prob: float):
        """Add transition to another state"""
        self.transitions[next_state] = log_prob
    
    def emission_prob(self, observation: torch.Tensor) -> float:
        """Compute emission probability for observation"""
        return self.gmm(observation).item()

class HMM:
    """Hidden Markov Model for ASR"""
    
    def __init__(self, states: List[HMMState], start_state: int = 0):
        self.states = {state.state_id: state for state in states}
        self.start_state = start_state
        self.n_states = len(states)
    
    def viterbi_decode(self, observations: torch.Tensor) -> List[int]:
        """Viterbi algorithm for finding best state sequence"""
        T = observations.shape[0]  # sequence length
        
        # Initialize
        delta = torch.full((T, self.n_states), float('-inf'))
        psi = torch.zeros((T, self.n_states), dtype=torch.int)
        
        # Initial step
        for s in self.states:
            if s == self.start_state:
                delta[0, s] = self.states[s].emission_prob(observations[0])
        
        # Forward pass
        for t in range(1, T):
            for s in self.states:
                # Find best previous state
                best_score = float('-inf')
                best_prev = 0
                
                for prev_s in self.states:
                    if s in self.states[prev_s].transitions:
                        score = (delta[t-1, prev_s] + 
                                self.states[prev_s].transitions[s] +
                                self.states[s].emission_prob(observations[t]))
                        if score > best_score:
                            best_score = score
                            best_prev = prev_s
                
                delta[t, s] = best_score
                psi[t, s] = best_prev
        
        # Backward pass - find best path
        path = []
        # Find best final state
        best_final = torch.argmax(delta[T-1]).item()
        path.append(best_final)
        
        # Trace back
        for t in range(T-1, 0, -1):
            path.append(psi[t, path[-1]].item())
        
        return path[::-1]

class SimpleASR:
    """Simple ASR system combining HMM/GMM models"""
    
    def __init__(self, vocabulary: List[str], n_components: int = 4, n_features: int = 39):
        self.vocabulary = vocabulary
        self.word_models = {}  # {word: HMM}
        self.feature_extractor = FeatureExtractor()
        self.n_components = n_components
        self.n_features = n_features
        
        # Create simple HMM topology for each word (3 states per word)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize HMM models for each word"""
        for word in self.vocabulary:
            # Create 3-state left-to-right HMM for each word
            states = []
            for i in range(3):
                state = HMMState(i, self.n_components, self.n_features)
                states.append(state)
            
            # Add transitions (left-to-right topology)
            states[0].add_transition(0, np.log(0.5))  # self-loop
            states[0].add_transition(1, np.log(0.5))  # forward
            states[1].add_transition(1, np.log(0.5))  # self-loop
            states[1].add_transition(2, np.log(0.5))  # forward
            states[2].add_transition(2, np.log(1.0))  # self-loop (final state)
            
            self.word_models[word] = HMM(states, start_state=0)
    
    def train_word_model(self, word: str, audio_files: List[str]):
        """Train HMM/GMM model for a specific word"""
        print(f"Training model for word: {word}")
        
        # Extract features from all training files
        all_features = []
        for audio_file in audio_files:
            features = self.feature_extractor.extract_mfcc(audio_file)
            features = self.feature_extractor.apply_cmn(features)
            all_features.append(features)
        
        # Concatenate all features for initial GMM training
        combined_features = torch.cat(all_features, dim=0)
        
        # Train GMM for each state (simple approach: use all data for all states)
        for state in self.word_models[word].states.values():
            print(f"  Training state {state.state_id}")
            state.gmm.fit(combined_features)
    
    def recognize(self, audio_file: str) -> str:
        """Recognize word from audio file"""
        # Extract features
        features = self.feature_extractor.extract_mfcc(audio_file)
        features = self.feature_extractor.apply_cmn(features)
        
        # Score against all word models
        scores = {}
        for word, model in self.word_models.items():
            # Simple scoring: sum of emission probabilities
            total_score = 0
            for t in range(features.shape[0]):
                # Use average emission probability across all states
                state_scores = []
                for state in model.states.values():
                    state_scores.append(state.emission_prob(features[t]))
                total_score += np.mean(state_scores)
            
            scores[word] = total_score
        
        # Return word with highest score
        return max(scores, key=scores.get)
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'vocabulary': self.vocabulary,
            'n_components': self.n_components,
            'n_features': self.n_features
        }
        
        # Save model parameters
        for word, model in self.word_models.items():
            model_data[f'{word}_states'] = {}
            for state_id, state in model.states.items():
                model_data[f'{word}_states'][state_id] = {
                    'gmm_weights': state.gmm.weights.detach().numpy(),
                    'gmm_means': state.gmm.means.detach().numpy(),
                    'gmm_log_vars': state.gmm.log_vars.detach().numpy(),
                    'transitions': state.transitions
                }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocabulary = model_data['vocabulary']
        self.n_components = model_data['n_components']
        self.n_features = model_data['n_features']
        
        # Restore models
        self._initialize_models()
        for word in self.vocabulary:
            states_data = model_data[f'{word}_states']
            for state_id, state_data in states_data.items():
                state = self.word_models[word].states[state_id]
                state.gmm.weights.data = torch.FloatTensor(state_data['gmm_weights'])
                state.gmm.means.data = torch.FloatTensor(state_data['gmm_means'])
                state.gmm.log_vars.data = torch.FloatTensor(state_data['gmm_log_vars'])
                state.transitions = state_data['transitions']

# Example usage and testing
if __name__ == "__main__":
    # Example: Create a simple ASR system for digits
    vocabulary = ["zero", "one", "two", "three", "four", "five"]
    asr_system = SimpleASR(vocabulary, n_components=2, n_features=39)
    
    print("ASR System initialized!")
    print(f"Vocabulary: {vocabulary}")
    print(f"Feature dimensions: {asr_system.n_features}")
    print(f"GMM components per state: {asr_system.n_components}")
    
    # Example feature extraction (you'll need actual audio files)
    # features = asr_system.feature_extractor.extract_mfcc("path/to/audio.wav")
    # print(f"Feature shape: {features.shape}")
    
    print("\nTo use this system:")
    print("1. Prepare audio files organized by word")
    print("2. Train each word model:")
    print("   asr_system.train_word_model('zero', ['zero1.wav', 'zero2.wav', ...])")
    print("3. Recognize new audio:")
    print("   result = asr_system.recognize('test_audio.wav')")
    print("4. Save/load models:")
    print("   asr_system.save_model('model.pkl')")
    print("   asr_system.load_model('model.pkl')")