#!/usr/bin/env python3
"""
ESPnet2-Integrated HMM/GMM ASR System
Combines classical HMM/GMM with modern ESPnet2 infrastructure

Written by Claude
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import List, Tuple, Dict, Optional, Union
import pickle
import json
import yaml
from pathlib import Path
import kaldiio
import soundfile as sf

# ESPnet2 imports
try:
    from espnet2.asr.frontend.default import DefaultFrontend
    from espnet2.asr.frontend.fused import FusedFrontends
    from espnet2.asr.specaug.specaug import SpecAug
    from espnet2.layers.log_mel import LogMel
    from espnet2.utils.types import str2bool
    from espnet2.train.dataset import ESPnetDataset
    from espnet2.fileio.datadir_writer import DatadirWriter
    ESPNET_AVAILABLE = True
except ImportError:
    print("ESPnet2 not available. Install with: pip install espnet")
    ESPNET_AVAILABLE = False

class ESPnetFeatureExtractor:
    """Feature extraction using ESPnet2 frontend"""
    
    def __init__(self, 
                 fs: int = 16000,
                 n_fft: int = 512,
                 n_mels: int = 80,
                 hop_length: int = 160,
                 win_length: int = 400,
                 fmin: int = 80,
                 fmax: int = 7600,
                 frontend_type: str = "default",
                 use_specaug: bool = False):
        
        if not ESPNET_AVAILABLE:
            raise ImportError("ESPnet2 not available. Please install espnet.")
        
        self.fs = fs
        self.frontend_type = frontend_type
        
        # Configure frontend
        if frontend_type == "default":
            self.frontend = DefaultFrontend(
                fs=fs,
                n_fft=n_fft,
                n_mels=n_mels,
                hop_length=hop_length,
                win_length=win_length,
                fmin=fmin,
                fmax=fmax,
            )
        elif frontend_type == "fused":
            frontends = [
                LogMel(
                    fs=fs,
                    n_fft=n_fft,
                    n_mels=n_mels,
                    hop_length=hop_length,
                    win_length=win_length,
                    fmin=fmin,
                    fmax=fmax,
                )
            ]
            self.frontend = FusedFrontends(
                frontends=frontends,
                align_method="linear_projection",
            )
        
        # Optional SpecAugment for training
        self.specaug = None
        if use_specaug:
            self.specaug = SpecAug(
                apply_time_warp=True,
                time_warp_window=5,
                time_warp_mode="bicubic",
                apply_freq_mask=True,
                freq_mask_width_range=[0, 30],
                num_freq_mask=2,
                apply_time_mask=True,
                time_mask_width_range=[0, 40],
                num_time_mask=2,
            )
        
        self.feature_dim = n_mels
    
    def extract_features(self, 
                        audio_path: str, 
                        apply_cmn: bool = True,
                        apply_specaug: bool = False) -> torch.Tensor:
        """Extract features using ESPnet2 frontend"""
        
        # Load audio using soundfile (ESPnet2 standard)
        speech, sample_rate = sf.read(audio_path)
        
        # Convert to tensor
        speech = torch.FloatTensor(speech)
        
        # Resample if necessary
        if sample_rate != self.fs:
            speech = librosa.resample(speech.numpy(), orig_sr=sample_rate, target_sr=self.fs)
            speech = torch.FloatTensor(speech)
    
        # Add batch dimension
        speech = speech.unsqueeze(0)  # [1, time]
        speech_lengths = torch.LongTensor([speech.shape[1]])
        
        # Extract features: feats contains [batch, time, features]
        feats, feat_lengths = self.frontend(speech, speech_lengths)
        
        # Apply SpecAugment during training
        if apply_specaug and self.specaug is not None:
            feats, feat_lengths = self.specaug(feats, feat_lengths)
        
        # Remove batch dimension
        feats = feats.squeeze(0)  # [time, features]
        
        # Apply cepstral mean normalization
        if apply_cmn:
            feats = feats - feats.mean(dim=0, keepdim=True)
        
        return feats
    
    def save_config(self, config_path: str):
        """Save feature extraction configuration"""
        config = {
            'fs': self.fs,
            'frontend_type': self.frontend_type,
            'feature_dim': self.feature_dim,
        }
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

class ESPnetDataManager:
    """Handle data in ESPnet2 format for HMM/GMM training"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
    
    def prepare_kaldi_style_data(self, 
                                audio_files: Dict[str, List[str]], 
                                output_dir: str = "data"):
        """
        Prepare data in Kaldi/ESPnet format
        Args:
            audio_files: {word: [list_of_audio_files]}
            output_dir: output directory name
        """
        output_path = self.data_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        # Create wav.scp, utt2spk, spk2utt, and text files
        wav_scp = []
        utt2spk = []
        text = []
        
        for word, files in audio_files.items():
            for i, audio_file in enumerate(files):
                utt_id = f"{word}_{i:03d}"
                spk_id = f"spk_{word}"
                
                wav_scp.append(f"{utt_id} {Path(audio_file).absolute()}")
                utt2spk.append(f"{utt_id} {spk_id}")
                text.append(f"{utt_id} {word}")
        
        # Write files
        self._write_file(output_path / "wav.scp", wav_scp)
        self._write_file(output_path / "utt2spk", utt2spk)
        self._write_file(output_path / "text", text)
        
        # Create spk2utt from utt2spk
        self._create_spk2utt(output_path / "utt2spk", output_path / "spk2utt")
        
        return output_path
    
    def _write_file(self, filepath: Path, lines: List[str]):
        """Write lines to file"""
        with open(filepath, 'w') as f:
            for line in lines:
                f.write(line + '\n')
    
    def _create_spk2utt(self, utt2spk_file: Path, spk2utt_file: Path):
        """Create spk2utt from utt2spk"""
        spk2utt = {}
        with open(utt2spk_file, 'r') as f:
            for line in f:
                utt, spk = line.strip().split()
                if spk not in spk2utt:
                    spk2utt[spk] = []
                spk2utt[spk].append(utt)
        
        with open(spk2utt_file, 'w') as f:
            for spk, utts in spk2utt.items():
                f.write(f"{spk} {' '.join(utts)}\n")
    
    def load_data_from_scp(self, wav_scp: str, text_file: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Load data from Kaldi-style scp files
        Returns: {word: [(utt_id, audio_path), ...]}
        """
        # Load wav.scp
        wav_dict = {}
        with open(wav_scp, 'r') as f:
            for line in f:
                utt_id, wav_path = line.strip().split(None, 1)
                wav_dict[utt_id] = wav_path
        
        # Load text file and organize by word
        word_data = {}
        with open(text_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                utt_id = parts[0]
                word = ' '.join(parts[1:])  # Handle multi-word labels
                
                if word not in word_data:
                    word_data[word] = []
                
                if utt_id in wav_dict:
                    word_data[word].append((utt_id, wav_dict[utt_id]))
        
        return word_data

class GMM(nn.Module):
    """Gaussian Mixture Model implementation in PyTorch"""
    
    def __init__(self, n_components: int, n_features: int): 
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        
        # Parameters
        self.weights = nn.Parameter(torch.ones(n_components) / n_components)
        self.means = nn.Parameter(torch.randn(n_components, n_features) * 0.1)
        self.log_vars = nn.Parameter(torch.zeros(n_components, n_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-likelihood of observations"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        log_probs = self._compute_log_probabilities(x)
        return torch.logsumexp(log_probs, dim=-1)
    
    def fit(self, data: torch.Tensor, n_iter: int = 100, tol: float = 1e-6):
        """Fit GMM using EM algorithm"""
        print(f"    Training GMM with {data.shape[0]} frames, {self.n_components} components")
        
        prev_log_likelihood = float('-inf')
        
        for iteration in range(n_iter):
            # E-step
            with torch.no_grad():
                responsibilities = self._e_step(data)
            
            # M-step
            self._m_step(data, responsibilities)
            
            # Check convergence
            current_log_likelihood = self.forward(data).sum().item()
            
            if iteration % 10 == 0:
                print(f"      Iteration {iteration}, Log-likelihood: {current_log_likelihood:.2f}")
            
            if abs(current_log_likelihood - prev_log_likelihood) < tol:
                print(f"      Converged after {iteration + 1} iterations")
                break
            
            prev_log_likelihood = current_log_likelihood
    
    def _e_step(self, data: torch.Tensor) -> torch.Tensor:
        """E-step: compute responsibilities"""
        log_probs = self._compute_log_probabilities(data)
        return torch.softmax(log_probs, dim=-1)
    
    def _m_step(self, data: torch.Tensor, responsibilities: torch.Tensor):
        """M-step: update parameters"""
        N_k = responsibilities.sum(dim=0) + 1e-8
        
        # Update weights
        self.weights.data = N_k / data.shape[0]
        
        # Update means and variances
        for k in range(self.n_components):
            # Update means
            weighted_sum = (responsibilities[:, k:k+1] * data).sum(dim=0)
            self.means[k].data = weighted_sum / N_k[k]
            
            # Update variances
            diff = data - self.means[k]
            weighted_var = (responsibilities[:, k:k+1] * diff**2).sum(dim=0) / N_k[k]
            self.log_vars[k].data = torch.log(weighted_var + 1e-6)

    def _compute_log_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """compute log probabilities"""
        log_probs = []
        for k in range(self.n_components):
            diff = x - self.means[k]
            vars = torch.exp(self.log_vars[k]) + 1e-6
            log_prob = -0.5 * (
                torch.sum(diff**2 / vars, dim=-1) +
                torch.sum(self.log_vars[k]) +
                self.n_features * np.log(2 * np.pi)
            )
            log_probs.append(log_prob + torch.log(self.weights[k] + 1e-8))
        return torch.stack(log_probs, dim=-1)


class HMMState:
    """HMM state with GMM emission model"""
    
    def __init__(self, state_id: int, n_components: int, n_features: int):
        self.state_id = state_id
        self.gmm = GMM(n_components, n_features)
        self.transitions = {}
    
    def add_transition(self, next_state: int, log_prob: float):
        self.transitions[next_state] = log_prob
    
    def emission_prob(self, observation: torch.Tensor) -> float:
        return self.gmm(observation).item()

class HMM:
    """Hidden Markov Model for ASR"""
    
    def __init__(self, states: List[HMMState], start_state: int = 0):
        self.states = {state.state_id: state for state in states}
        self.start_state = start_state
        self.n_states = len(states)

class ESPnetHMMGMM:
    """ESPnet2-integrated HMM/GMM ASR system"""
    
    def __init__(self, 
                 vocabulary: List[str],
                 n_components: int = 4,
                 n_states_per_word: int = 3,
                 frontend_config: Dict = None):
        
        self.vocabulary = vocabulary
        self.word_models = {}
        self.n_components = n_components
        self.n_states_per_word = n_states_per_word
        
        # Initialize ESPnet2 feature extractor
        if frontend_config is None:
            frontend_config = {
                'fs': 16000,
                'n_mels': 80,
                'hop_length': 160,
                'win_length': 400,
            }
        
        self.feature_extractor = ESPnetFeatureExtractor(**frontend_config)
        self.n_features = self.feature_extractor.feature_dim
        
        # Initialize data manager
        self.data_manager = ESPnetDataManager("./data")
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize HMM models for each word"""
        for word in self.vocabulary:
            states = []
            for i in range(self.n_states_per_word):
                state = HMMState(i, self.n_components, self.n_features)
                states.append(state)
            
            # Left-to-right topology
            for i in range(self.n_states_per_word):
                if i < self.n_states_per_word - 1:
                    states[i].add_transition(i, np.log(0.5))      # self-loop
                    states[i].add_transition(i + 1, np.log(0.5))  # forward
                else:
                    states[i].add_transition(i, np.log(1.0))      # final state self-loop
            
            self.word_models[word] = HMM(states, start_state=0)
    
    def prepare_training_data(self, audio_files: Dict[str, List[str]]) -> str:
        """Prepare training data in ESPnet format"""
        print("Preparing training data in ESPnet format...")
        data_dir = self.data_manager.prepare_kaldi_style_data(audio_files, "train")
        print(f"Data prepared in: {data_dir}")
        return str(data_dir)
    
    def train_from_data_dir(self, data_dir: str):
        """Train models from ESPnet-style data directory"""
        print("Loading training data...")
        
        wav_scp = Path(data_dir) / "wav.scp"
        text_file = Path(data_dir) / "text"
        
        word_data = self.data_manager.load_data_from_scp(str(wav_scp), str(text_file))
        
        # Train each word model
        for word in self.vocabulary:
            if word in word_data:
                print(f"Training model for word: {word}")
                self._train_word_model(word, word_data[word])
            else:
                print(f"Warning: No training data found for word: {word}")
    
    def _train_word_model(self, word: str, utterances: List[Tuple[str, str]]):
        """Train HMM/GMM model for a specific word"""

        # NOTE: don't do this, do alignment
        
        # Extract features from all utterances
        all_features = []
        
        for utt_id, audio_path in utterances:
            try:
                features = self.feature_extractor.extract_features(
                    audio_path, 
                    apply_cmn=True,
                    apply_specaug=False  # No augmentation during training for HMM/GMM
                )
                all_features.append(features)
                print(f"  Processed {utt_id}: {features.shape}")
            except Exception as e:
                print(f"  Error processing {utt_id}: {e}")
                continue
        
        if not all_features:
            print(f"  No valid features extracted for {word}")
            return
        
        # Concatenate all features
        combined_features = torch.cat(all_features, dim=0)
        print(f"  Combined features shape: {combined_features.shape}")
        
        # Train GMM for each state
        for state_id, state in self.word_models[word].states.items():
            print(f"  Training state {state_id}")
            
            # For simplicity, use all data for all states
            # In practice, you'd use forced alignment or uniform segmentation
            state.gmm.fit(combined_features, n_iter=50)
    
    def recognize(self, audio_path: str) -> str:
        """Recognize word from audio file"""
        # Extract features
        features = self.feature_extractor.extract_features(audio_path, apply_cmn=True)
        
        # Score against all word models
        scores = {}
        for word, model in self.word_models.items():
            total_score = 0
            for t in range(features.shape[0]):
                # Simple scoring: average emission probability across states
                state_scores = []
                for state in model.states.values():
                    state_scores.append(state.emission_prob(features[t]))
                total_score += np.mean(state_scores)
            
            scores[word] = total_score / features.shape[0]  # Normalize by length
        
        return max(scores, key=scores.get)
    
    def save_model(self, model_dir: str):
        """Save trained model with ESPnet2 compatibility"""
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True, parents=True)
        
        # Save feature extraction config
        self.feature_extractor.save_config(str(model_path / "frontend_config.yaml"))
        
        # Save model parameters
        model_data = {
            'vocabulary': self.vocabulary,
            'n_components': self.n_components,
            'n_states_per_word': self.n_states_per_word,
            'n_features': self.n_features
        }
        
        # Save HMM/GMM parameters
        for word, model in self.word_models.items():
            model_data[f'{word}_states'] = {}
            for state_id, state in model.states.items():
                model_data[f'{word}_states'][state_id] = {
                    'gmm_weights': state.gmm.weights.detach().numpy(),
                    'gmm_means': state.gmm.means.detach().numpy(),
                    'gmm_log_vars': state.gmm.log_vars.detach().numpy(),
                    'transitions': state.transitions
                }
        
        with open(model_path / "model.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_dir: str):
        """Load trained model"""
        model_path = Path(model_dir)
        
        with open(model_path / "model.pkl", 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocabulary = model_data['vocabulary']
        self.n_components = model_data['n_components']
        self.n_states_per_word = model_data['n_states_per_word']
        self.n_features = model_data['n_features']
        
        # Restore models
        self._initialize_models()
        for word in self.vocabulary:
            if f'{word}_states' in model_data:
                states_data = model_data[f'{word}_states']
                for state_id, state_data in states_data.items():
                    state = self.word_models[word].states[int(state_id)]
                    state.gmm.weights.data = torch.FloatTensor(state_data['gmm_weights'])
                    state.gmm.means.data = torch.FloatTensor(state_data['gmm_means'])
                    state.gmm.log_vars.data = torch.FloatTensor(state_data['gmm_log_vars'])
                    state.transitions = state_data['transitions']
        
        print(f"Model loaded from {model_path}")

# Example usage and digit recognition setup
if __name__ == "__main__":
    if not ESPNET_AVAILABLE:
        print("Please install ESPnet2: pip install espnet")
        exit(1)
    
    # Initialize digit recognition system
    digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    
    # Configure frontend for digit recognition
    frontend_config = {
        'fs': 16000,
        'n_mels': 40,  # Fewer mel-bins for digits
        'hop_length': 160,  # 10ms frame shift
        'win_length': 400,  # 25ms window
        'fmin': 80,
        'fmax': 7600,
    }
    
    asr_system = ESPnetHMMGMM(
        vocabulary=digits,
        n_components=3,
        n_states_per_word=3,
        frontend_config=frontend_config
    )
    
    print("ESPnet2-integrated HMM/GMM ASR System initialized!")
    print(f"Vocabulary: {digits}")
    print(f"Feature dimensions: {asr_system.n_features}")
    print(f"Frontend configuration: {frontend_config}")
    
    print("\n=== Usage Examples ===")
    print("1. Prepare training data:")
    print("   audio_files = {'zero': ['zero1.wav', 'zero2.wav'], 'one': ['one1.wav']}")
    print("   data_dir = asr_system.prepare_training_data(audio_files)")
    print("\n2. Train the system:")
    print("   asr_system.train_from_data_dir(data_dir)")
    print("\n3. Recognize speech:")
    print("   result = asr_system.recognize('test.wav')")
    print("\n4. Save/load models:")
    print("   asr_system.save_model('digit_model')")
    print("   asr_system.load_model('digit_model')")