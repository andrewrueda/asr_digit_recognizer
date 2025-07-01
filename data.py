import json
import torch
import librosa
import os
import logging
import random
import regex as re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from pathlib import Path
import soundfile as sf
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.specaug.specaug import SpecAug
from torch.nn.utils.rnn import pad_sequence


class DataManager:
    """Handles writing kaldi-like files and loading data.
    file name format: word_speaker_indx.wav"""
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)


    def gather_by_word(self, test: bool = False, test_indx: int = 10, separated_by_word = False,
                       vocab: List[str] = None) -> defaultdict[str, List[str]]:
        """gather audio files by target word."""

        audio_files = defaultdict(list)

        if separated_by_word:
            # for Kaggle dataset
            for dir_path, _, files in os.walk(self.data_dir):
                if dir_path == self.data_dir:
                    continue

                word = os.path.basename(dir_path)

                for file in files:
                    if file.endswith('wav'):
                        file_path = os.path.join(dir_path, file)

                        is_train = random.random() > (test_indx / 100)

                        if (is_train ^ test):
                            audio_files[word].append(file_path)

        else:
            # for smaller free spoken digit dataset
            for file_name in os.listdir(self.data_dir):
                word, _, indx = file_name.split("_")
                indx, _ = indx.split(".")

                if (int(indx) < test_indx and test) or (int(indx) >= test_indx and not test):
                    if vocab:
                        word = vocab[int(word)]

                    audio_files[word].append(os.path.join(self.data_dir, file_name))

        return audio_files


    def _write_file(self, file_path: Path, lines: List[str]):
        with open(file_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")


    def kaldi_prepare(self, audio_files: defaultdict[str, List[str]],
                      output_dir: str = "data"):
        """Prepare dataset in Kaldi-style format."""
        wav_scp = []
        utt2spk = []
        text = []

        for word, files in audio_files.items():
            for file in files:
                file_name = os.path.basename(file)

                pattern = f"nohash"
                match = re.search(pattern, file)

                if match:
                    indx, speaker, *_ = file_name.split("_")

                else:
                    _, speaker, end = file_name.split("_")
                    indx, _ = end.split(".")

                utt_id = f"{word}-{speaker}-{indx}"
                wav_scp.append(f"{utt_id} {Path(file)}")
                utt2spk.append(f"{utt_id} {speaker}")
                text.append(f"{utt_id} {word}")
            
        self._write_file(f"{output_dir}/wav.scp", wav_scp)
        self._write_file(f"{output_dir}/utt2spk.scp", utt2spk)
        self._write_file(f"{output_dir}/text.scp", text)

        # create spk2utt
        spk2utt = {}

        with open(f"{output_dir}/utt2spk.scp", "r", encoding="utf-8") as f:
            for line in f:
                utt, spk = line.strip().split()

                if spk not in spk2utt:
                    spk2utt[spk] = []
                spk2utt[spk].append(utt)
        
        with open(f"{output_dir}/spk2utt.scp", "w", encoding="utf-8") as f:
            for spk, utts in spk2utt.items():
                f.write(f"{spk} {' '.join(utts)}\n")
            

    def load_data(self, split = "train", wav_scp_file: str = "wav.scp",
                    text_scp_file: str = "text.scp") -> defaultdict[str, List[Tuple[str, str]]]:
        """Load data from scp files."""

        wav_scp_path = os.path.join("data", split, wav_scp_file)
        text_scp_path = os.path.join("data", split, text_scp_file)

        wav_dict = {}
        with open(wav_scp_path, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, wav_path = line.strip().split()
                wav_dict[utt_id] = wav_path

        word_data = defaultdict(list)
        with open(text_scp_path, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, target_word = line.strip().split()
                word_data[target_word].append((utt_id, wav_dict[utt_id]))

        return word_data


class FeatureExtractor:
    """Handles feature extraction and loading data as tensors"""
    def __init__(self,
                 fs: int = 16000,
                 n_fft: int = 512, # num of samples in each window
                 n_mels: int = 80,
                 hop_length: int = 160,
                 win_length: int = 400,
                 fmin: int = 80,
                 fmax: int = 7600,
                 frontend_type: str = "default",
                 use_specaug: bool = False):
        
        self.fs = fs
        self.frontend_type = frontend_type
        self.feature_dim = n_mels

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
            self.frontend = FusedFrontends(
                fs=fs,
                n_fft=n_fft,
                n_mels=n_mels,
                hop_length=hop_length,
                win_length=win_length,
                fmin=fmin,
                fmax=fmax,
            )
        
        self.specaug = None
        if use_specaug:
            self.specaug = SpecAug(
                time_mask_width_range=(0, 40),
                freq_mask_width_range=(0, 30),
                num_time_mask=2,
                num_freq_mask=2
            )

    def extract_features(self, audio_path: str, cmn: bool = True,
                         apply_specaug: bool = False) -> torch.Tensor:
        speech, sample_rate = sf.read(audio_path)

        # downsample if necessary
        if sample_rate != self.fs:
            speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=self.fs)
        speech = torch.FloatTensor(speech)

        # temporarily add batch dimension
        speech = speech.unsqueeze(0) # [batch, time]
        speech_lengths = torch.LongTensor([speech.shape[1]])

        # extract features
        feats, feat_lengths = self.frontend(speech, speech_lengths)

        # apply specaug
        if self.specaug and apply_specaug:
            feats, feat_lengths = self.specaug(feats, feat_lengths)

        # remove the temp batch dimension
        feats = feats.squeeze(0)
     
        # apply cepstral mean normalization
        if cmn:
            feats = feats - feats.mean(dim=0, keepdim=True)

        return feats
    
    def target_tensors(self, target_dict: Dict[str, Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """Return tensors from kaldi-style files"""
        tensors = dict()

        for target, paths in target_dict.items():
            logging.info(f"word: {target}")
            sequences = []

            for _, path in paths:                
                sequences.append(self.extract_features(path))

            sequences = sorted(sequences, key=lambda x: x.shape[0]) 
            padded = pad_sequence(sequences, batch_first=True)
            logging.info(f"padded tensor shape: {list(padded.shape)}")

            shuffled_indices = torch.randperm(len(padded))
            padded_shuffled = padded[shuffled_indices]
            tensors[target] = padded_shuffled

        return tensors
    

    def save_config(self, config_path: str):
        # ToDO
        pass


if __name__ == "__main__":
    with open("configs/config.json", "r", encoding="utf-8") as config_file:
        configs = json.load(config_file)

    data_dir = configs["data"]["data_dir"]

    feature_extractor = FeatureExtractor()
    data_manager = DataManager(data_dir)

    if not os.path.isdir("data"):
        os.mkdir("data")
        os.mkdir("data/train")
        os.mkdir("data/test")

        audio_files = data_manager.gather_by_word(test=False)
        data_manager.kaldi_prepare(audio_files, output_dir="train")

        word_data = data_manager.load_data(split="train")
        tensors = feature_extractor.target_tensors(word_data)