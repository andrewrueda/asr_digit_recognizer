import json
import torch
import librosa
import os
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import soundfile as sf
from pathlib import Path
from espnet2.asr.frontend.default import DefaultFrontend


class DataManager:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def _gather_by_word(self, vocab: Dict[str, str] = None) -> defaultdict[str, List[str]]:
        """Gather audio files by target word."""
        audio_files = defaultdict(list)

        for file_name in os.listdir(data_dir):
            word, *_ = file_name.split("_")

            if vocab:
                word = vocab[word]

            audio_files[word].append(os.path.join(data_dir, file_name))

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
                _, speaker, end = file.split("_")
                indx, _ = end.split(".")
                utt_id = f"{speaker}-{indx}"

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


    def load_data(self, wav_file: str = "data/wav.scp",
                    text_file: str = "data/text.scp") -> defaultdict[str, List[Tuple[str, str]]]:
        """Load data from scp files."""
        wav_dict = {}
        with open(wav_file, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, wav_path = line.strip().split()
                wav_dict[utt_id] = wav_path

        word_data = defaultdict(list)
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, target_word = line.strip().split()
                word_data[target_word].append((utt_id, wav_dict[utt_id]))

        return word_data



class FeatureExtractor:
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
            # ToDo
            pass
        
        self.specaug = None
        if use_specaug:
            # ToDo
            pass


    def extract_features(self, audio_path: str, cmn: bool = True,
                         apply_specaug: bool = False):
        speech, sample_rate = sf.read(audio_path)

        # downsample if necessary
        if sample_rate != self.fs:
            speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=self.fs)
        speech = torch.FloatTensor(speech)

        # temporarily add batch dimension
        speech = speech.unsqueeze(0) # [batch, time]
        speech_lengths = torch.LongTensor(speech.shape[1])

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
    

    def save_config(self, config_path: str):
        # ToDO
        pass


    
if __name__ == "__main__":
    with open("configs/config.json", "r", encoding="utf-8") as config_file:
        configs = json.load(config_file)

    data_dir = configs["data_dir"]

    feature_extractor = FeatureExtractor()

    data_manager = DataManager(data_dir)

    vocab_list = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    vocab = { str(i) : vocab_list[i] for i in range(10) }

    # info = data_manager._gather_by_word(vocab)
    # data_manager.kaldi_prepare(info)

    word_data = data_manager.load_data()
    print(word_data)
