import torch
import torch.functional as F
import json
import os
from data import DataManager, FeatureExtractor
from model import HMM, HMMState, GMM
import numpy as np

with open("configs/config.json", "r", encoding="utf-8") as config_file:
    configs = json.load(config_file)
    data_dir = configs["data_dir"]

# Load and prepare data
feature_extractor = FeatureExtractor()
data_manager = DataManager(data_dir)


if not os.path.isdir("data"):
    os.mkdir("data")
    os.mkdir("data/train")
    os.mkdir("data/test")

    train_files = data_manager.gather_by_word(test=False)
    data_manager.kaldi_prepare(train_files, output_dir="data/train")

    test_files = data_manager.gather_by_word(test=True)
    data_manager.kaldi_prepare(test_files, output_dir="data/test")


vocab_list = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
vocab = { str(i) : vocab_list[i] for i in range(10) }

train_data = data_manager.load_data(split="train")
test_data = data_manager.load_data(split="test")

print(f"Loading training tensors...")
training_tensors = feature_extractor.target_tensors(train_data)


# Load HMM
word_models = dict()
n_states = configs["n_states"]

for word in vocab_list:
    states = [HMMState(id=i) for i in range(n_states)]

    for i in range(n_states):
        if i < n_states - 1:
            states[i].add_transition(i, np.log(0.5))      # self-loop
            states[i].add_transition(i + 1, np.log(0.5))  # forward
        else:
            states[i].add_transition(i, np.log(1.0))      # final state self-loop
    
    word_models[word] = HMM(states, start_id=0)

print(word_models)


# Train
# for epoch in epochs: 100?
# Baum-Welch algorithm