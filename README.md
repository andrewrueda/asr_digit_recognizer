# HMM-GMM Spoken Digit Recognizer

## Overview
This is an HMM-GMM model for spoken digit recognition implemented from scratch using PyTorch and ESPnet2. `train.py` is set up for altering training configurations in the commmand line, such as number of states per HMM, number of mel filterbanks during feature extraction, and the use of Spectrogram Augmentation.

Also included is a Streamlit app for demo-able inference on a trained model.

## Usage:
`pip install -r requirements.txt`

`streamlit run main.py`

## Datasets:

1. **[Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset.git)**
   - 3,000 recordings   
   - Source: GitHub

3. **[Free Spoken Digit Database](https://www.kaggle.com/datasets/subhajournal/free-spoken-digit-database)**
   - 17,000 recordings  
   - Source: Kaggle

## Best Results:

**Free Spoken Digit Dataset:**

| Digit    | Accuracy |    
|----------|----------|
| zero     | 0.95     |
| one      | 0.883    |
| two      | 0.933    |
| three    | 0.817    |
| four     | 0.933    |
| five     | 0.9      |
| six      | 0.783    |
| seven    | 0.933    |
| eight    | 0.767    |
| nine     | 0.717    |
| total    | 0.862    |


\
**Free Spoken Digit Database:**

| Digit    | Accuracy |    
|----------|----------|
| zero     | 0.823    |
| one      | 0.713    |
| two      | 0.558    |
| three    | 0.676    |
| four     | 0.714    |
| five     | 0.466    |
| six      | 0.774    |
| seven    | 0.627    |
| eight    | 0.731    |
| nine     | 0.621    |
| total    | 0.671    |
