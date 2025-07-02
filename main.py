import streamlit as st
import torch
from model import WordRecognizer
from datetime import datetime
import argparse

from eval import HMMTest, prepare_one_inference

import json
import os


BEST = "humane-noctule-37"
model_path = f"./saved/{BEST}/{BEST}.pt"


# Initialize

title = "ASR Digit Recognizer"

st.markdown(
    f"""
    <style>
    .custom-title {{
        font-family: 'Georgia', serif;
        font-size: 36px;
        color: #3e0ffa;
        text-align: center;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }}
    </style>

    <div class="custom-title">{title}</div>
    """,
    unsafe_allow_html=True
)

st.markdown("HMM-GMM model for spoken digit recognition")

if 'recognizer' not in st.session_state:
    st.session_state['recognizer'] = torch.load(model_path, weights_only=False)

if 'configs' not in st.session_state:
    parser = argparse.ArgumentParser(description='Eval')
    with open(f"saved/{BEST}/settings.json", "r", encoding="utf-8") as settings_file:
        st.session_state['configs'] = json.load(settings_file) 

if 'tester' not in st.session_state:
    st.session_state['tester'] = HMMTest(st.session_state['recognizer'], st.session_state['configs']['vocab'])


# Record audio snippet
recordings_dir = "./recordings"
os.makedirs(recordings_dir, exist_ok=True)
wav_filename = None

audio = st.audio_input("Record yourself saying a digit zero through nine!")

if audio:
    wav_path = os.path.join(recordings_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav")
    with open(wav_path, "wb") as f:
        f.write(audio.getbuffer())


    st.session_state['configs']['audio_file'] = wav_path
    features = prepare_one_inference(st.session_state['configs'])

    prediction = st.session_state['tester'].predict(features)[0]

    st.markdown(
        f"""
        <style>
        .green-box {{
            background-color: #d4edda;
            border-left: 6px solid #28a745;
            padding: 16px;
            margin: 2rem auto;  /* center horizontally */
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            color: #155724;
            font-size: 26px;
            font-weight: bold;
            max-width: 300px;  /* make it narrower */
            text-align: center;  /* optional: center the text too */
        }}
        </style>

        <div class="green-box">
            {prediction}!
        </div>
        """,
        unsafe_allow_html=True
    )