import pandas as pd
import json
import ast
import torchaudio
import torch
import numpy as np
from tqdm import tqdm
from g2p_en import G2p
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import nltk

nltk.download('averaged_perceptron_tagger_eng')

csv_path = "torgo_features.csv"
df = pd.read_csv(csv_path)


g2p = G2p()

model_name = "facebook/wav2vec2-large-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

def align_phonemes(audio_path, mfcc_features, text):
    """Aligns phonemes with MFCC features using Wav2Vec2 & DTW"""

    # Convert text to phonemes
    phonemes = g2p(text)

    # Convert MFCC string to numpy array
    mfcc_features = np.array(ast.literal_eval(mfcc_features))

    # Load audio and process with Wav2Vec2
    waveform, sample_rate = torchaudio.load(audio_path)
    input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].lower()

    # Convert ASR transcription to phonemes
    asr_phonemes = g2p(transcription)

    # Align phonemes with MFCCs using DTW
    _, path = fastdtw(phonemes, asr_phonemes, dist=euclidean)

    aligned_segments = []
    for i, (p_idx, m_idx) in enumerate(path):
        aligned_segments.append({
            "phoneme": phonemes[p_idx],
            "start_time": m_idx * (len(waveform[0]) / sample_rate / len(mfcc_features)),
            "end_time": (m_idx + 1) * (len(waveform[0]) / sample_rate / len(mfcc_features))
        })

    return aligned_segments

# Process each row in the CSV with progress bar
results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Audio Files"):
    audio_file = row["Audio"]
    text_file = row["Prompts"]
    mfcc_data = row["MFCCs"]  # Precomputed 13 MFCCs

    # Read text prompt
    with open(text_file, "r") as f:
        text = f.read().strip()

    # Get phoneme segmentation
    segments = align_phonemes(audio_file, mfcc_data, text)

    # Store result
    results.append({
        "audio_file": audio_file,
        "segments": segments
    })

# Save results to JSON
with open("phoneme_segments.json", "w") as f:
    json.dump(results, f, indent=4)

