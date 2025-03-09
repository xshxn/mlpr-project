import pandas as pd
import librosa
import numpy as np
import json
from tqdm import tqdm 


df = pd.read_csv("torgo_mainfest.csv")


def extract_mfcc(audio_path):
    sr = 16000
    frame_length = 0.025
    frame_stride = 0.01
    n_fft = int(sr*frame_length)
    hop_length = int(sr*frame_stride)
    n_mfcc = 13
    try:
        y, sr = librosa.load(audio_path, sr = sr)
        mfccs = librosa.feature.mfcc(y = y, sr = sr, n_mfcc=n_mfcc, n_fft = n_fft, hop_length = hop_length)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

        return {
            "MFCCs": json.dumps(mfccs.T.tolist()),  
            "Delta_MFCCs": json.dumps(delta_mfccs.T.tolist()),  
            "Delta2_MFCCs": json.dumps(delta2_mfccs.T.tolist()),  
            "Chroma": json.dumps(chroma.T.tolist()),  
            "Spectral_Contrast": json.dumps(spectral_contrast.T.tolist())  
        }
    except Exception as e:
        print(e)
        return {
            "MFCCs": None,
            "Delta_MFCCs": None,
            "Delta2_MFCCs": None,
            "Chroma": None,
            "Spectral_Contrast": None
        }


tqdm.pandas()
feature_dictionary= df["Processed Audio"].progress_apply(lambda x: json.dumps(extract_mfcc(x)) if pd.notna(x) else {})
df["MFCCs"] = feature_dictionary.apply(lambda x: x.get("MFCCs"))
df["Delta_MFCCs"] = feature_dictionary.apply(lambda x: x.get("Delta_MFCCs"))
df["Delta2_MFCCs"] = feature_dictionary.apply(lambda x: x.get("Delta2_MFCCs"))
df["Chroma"] = feature_dictionary.apply(lambda x: x.get("Chroma"))
df["Spectral_Contrast"] = feature_dictionary.apply(lambda x: x.get("Spectral_Contrast"))

df.to_csv('torgo_features.csv', index = False)