import os
import torch
import pandas as pd
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from tqdm import tqdm

def extractAudioFeatures(audio_path, extractor, model):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading audio file {audio_path} - {e}")
        return None
    
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    
    waveform_np = waveform.numpy()
    
    try:

        features = extractor(raw_speech=waveform_np, sampling_rate=sample_rate, return_tensors="pt")
    except Exception as e:
        print(f"Error extracting features - {e}")
        return None

    with torch.no_grad():
        outputs = model(**features)

    hidden_states = outputs.last_hidden_state  
    
    return hidden_states

def saveUpdateCSV(csv_file, output_csv="torgo_features_paths.csv", output_dir=r"E:\MLPR Data\Features"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        df = pd.read_csv(csv_file)
        df.drop(columns=['MFCCs', 'Delta_MFCCs', 'Delta2_MFCCs', 'Chroma', 'Spectral_Contrast'], inplace=True)
    except Exception as e:
        print(f"Couldn't open csv file - {e}")
        return

    expected_cols = ["Audio", "Speaker", "Session", "Sequence"]
    for col in expected_cols:
        if col not in df.columns:
            print(f"csv missing: '{col}'.")
            return

    model_name = "facebook/wav2vec2-large-960h"
    try:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
        model.eval() 
    except Exception as e:
        print(f"Couldn't load model - {e}")
        return

    feature_paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='extracting features'):
        audio_path = row.get('Processed Audio')
        if not isinstance(audio_path, str) or pd.isna(audio_path):
            print(f"Invalid audio path - {audio_path}")
            feature_paths.append(None)
            continue
        
        speaker = str(row.get('Speaker')).strip()
        session = str(row.get("Session")).strip()
        sequence = str(row.get("Sequence")).strip()
        file_name = f"{speaker}_{session}_{sequence}"
        save_path = os.path.join(output_dir, f"feature_{file_name}.pt")

        hidden_states = extractAudioFeatures(audio_path, extractor, model)
        if hidden_states is None:
            print("Feature extraction failed; feature path not saved.")
            feature_paths.append(None)
            continue

        torch.save(hidden_states, save_path)
        feature_paths.append(save_path)

    df["FeaturePath"] = feature_paths

    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved as: {output_csv}")

def main():
    csv_file = "torgo_features.csv" 
    saveUpdateCSV(csv_file)

if __name__ == "__main__":
    main()
