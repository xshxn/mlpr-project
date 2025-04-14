import pandas as pd
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm


def extract_features_from_audio(audio_path, extractor):

    try:

        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading audio file '{audio_path}': {e}")
        return None

    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    waveform_np = waveform.numpy()

    try:
        features = extractor(raw_speech=waveform_np, sampling_rate=sample_rate, return_tensors="pt")
        return features
    except Exception as e:
        return None

def main():
    csv_file = "torgo_features.csv"  
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file '{csv_file}': {e}")
        return

    if "Audio" not in df.columns:
        print("CSV file does not contain an 'Audio' column.")
        return

    model_name = "facebook/wav2vec2-large-960h"
    try:

        extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading the Wav2Vec2 feature extractor: {e}")
        return


    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Audio Files"):
        audio_path = row.get("Audio")
        if not isinstance(audio_path, str) or pd.isna(audio_path):
            print(f"Skipping row {index}: missing or invalid audio path.")
            continue

        features = extract_features_from_audio(audio_path, extractor)
        if features is None:
            print(f"Feature extraction failed for '{audio_path}'.")

if __name__ == "__main__":
    main()
