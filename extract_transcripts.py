import pandas as pd
import re

df = pd.read_csv('torgo_features_paths_processed.csv')

def normalize_transcript(file_path):
    try:

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""
        
    text = text.lower()
    normalized = re.sub(r'[^a-z ]+', '', text)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

df['transcipt'] = df['Prompts'].apply(normalize_transcript)
df.to_csv('torgo_vectors_transcripts_processed.csv', index=False)