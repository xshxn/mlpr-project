import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm

def remove_silence(audio, sr, threshold_db = -40, min_silence_duration = - 0.1):
    threshold_amp = 10**(threshold_db/20) #coverting threshold to amplitude ratio
    amplitude = np.abs(audio)
    is_sound = amplitude > threshold_amp
    min_silence_samples = int(min_silence_duration * sr) #min silence duration to samples
    processed_mask = np.copy(is_sound) #array for processed silence mask

    silence_starts = np.where(np.logical_and(is_sound[:-1], ~is_sound[1:]))[0] + 1 #Find silent segments that are shorter than min_silence_duration
    silence_ends = np.where(np.logical_and(~is_sound[:-1], is_sound[1:]))[0] + 1

    # Handle case where audio starts with or ends with silence
    if not is_sound[0]:
        silence_starts = np.insert(silence_starts, 0, 0)

    if not is_sound[-1]:
        silence_ends = np.append(silence_ends, len(is_sound))

    for start, end in zip(silence_starts, silence_ends): #keeps short silence segments - between words  
        if end - start < min_silence_samples:
            processed_mask[start:end] = True

    return audio[processed_mask]

def process_dataset(csv_file, output_dir, threshold_db = -40, min_silence_duration = 0.1, processed_column='Processed_Audio'):

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)

    if 'Audio' not in df.columns:
        raise ValueError("CSV file must contain an 'Audio' column with file paths")
    df[processed_column] = None
    df['Processing_Error'] = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Removing silence"):
        try:
            audio_path = row['Audio']

            parts = audio_path.split(os.sep)
            base_dir_index = parts.index("MLPR Data")
            speaker_id = parts[base_dir_index + 1]
            session_id = parts[base_dir_index + 2]
            mic_id = parts[base_dir_index + 3]


            audio, sr = librosa.load(audio_path, sr=None)

            audio_no_silence = remove_silence(audio, sr, threshold_db, min_silence_duration)

            filename = os.path.basename(audio_path)

            speaker_dir = os.path.join(output_dir, speaker_id)
            session_dir = os.path.join(speaker_dir, session_id)
            mic_dir = os.path.join(session_dir, mic_id)
            os.makedirs(mic_dir, exist_ok=True) 
            output_path = os.path.join(mic_dir, f"no_silence_{filename}")
            
            sf.write(output_path, audio_no_silence, sr)

            df.loc[idx, processed_column] = output_path
        
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            df.loc[idx, 'Processing_Error'] = str(e)


    df.to_csv(csv_file, index=False)
    print(f"Processing complete. Original CSV updated with {processed_column} column.")

    backup_path = os.path.join(output_dir, "processed_files_backup.csv")
    df.to_csv(backup_path, index=False)
    print(f"Backup CSV saved to {backup_path}")

