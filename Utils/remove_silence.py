import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm

def remove_silence(audio, sr, threshold_db=-40, min_silence_duration=0.1):

    threshold_amp = 10**(threshold_db/20) #converting threshold to amp ratio
    amplitude = np.abs(audio)
    is_sound = amplitude > threshold_amp
    min_silence_samples = int(min_silence_duration * sr) #min silence duration to samples
    

    processed_mask = np.copy(is_sound) # array for processed silence mask
    

    silence_starts = np.where(np.logical_and(is_sound[:-1], ~is_sound[1:]))[0] + 1 #segnments shorter than min silence time
    silence_ends = np.where(np.logical_and(~is_sound[:-1], is_sound[1:]))[0] + 1
    

    if not is_sound[0]: #checking if silence is at the beginning or end of file
        silence_starts = np.insert(silence_starts, 0, 0)
    if not is_sound[-1]:
        silence_ends = np.append(silence_ends, len(is_sound))
    

    for start, end in zip(silence_starts, silence_ends):
        if end - start < min_silence_samples:
            processed_mask[start:end] = True

    return audio[processed_mask]

def process_dataset(csv_file, output_dir, 
                   input_column='Audio',  
                   output_column='Silence_Removed_Audio',
                   threshold_db=-40, 
                   min_silence_duration=0.1):


    os.makedirs(output_dir, exist_ok=True)
    

    df = pd.read_csv(csv_file)
    

    if input_column not in df.columns:
        raise ValueError(f"CSV file must contain an '{input_column}' column with file paths")
    

    df[output_column] = None
    df['Silence_Removal_Error'] = None
    

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Removing silence"):
        try:

            audio_path = row[input_column]

            if pd.isna(audio_path):
                print(f"Skipping row {idx}: No audio file")
                continue

            audio, sr = librosa.load(audio_path, sr=None)

            audio_no_silence = remove_silence(audio, sr, threshold_db, min_silence_duration)

            if input_column == 'Audio':

                try:
                    parts = audio_path.split(os.sep)
                    base_dir_index = parts.index("MLPR Data")
                    speaker_id = parts[base_dir_index + 1]
                    session_id = parts[base_dir_index + 2]
                    mic_id = parts[base_dir_index + 3]

                    speaker_dir = os.path.join(output_dir, speaker_id)
                    session_dir = os.path.join(speaker_dir, session_id)
                    mic_dir = os.path.join(session_dir, mic_id)
                    os.makedirs(mic_dir, exist_ok=True)
                    
                    filename = os.path.basename(audio_path)
                    output_path = os.path.join(mic_dir, f"no_silence_{filename}")
                
                except ValueError:
                    filename = os.path.basename(audio_path)
                    output_path = os.path.join(output_dir, f"no_silence_{filename}")
            else:

                filename = os.path.basename(audio_path)
                dirname = os.path.dirname(audio_path)
                

                rel_path = os.path.relpath(dirname, os.path.dirname(output_dir))
                new_dir = os.path.join(output_dir, rel_path)
                os.makedirs(new_dir, exist_ok=True)
                output_path = os.path.join(new_dir, f"no_silence_{filename}")

            sf.write(output_path, audio_no_silence, sr)

            df.loc[idx, output_column] = output_path
            
        except Exception as e:
            print(f"Error processing {row.get(input_column, 'unknown')}: {e}")
            df.loc[idx, 'Silence_Removal_Error'] = str(e)

    df.to_csv(csv_file, index=False)
    print(f"Processing complete. Original CSV updated with {output_column} column.")

    backup_path = os.path.join(output_dir, "silence_removed_files_backup.csv")
    df.to_csv(backup_path, index=False)
    print(f"Backup CSV saved to {backup_path}")

