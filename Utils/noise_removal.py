import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm
from scipy import signal

def spectral_subtraction(audio, sr, frame_length = 2048, hop_length = 512, beta = 0.002):
    noise_duration = min(0.3, len(audio)/sr/3)
    noise_sample = audio[:int(sr*noise_duration)]

    noise_spec = np.mean(np.abs(librosa.stft(noise_sample, n_fft=frame_length, 
                                             hop_length=hop_length))**2, axis=1)
    
    stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    stft_mag = np.abs(stft)
    stft_phase = np.angle(stft)

    stft_mag_clean = np.maximum(stft_mag**2 - beta * noise_spec[:, np.newaxis], 0)**(1/2)

    stft_clean = stft_mag_clean * np.exp(1j * stft_phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))
    
    return audio_clean

def wiener_filter(audio, noise_leel = 0.01):
    audio_clean = signal.wiener(audio, mysize=11, noise=noise_leel)
    return audio_clean

def process_noise_reduction(csv_file, output_dir, 
                           noise_reduction_method='spectral_subtraction',
                           processed_column='Processed_Audio', 
                           noise_reduced_column='Noise_Reduced_Audio'):
    
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_file)

    if processed_column not in df.columns:
        raise ValueError(f"CSV file must contain a '{processed_column}' column with processed file paths")
    
    df[noise_reduced_column] = None
    df['Noise_Reduction_Error'] = None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Removing noise"):
        try:

            processed_audio_path = row[processed_column]

            if pd.isna(processed_audio_path):
                print(f"Skipping row {idx}: No processed audio file")
                continue

            audio, sr = librosa.load(processed_audio_path, sr=None)

            if noise_reduction_method == 'spectral_subtraction':
                audio_clean = spectral_subtraction(audio, sr)
            elif noise_reduction_method == 'wiener':
                audio_clean = wiener_filter(audio)
            else:
                raise ValueError(f"Unknown noise reduction method: {noise_reduction_method}")

            filename = os.path.basename(processed_audio_path)
            dirname = os.path.dirname(processed_audio_path)

            if processed_column == 'Audio':

                parts = processed_audio_path.split(os.sep)
                try:
                    base_dir_index = parts.index("MLPR Data")
                    speaker_id = parts[base_dir_index + 1]
                    session_id = parts[base_dir_index + 2]
                    mic_id = parts[base_dir_index + 3]
                    
                    speaker_dir = os.path.join(output_dir, speaker_id)
                    session_dir = os.path.join(speaker_dir, session_id)
                    mic_dir = os.path.join(session_dir, mic_id)
                    os.makedirs(mic_dir, exist_ok=True)
                    
                    output_path = os.path.join(mic_dir, f"noise_reduced_{filename}")
                except ValueError:
                    output_path = os.path.join(output_dir, f"noise_reduced_{filename}")
            else:

                rel_path = os.path.relpath(dirname, os.path.dirname(output_dir))
                new_dir = os.path.join(output_dir, rel_path)
                os.makedirs(new_dir, exist_ok=True)
                output_path = os.path.join(new_dir, f"noise_reduced_{filename}")
            

            sf.write(output_path, audio_clean, sr)
            

            df.loc[idx, noise_reduced_column] = output_path
            
        except Exception as e:
            print(f"Error processing {row.get(processed_column, 'unknown')}: {e}")
            df.loc[idx, 'Noise_Reduction_Error'] = str(e)
    

    df.to_csv(csv_file, index=False)
    print(f"Processing complete. Original CSV updated with {noise_reduced_column} column.")
    

    backup_path = os.path.join(output_dir, "noise_reduced_files_backup.csv")
    df.to_csv(backup_path, index=False)
    print(f"Backup CSV saved to {backup_path}")