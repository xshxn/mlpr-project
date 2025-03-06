from data_csv_manifest import *
from remove_silence import *
from noise_removal import *

audio_file_paths, prompt_file_paths = create_paths()
filtered_audio, filtered_prompts = filter_matching_pairs(audio_file_paths, prompt_file_paths)

base_dir = r'E:\MLPR Data'

create_csv(filtered_audio, filtered_prompts, base_dir)

csv_file = "torgo_mainfest.csv"
output_dir = "E:\Processed_Audio"
threshold_db = -40
min_silence_duration = 0.1
processed_column = "Processed Audio"

noise_reduced_column = "Noise_Reduced_Audio"  
noise_reduction_method = "spectral_subtraction" 



#process_noise_reduction(csv_file, output_dir, noise_reduction_method=noise_reduction_method, processed_column='Audio', noise_reduced_column=noise_reduced_column)

process_dataset(csv_file, output_dir)