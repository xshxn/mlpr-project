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

noise_reduced_column = "Audio"  
noise_reduction_method = "wiener" 

process_dataset(csv_file, output_dir, input_column=noise_reduced_column, output_column= processed_column, threshold_db=threshold_db, min_silence_duration=min_silence_duration)