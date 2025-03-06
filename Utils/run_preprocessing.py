from data_csv_manifest import *

audio_file_paths, prompt_file_paths = create_paths()
filtered_audio, filtered_prompts = filter_matching_pairs(audio_file_paths, prompt_file_paths)

base_dir = r'E:\MLPR Data'

create_csv(filtered_audio, filtered_prompts, base_dir)