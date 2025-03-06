import os
import pandas as pd

def create_paths():

    base_dir = r'E:\MLPR Data'
    audio_file_paths = []
    prompt_file_paths = []

    for speaker_folder in os.listdir(base_dir):
        speaker_path = os.path.join(base_dir, speaker_folder)
        if os.path.isdir(speaker_path):

            for session_folder in ["Session1", "Session2", "Session3"]:
                session_path = os.path.join(speaker_path, session_folder)
                audio_path = os.path.join(session_path, "wav_arrayMic")
                if os.path.exists(audio_path):
                    for root, _, files in os.walk(audio_path):
                        for file in files:
                            if file.endswith(".wav"):
                                audio_file_paths.append(os.path.join(root, file))
            
            for session_folder in ["Session1", "Session2", "Session3"]:
                session_path = os.path.join(speaker_path, session_folder)
            
                prompts_path = os.path.join(session_path, "prompts")
                if os.path.exists(prompts_path):
                    for root, _, files in os.walk(prompts_path):
                        for file in files:
                            if file.endswith(".txt"):
                                prompt_file_paths.append(os.path.join(root, file))

    return audio_file_paths, prompt_file_paths



def filter_matching_pairs(audio_file_paths, prompts_file_paths):

    audio_mapping = {}
    for audio_path in audio_file_paths:
        parts = audio_path.split(os.sep)
        try:
            base_dir_index = parts.index("MLPR Data")
            speaker = parts[base_dir_index + 1]
            session = parts[base_dir_index + 2]
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            
            key = (speaker, session, basename)
            audio_mapping[key] = audio_path
        except (ValueError, IndexError):
            continue
    
    prompt_mapping = {}
    for prompt_path in prompts_file_paths:
        parts = prompt_path.split(os.sep)
        try:
            base_dir_index = parts.index("MLPR Data")
            speaker = parts[base_dir_index + 1]
            session = parts[base_dir_index + 2]
            basename = os.path.splitext(os.path.basename(prompt_path))[0]
            
            key = (speaker, session, basename)
            prompt_mapping[key] = prompt_path
        except (ValueError, IndexError):
            continue

    common_keys = set(audio_mapping.keys()) & set(prompt_mapping.keys())
    

    filtered_audio = [audio_mapping[key] for key in common_keys]
    filtered_prompts = [prompt_mapping[key] for key in common_keys]
    
    return filtered_audio, filtered_prompts


def get_speaker_category(file_path):
    parts = file_path.split(os.sep)
    try:
        base_dir_index = parts.index("MLPR Data")
        speaker = parts[base_dir_index + 1].lower()  

        

        if speaker.startswith('fc') or speaker.startswith('mc'):
            return 'control'

        elif (speaker.startswith('f') and not speaker.startswith('fc')) or \
             (speaker.startswith('m') and not speaker.startswith('mc')):
            return 'dysarthric'
        else:
            return 'unknown'  
    except (ValueError, IndexError):
        return 'unknown'

def extract_speaker_id(file_path):
    parts = file_path.split(os.sep)
    try:
        base_dir_index = parts.index("MLPR Data")
        speaker_id = parts[base_dir_index + 1]
        return speaker_id
    except (ValueError, IndexError):
        return None
    
def extract_session_id(file_path):
    parts = file_path.split(os.sep)
    try:
        base_dir_index = parts.index("MLPR Data")
        session_id = parts[base_dir_index + 2]
        return session_id
    except (ValueError, IndexError):
        return None
    
def create_csv(filtered_audio, filtered_prompts, base):
    df = pd.DataFrame({'Audio': filtered_audio, 'Prompts': filtered_prompts})


    df['Category'] = df['Audio'].apply(get_speaker_category)
    df['Speaker'] = df['Audio'].apply(extract_speaker_id)
    df['Session'] = df['Audio'].apply(extract_session_id)

    csv_path = os.path.join(base, "torgo_manifest.csv")

    df.to_csv(csv_path, index=False)

    
audio_file_paths, prompt_file_paths = create_paths()
filtered_audio, filtered_prompts = filter_matching_pairs(audio_file_paths, prompt_file_paths)



