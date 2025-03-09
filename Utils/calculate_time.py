import os
from pydub import AudioSegment


def get_audio_duration(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000  
    except Exception as e:
        print(e)
        return 0  

def calculate_time():
    count = 0
    total_time = 0
    times = []
    
    base_dir = r'E:\MLPR Data'
    files = []

    for speaker_folder in os.listdir(base_dir):
        speaker_path = os.path.join(base_dir, speaker_folder)
        if os.path.isdir(speaker_path):
            for session_folder in ["Session1", "Session2", "Session3"]:
                session_path = os.path.join(speaker_path, session_folder)
                mics = ["wav_arrayMic"]

                for mic_path in mics:
                    audio_path = os.path.join(session_path, mic_path)
                    if os.path.exists(audio_path):
                        for audio_file in os.listdir(audio_path):
                            count += 1
                            file_path = os.path.join(audio_path, audio_file)
                            if file_path.endswith('.wav'):
                                files.append(file_path)

                                duration = get_audio_duration(file_path)

                                times.append(duration)
                                total_time += duration
    
    if files:
        print("Sample files:", files[:5])
        print("Sample durations (seconds):", times[:5])


    print(count)
    return total_time / 3600  

print("Total duration (hours):", calculate_time())
