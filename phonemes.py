import torch
import torchaudio
import torchaudio.functional as F
import pandas as pd
import json
from tqdm import tqdm
import re  # Add re for regular expressions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Read the CSV file with paths to audio and prompt files
df = pd.read_csv('torgo_features.csv')

# Load the torchaudio pipeline and model (do this once)
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).to(device)
LABELS = bundle.get_labels(star=None)
DICTIONARY = bundle.get_dict(star=None)

# Keep your alignment function as before
def align(emission, tokens):
    # Remove any blank tokens (0) from the targets
    tokens = [t for t in tokens if t != 0]
    
    # If no tokens remain after filtering, return empty results
    if not tokens:
        return torch.tensor([]), torch.tensor([])
    
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    try:
        alignments, scores = F.forced_align(emission, targets, blank=0)
        alignments, scores = alignments[0], scores[0]
        scores = scores.exp()
        return alignments, scores
    except Exception as e:
        print(f"Error in alignment: {str(e)}")
        return torch.tensor([]), torch.tensor([])

# Utility: group consecutive frames with the same token into segments,
# calculating start and end times based on the time per frame.
def get_segments(aligned_tokens, time_per_frame):
    segments = []
    current_token = aligned_tokens[0].item()
    start_frame = 0
    for i in range(1, len(aligned_tokens)):
        if aligned_tokens[i].item() != current_token:
            end_frame = i - 1
            segments.append({
                "phoneme": LABELS[current_token],
                "start_time": start_frame * time_per_frame,
                "end_time": (end_frame + 1) * time_per_frame
            })
            current_token = aligned_tokens[i].item()
            start_frame = i
    # Add the final segment
    end_frame = len(aligned_tokens) - 1
    segments.append({
        "phoneme": LABELS[current_token],
        "start_time": start_frame * time_per_frame,
        "end_time": (end_frame + 1) * time_per_frame
    })
    return segments

results = []

# Iterate over each row in the CSV file using tqdm for progress tracking
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
    try:
        # Adjust these column names if your CSV structure is different.
        audio_path = row["Audio"]
        prompt_path = row["Prompts"]

        # Read and tokenize the prompt transcript
        with open(prompt_path, "r") as f:
            # Remove punctuation, braces, and other non-text characters
            raw_transcript = f.read().strip().lower()
            # Keep only letters, numbers and spaces
            clean_transcript = re.sub(r'[^\w\s]', '', raw_transcript)
            # Remove extra whitespace
            clean_transcript = ' '.join(clean_transcript.split())
            transcript = clean_transcript.split()
        
        # Add safety check for characters not in dictionary
        missing_chars = set()
        tokenized_transcript = []
        for word in transcript:
            for c in word:
                if c in DICTIONARY:
                    token = DICTIONARY[c]
                    # Skip blank tokens (usually index 0)
                    if token != 0:
                        tokenized_transcript.append(token)
                else:
                    missing_chars.add(c)
        
        if missing_chars:
            print(f"Warning for {audio_path}: Characters not found in dictionary: {missing_chars}")
        
        if not tokenized_transcript:
            print(f"Warning: No valid tokens for {audio_path}, skipping")
            continue

        # Load audio waveform and get sample rate
        waveform, sample_rate = torchaudio.load(audio_path)

        # Get model emission and calculate forced alignments
        with torch.inference_mode():
            emission, _ = model(waveform.to(device))

        # Compute time per frame using total audio duration
        duration = waveform.shape[1] / sample_rate
        num_frames = emission.shape[0]
        time_per_frame = duration / num_frames

        aligned_tokens, alignment_scores = align(emission, tokenized_transcript)
        
        # Skip if alignment failed
        if len(aligned_tokens) == 0:
            print(f"Alignment failed for {audio_path}, saving basic info only")
            results.append({
                "audio_file": audio_path,
                "prompt_file": prompt_path,
                "error": "Alignment failed",
                "segments": []
            })
            continue

        # Group aligned tokens into segments with start and end times
        segments = get_segments(aligned_tokens, time_per_frame)

        # Store results for this file
        results.append({
            "audio_file": audio_path,
            "prompt_file": prompt_path,
            "segments": segments
        })
    except Exception as e:
        print(f"Error processing {row.get('Audio', 'unknown')}: {str(e)}")
        results.append({
            "audio_file": row.get("Audio", "unknown"),
            "prompt_file": row.get("Prompts", "unknown"),
            "error": str(e),
            "segments": []
        })

# Save all phoneme segmentation results to a JSON file.
with open("phoneme_alignments.json", "w") as f:
    json.dump(results, f, indent=4)
