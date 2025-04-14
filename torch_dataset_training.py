import torch
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
import string
import re
import os
import torch.nn.utils.rnn as rnn_utils
from transformers.feature_extraction_utils import BatchFeature
import torch.nn as nn
import torch.optim as optim


def normalizeText(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

class TorgoASRDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = row["FeaturePath"]
        prompt_path = row["Prompts"]

        try:
        # Load the feature file without using safe_globals
            features = torch.load(feature_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Failed to load feature from '{feature_path}': {e}")
        
        if isinstance(features, dict):
            input_values = features.get("input_values")
            if input_values is None:
                raise ValueError(f"'input_values' key not found in features loaded from {feature_path}")
        elif hasattr(features, "input_values"):
            input_values = features.input_values
        else:
            input_values = features

        if not isinstance(input_values, torch.Tensor):
            input_values = torch.tensor(input_values)

        if input_values.dim() == 2:
            seq_length = input_values.size(1)
        else:
            seq_length = input_values.size(0)

        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            transcript = f.read()
        transcript = normalizeText(transcript)

        return {
            "input_values": input_values,  # The feature tensor
            "seq_length": seq_length,        # Length of the feature sequence
            "transcript": transcript         # Normalized transcript text
        }
    
def collate(batch):
    input_values_list = [
        sample["input_values"].squeeze(0) if sample["input_values"].dim() == 2 else sample["input_values"]
        for sample in batch
    ]
    seq_lengths = [sample["seq_length"] for sample in batch]
    transcripts = [sample["transcript"] for sample in batch]

    padded_inputs = torch.nn.utils.rnn.pad_sequence(input_values_list, batch_first=True, padding_value=0)
    padded_inputs = padded_inputs.unsqueeze(-1)

    return {
        "input_values": padded_inputs,  # (batch, max_seq_length, feature_dim)
        "seq_lengths": torch.tensor(seq_lengths),
        "transcripts": transcripts
    }

def transcript_to_indices(transcript, char_to_idx):
    return [char_to_idx[char] for char in transcript if char in char_to_idx]

class Model(nn.Module):
        def __init__(self, input_dim, hidden_dim, vocab_size, num_layers = 2):
            super(Model, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(hidden_dim*2, vocab_size)

        def forward(self, x):
            outputs, _ = self.lstm(x)
            logits = self.fc(outputs)
            logits = logits.transpose(0, 1)
            return logits
        

def trainModel(model, train_loader, test_loader, char_to_idx, num_epochs=10, learning_rate = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch["input_values"].to(device) 
            input_lengths = batch["seq_lengths"].to(device) 

            targets_list = [torch.tensor(transcript_to_indices(t, char_to_idx), dtype=torch.long) 
                            for t in batch["transcripts"]]
            targets = torch.cat(targets_list).to(device)
            target_lengths = torch.tensor([len(t) for t in targets_list], dtype=torch.long).to(device)
            optimizer.zero_grad()

            outputs = model(inputs) 
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_loss:.4f}")

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input_values"].to(device)
                input_lengths = batch["seq_lengths"].to(device)
                targets_list = [torch.tensor(transcript_to_indices(t, char_to_idx), dtype=torch.long) 
                                for t in batch["transcripts"]]
                targets = torch.cat(targets_list).to(device)
                target_lengths = torch.tensor([len(t) for t in targets_list], dtype=torch.long).to(device)
                outputs = model(inputs)
                log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
                test_loss += loss.item()
            avg_test_loss = test_loss / len(test_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Test Loss: {avg_test_loss:.4f}")

    

if __name__ == "__main__":
    vocab = "abcdefghijklmnopqrstuvwxyz "
    char_to_idx = {char: i+1 for i, char in enumerate(vocab)}
    idx_to_char = {i: char for char, i in char_to_idx.items()}
    vocab_size = len(vocab) + 1


    csv_file = "torgo_features_paths.csv"
    full_dataset = TorgoASRDataset(csv_file)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate)

    input_dim = 1
    hidden_dim = 256
    model = Model(input_dim=input_dim, hidden_dim=hidden_dim, vocab_size=vocab_size, num_layers=2)
    trainModel(model, train_loader, test_loader, char_to_idx, num_epochs=20, learning_rate=1e-4)


    
            





        
