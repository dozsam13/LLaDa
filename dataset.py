from torch.utils.data import Dataset
import requests
import torch

class SimpleTextDataset(Dataset):
    def __init__(self, text, tokenizer, device, seq_length=128, stride=None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride or seq_length
        encoded_text = tokenizer.encode(text)
        self.device = device

        self.sequences = []
        for i in range(0, len(encoded_text) - seq_length + 1, self.stride):
            sequence = encoded_text[i:i + seq_length]
            self.sequences.append(sequence)

        self.num_tokens = len(encoded_text)
        self.num_sequences = len(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], device=self.device, dtype=torch.int)

def create_dataset_from_text(raw_text, tokenizer, context_size, device):
    split_idx = int(0.8 * len(raw_text))
    train_text = raw_text[:split_idx]
    val_text = raw_text[split_idx:]

    train_dataset = SimpleTextDataset(train_text, tokenizer, device, seq_length=context_size, stride=context_size)
    val_dataset = SimpleTextDataset(val_text, tokenizer, device, seq_length=context_size, stride=context_size//2)

    print(f"Train datapoints: {len(train_dataset.sequences):,} sequences of length {context_size}")
    print(f"Validation datapoints: {len(val_dataset.sequences):,} sequences of length {context_size}")

    return train_dataset, val_dataset


def download_text(url):
    response = requests.get(url)
    response.raise_for_status()

    return response.text