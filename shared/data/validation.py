"""Validation set loading from the pre-carved parquet file."""
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset


class ValidationDataset(Dataset):
    def __init__(self, parquet_path: str | Path):
        df = pd.read_parquet(parquet_path)
        # Expects columns: 'input_ids', 'labels' (both lists of ints)
        self.inputs = df["input_ids"].tolist()
        self.labels = df["labels"].tolist()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
