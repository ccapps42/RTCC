"""Dataset loading — pre-tokenized bin files and HF cache streaming."""
import os
from pathlib import Path
from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


STAGE2_BIN = Path("K:/projects/Model_Paper_1/data/stage2/stage2_train.bin")


class FixedOrderDataset(Dataset):
    """Reads a pre-tokenized uint16 .bin file in fixed order.

    No shuffling — every run sees identical token sequences at identical positions,
    which is required for sweep comparability across architectures.
    Compatible with CART's stage2_train.bin (same Llama-2 32k tokenizer).
    seq_len=1024 matches CART Stage 2 — the bin was interleaved in 1024-token chunks.
    """

    def __init__(self, bin_path: str | Path = STAGE2_BIN, seq_len: int = 1024):
        self.data = np.fromfile(str(bin_path), dtype=np.uint16)
        self.seq_len = seq_len
        self.n_seqs = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        tokens = torch.from_numpy(
            self.data[start:start + self.seq_len + 1].astype(np.int64)
        )
        return tokens[:-1], tokens[1:]

# Prevent any network calls — data is already cached
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


HF_CACHE = Path("K:/projects/Model_Paper_1/data/hf_cache")

DATASETS = {
    "tinystories": ("roneneldan/TinyStories", None),
    "wikipedia":   ("wikimedia/wikipedia", "20231101.en"),
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "fineweb":     ("HuggingFaceFW/fineweb", "sample-10BT"),
}


def load_hf_dataset(name: str, split: str = "train"):
    dataset_name, config = DATASETS[name]
    kwargs = dict(split=split, cache_dir=str(HF_CACHE), streaming=True)
    if config:
        kwargs["name"] = config
    return load_dataset(dataset_name, **kwargs)


class TokenizedStream(IterableDataset):
    """Streams tokenized documents from an HF dataset."""

    def __init__(self, hf_dataset, tokenizer, text_field: str = "text"):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.text_field = text_field

    def __iter__(self):
        for example in self.dataset:
            text = example[self.text_field]
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if ids:
                yield ids
