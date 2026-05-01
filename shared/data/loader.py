"""Dataset loading from HF cache. Reads directly — no re-downloading."""
import os
from pathlib import Path
from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset

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
