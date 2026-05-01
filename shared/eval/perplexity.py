"""Validation perplexity evaluation."""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path


def evaluate_perplexity(model: torch.nn.Module, val_parquet: str,
                        device: torch.device, seq_len: int = 512,
                        batch_size: int = 8) -> tuple[float, float]:
    from shared.data.validation import ValidationDataset

    if not Path(val_parquet).exists():
        print(f"WARNING: val parquet not found at {val_parquet}, skipping eval")
        return float("nan"), float("nan")

    model.eval()
    dataset = ValidationDataset(val_parquet)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()

    model.train()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl
