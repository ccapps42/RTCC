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


def evaluate_perplexity_bin(model: torch.nn.Module, bin_path: str,
                            device: torch.device, seq_len: int = 1024,
                            batch_size: int = 8, max_batches: int = 50) -> tuple[float, float]:
    """Evaluate perplexity on a CART-format uint16 .bin file.
    Caps at max_batches per eval call to match CART's eval-time budget exactly.
    50 batches x batch_size 8 = 400 sequences = ~410k tokens per source."""
    from shared.data.loader import FixedOrderDataset

    if not Path(bin_path).exists():
        return float("nan"), float("nan")

    model.eval()
    dataset = FixedOrderDataset(bin_path, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()

    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
    return avg_loss, math.exp(avg_loss)
