"""Checkpoint save/load/prune. Saves model + optimizer + step state."""
import os
import shutil
from pathlib import Path
import torch


def save_checkpoint(step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, checkpoint_dir: str, keep_last_n: int = 5) -> Path:
    ckpt_dir = Path(checkpoint_dir) / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
    }, ckpt_dir / "checkpoint.pt")

    _prune_old_checkpoints(checkpoint_dir, keep_last_n)
    return ckpt_dir


def load_checkpoint(checkpoint_dir: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler=None) -> int:
    """Load latest checkpoint. Returns the step number."""
    ckpt_dir = _latest_checkpoint(checkpoint_dir)
    if ckpt_dir is None:
        return 0

    state = torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"])
    if optimizer and state.get("optimizer_state"):
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler and state.get("scheduler_state"):
        scheduler.load_state_dict(state["scheduler_state"])
    return state["step"]


def get_checkpoint_size_mb(ckpt_dir: Path) -> float:
    total = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file())
    return total / (1024 ** 2)


def _latest_checkpoint(checkpoint_dir: str) -> Path | None:
    ckpt_root = Path(checkpoint_dir)
    if not ckpt_root.exists():
        return None
    dirs = sorted(ckpt_root.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return dirs[-1] if dirs else None


def _prune_old_checkpoints(checkpoint_dir: str, keep_last_n: int):
    ckpt_root = Path(checkpoint_dir)
    dirs = sorted(ckpt_root.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    for old in dirs[:-keep_last_n]:
        shutil.rmtree(old, ignore_errors=True)
