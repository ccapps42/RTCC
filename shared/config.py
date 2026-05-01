"""Base config dataclass and YAML loader. All architecture configs inherit BaseConfig."""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class BaseConfig:
    # Model dimensions
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    model_dim: int = 768
    max_seq_len: int = 512
    head_dim: int = 64

    # MLA attention — head counts differ by location
    n_heads_prelude: int = 16     # 16 × 64 = 1024, projected back to 768
    n_heads_recurrent: int = 12   # 12 × 64 = 768, matches model_dim
    n_heads_coda: int = 8         # 8 × 64 = 512, projected back to 768
    mla_latent_dim: int = 192     # KV compression latent dim (~model_dim / 4)

    # Layer counts
    prelude_layers: int = 4
    coda_layers: int = 1
    max_loop_iters: int = 8

    # Recurrent loop components (match CART)
    n_hyper: int = 3
    lie_dim: int = 32
    lti_init_value: float = 0.9

    # Training
    warmup_steps: int = 2000
    lr_max: float = 3.0e-4
    lr_min: float = 3.0e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    batch_size: int = 4
    grad_accum_steps: int = 8
    seed: int = 42
    total_steps: int = 75_000
    checkpoint_every: int = 1000
    keep_last_n_checkpoints: int = 5
    eval_every: int = 5000

    # Run identity
    run_name: str = "unnamed"
    checkpoint_dir: str = ""
    config_path: str = ""
    hardware: str = "rtx3050"

    # Paths
    db_path: str = "K:/projects/RTCC_Paper_2/db/rtcc_experiments.db"
    hf_cache_dir: str = "K:/projects/Model_Paper_1/data/hf_cache"
    val_parquet: str = "K:/projects/RTCC_Paper_2/data/validation/val.parquet"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaseConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save_yaml(self, path: str | Path):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
