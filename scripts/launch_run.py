"""CLI entry point for launching a training run.

Usage:
    python scripts/launch_run.py --arch dense_ffn --config configs/dev_512/smoke.yaml
    python scripts/launch_run.py --arch standard_moe --config configs/dev_512/smoke_moe.yaml
"""
import argparse
import importlib.util
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Maps arch name → (model_file, ModelClass, config_file, ConfigClass, trainer_file_or_None, TrainerClass_or_None)
ARCH_MAP = {
    "dense_ffn": (
        "architectures/01_dense_ffn/model.py", "DenseFfnModel",
        "architectures/01_dense_ffn/config.py", "DenseFfnConfig",
        None, None,
    ),
    "standard_moe": (
        "architectures/02_standard_moe/model.py", "StandardMoEModel",
        "architectures/02_standard_moe/config.py", "StandardMoEConfig",
        "architectures/02_standard_moe/trainer.py", "MoETrainer",
    ),
    "slicemoe_flat": (
        "architectures/03_slicemoe_flat/model.py", "SliceMoeFlatModel",
        "architectures/03_slicemoe_flat/config.py", "SliceMoeFlatConfig",
        None, None,
    ),
    "flat_grid_overlap": (
        "architectures/04_flat_grid_overlap/model.py", "FlatGridModel",
        "architectures/04_flat_grid_overlap/config.py", "FlatGridConfig",
        None, None,
    ),
    "rtcc": (
        "architectures/05_rtcc/model.py", "RTCCModel",
        "architectures/05_rtcc/config.py", "RTCCConfig",
        None, None,
    ),
    "rtcc_coda_topk": (
        "architectures/06_rtcc_coda_topk/model.py", "TopKToroidCodaModel",
        "architectures/06_rtcc_coda_topk/config.py", "TopKToroidCodaConfig",
        "architectures/06_rtcc_coda_topk/trainer.py", "TopKToroidTrainer",
    ),
}


def _load_module(label: str, rel_path: str):
    path = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(label, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[label] = mod
    spec.loader.exec_module(mod)
    return mod


def register_run(cfg, config_path: str) -> int:
    conn = sqlite3.connect(cfg.db_path)
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT OR REPLACE INTO runs
           (run_name, architecture, tier, status, hardware, model_dim,
            vocab_size, max_seq_len, total_steps, batch_size, grad_accum,
            lr_max, lr_min, seed, tokens_target, config_path, checkpoint_dir, started_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            cfg.run_name,
            getattr(cfg, "architecture", "unknown"),
            "dev",
            "queued",
            cfg.hardware,
            cfg.model_dim,
            cfg.vocab_size,
            cfg.max_seq_len,
            cfg.total_steps,
            cfg.batch_size,
            cfg.grad_accum_steps,
            cfg.lr_max,
            cfg.lr_min,
            cfg.seed,
            cfg.total_steps * cfg.batch_size * cfg.grad_accum_steps * cfg.max_seq_len,
            config_path,
            cfg.checkpoint_dir,
            now,
        ),
    )
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def build_data_iter(cfg):
    from shared.data.loader import FixedOrderDataset
    from torch.utils.data import DataLoader
    ds = FixedOrderDataset(seq_len=cfg.max_seq_len)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
                      pin_memory=True, drop_last=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=list(ARCH_MAP.keys()))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    model_file, model_cls, cfg_file, cfg_cls, trainer_file, trainer_cls = ARCH_MAP[args.arch]
    mdl_mod = _load_module(f"_arch_{args.arch}_model", model_file)
    cfg_mod = _load_module(f"_arch_{args.arch}_config", cfg_file)
    ModelCls = getattr(mdl_mod, model_cls)
    CfgCls = getattr(cfg_mod, cfg_cls)

    cfg = CfgCls.from_yaml(args.config)
    cfg.config_path = args.config
    cfg.checkpoint_dir = str(PROJECT_ROOT / "runs" / cfg.run_name / "checkpoints")
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print(f"REMINDER: Disable Windows Update auto-restart before a long run.")
    print(f"Settings > Windows Update > Advanced > Pause updates for 5 weeks\n")
    print(f"\nRun: {cfg.run_name}")
    print(f"Arch: {args.arch} | dim={cfg.model_dim} | steps={cfg.total_steps}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}\n")

    run_id = register_run(cfg, args.config)
    print(f"Registered run_id={run_id} in DB")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cfg.hf_cache_dir)
    # Override vocab_size from actual tokenizer — config default may not match
    cfg.vocab_size = len(tokenizer)

    import torch
    torch.manual_seed(cfg.seed)
    model = ModelCls(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.1f}M")

    if trainer_file:
        trn_mod = _load_module(f"_arch_{args.arch}_trainer", trainer_file)
        TrainerCls = getattr(trn_mod, trainer_cls)
    else:
        from shared.training.trainer import Trainer as TrainerCls
    trainer = TrainerCls(model, cfg, run_id)
    data_iter = build_data_iter(cfg)
    trainer.train(data_iter)


if __name__ == "__main__":
    main()
