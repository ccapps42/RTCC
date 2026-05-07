"""Resume a training run from its latest checkpoint.

Usage:
    python scripts/resume_run.py --run-id 1
"""
import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARCH_MAP = {
    "dense_ffn":       ("architectures.01_dense_ffn.model",       "DenseFfnModel",
                        "architectures.01_dense_ffn.config",       "DenseFfnConfig"),
    "standard_moe":    ("architectures.02_standard_moe.model",    "StandardMoEModel",
                        "architectures.02_standard_moe.config",    "StandardMoEConfig"),
    "slicemoe_flat":   ("architectures.03_slicemoe_flat.model",   "SliceMoeFlatModel",
                        "architectures.03_slicemoe_flat.config",   "SliceMoeFlatConfig"),
    "flat_grid_overlap": ("architectures.04_flat_grid_overlap.model", "FlatGridModel",
                          "architectures.04_flat_grid_overlap.config", "FlatGridConfig"),
    "rtcc":            ("architectures.05_rtcc.model",             "RTCCModel",
                        "architectures.05_rtcc.config",             "RTCCConfig"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args()

    db_path = str(PROJECT_ROOT / "db" / "rtcc_experiments.db")
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT run_name, architecture, config_path, checkpoint_dir FROM runs WHERE run_id=?",
        (args.run_id,)
    ).fetchone()
    conn.close()

    if not row:
        print(f"run_id {args.run_id} not found in DB")
        sys.exit(1)

    run_name, arch, config_path, checkpoint_dir = row
    print(f"Resuming run_id={args.run_id}: {run_name} ({arch})")

    import importlib
    model_module, model_cls_name, cfg_module, cfg_cls_name = ARCH_MAP[arch]
    ModelCls = getattr(importlib.import_module(model_module), model_cls_name)
    CfgCls = getattr(importlib.import_module(cfg_module), cfg_cls_name)

    cfg = CfgCls.from_yaml(config_path)
    cfg.config_path = config_path
    cfg.checkpoint_dir = checkpoint_dir

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cfg.hf_cache_dir)

    import torch
    torch.manual_seed(cfg.seed)
    model = ModelCls(cfg)

    from shared.training.trainer import Trainer
    from scripts.launch_run import build_data_iter
    trainer = Trainer(model, cfg, args.run_id)
    data_iter = build_data_iter(cfg, tokenizer)
    trainer.train(data_iter)


if __name__ == "__main__":
    main()
