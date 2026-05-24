"""gen_winner_configs.py — generate B4 + extra-seed configs from a winning ablation cell.

Run this once the 16-cell K x overlap sweep is complete and a winner is identified.
Emits the three remaining training configs for the 20-run plan:

    B4_{cell}.yaml         — identical to the winner but padding_mode=zeros
                             (flat-grid; isolates the toroidal-boundary contribution)
    {cell}_seed137.yaml    — same as winner with seed=137 (stability evidence)
    {cell}_seed271.yaml    — same as winner with seed=271 (stability evidence)

Usage:
    python scripts/gen_winner_configs.py --cell K2_O1

Refuses to overwrite existing files; delete them first if intentional.
"""
import argparse
import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" / "paper_1024_rtcc_coda"
EXTRA_SEEDS = (137, 271)


def load_cell(cell: str) -> dict:
    path = CONFIG_DIR / f"{cell}.yaml"
    if not path.exists():
        sys.exit(f"ERROR: cell {cell!r} not found at {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict, header: str):
    if path.exists():
        sys.exit(f"ERROR: refusing to overwrite existing {path}\n"
                 f"       delete it first if intentional.")
    with open(path, "w") as f:
        f.write(header.rstrip() + "\n")
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")


def emit_b4(cell: str, base: dict):
    out = dict(base)
    out["padding_mode"] = "zeros"
    out["run_name"] = f"paper_1024_rtcc_coda/B4_{cell}"
    header = (
        f"# B4 flat-grid baseline for the winning cell {cell}\n"
        f"# Identical to {cell}.yaml except padding_mode=zeros (no toroidal wrap).\n"
        f"# Any delta vs {cell} is attributable to the toroidal boundary.\n"
    )
    write_yaml(CONFIG_DIR / f"B4_{cell}.yaml", out, header)


def emit_seed(cell: str, base: dict, seed: int):
    out = dict(base)
    out["seed"] = seed
    out["run_name"] = f"paper_1024_rtcc_coda/{cell}_seed{seed}"
    header = (
        f"# Extra seed for stability evidence at the winning cell {cell}\n"
        f"# Identical to {cell}.yaml except seed={seed}.\n"
    )
    write_yaml(CONFIG_DIR / f"{cell}_seed{seed}.yaml", out, header)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cell", required=True,
                    help="Winning ablation cell name (e.g., K2_O1, K4_O3).")
    args = ap.parse_args()

    base = load_cell(args.cell)
    print(f"Generating derived configs from {args.cell}.yaml:")
    emit_b4(args.cell, base)
    for s in EXTRA_SEEDS:
        emit_seed(args.cell, base, s)
    print("Done.")


if __name__ == "__main__":
    main()
