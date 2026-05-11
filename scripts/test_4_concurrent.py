"""Launch 4 concurrent RTCC runs at batch=4 to verify VRAM fits on a single 3090.

Each run is a quick 100-step VRAM probe. Generates temp configs with unique
run_names so the runs don't collide in DB/checkpoint storage. Captures the
peak VRAM from each run's per-50-step print line and reports the total.

Usage:
    python scripts/test_4_concurrent.py
"""
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" / "dev_576"
LOG_DIR = PROJECT_ROOT / "runs" / "dev_576" / "test_4conc_logs"

BASE_CONFIG = {
    "model_dim": 576,
    "n_heads_prelude": 9,
    "n_heads_recurrent": 9,
    "n_heads_coda": 9,
    "mla_latent_dim": 144,
    "max_seq_len": 1024,
    "prelude_layers": 6,
    "coda_layers": 1,
    "max_loop_iters": 8,
    "total_steps": 100,
    "warmup_steps": 10,
    "batch_size": 4,
    "grad_accum_steps": 8,
    "checkpoint_every": 200,  # won't trigger
    "eval_every": 200,        # won't trigger
    "hardware": "rtx3090",
    "grid_rows": 24,
    "grid_cols": 24,
    "patch_size": 9,
    "stride": 8,
    "expert_hidden": 512,
}

VRAM_RE = re.compile(r"vram\s+([\d.]+)GB")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs = []

    for slot in ("a", "b", "c", "d"):
        cfg = dict(BASE_CONFIG)
        cfg["run_name"] = f"dev_576/vram_test_4conc_{slot}"
        cfg_path = CONFIG_DIR / f"vram_test_4conc_{slot}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        log_path = LOG_DIR / f"slot_{slot}.log"
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            [sys.executable, "scripts/launch_run.py",
             "--arch", "rtcc",
             "--config", str(cfg_path.relative_to(PROJECT_ROOT))],
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        proc._log_path = log_path
        proc._log_file = log_file
        proc._slot = slot
        procs.append(proc)
        print(f"[launch] slot {slot}: PID {proc.pid}  log: {log_path}")

    print(f"\nWaiting for 4 concurrent runs to finish (~5 minutes)...\n")
    t0 = time.time()
    for proc in procs:
        proc.wait()
        proc._log_file.close()
    elapsed = time.time() - t0
    print(f"All runs finished in {elapsed:.0f}s\n")

    print(f"{'='*60}\nVRAM SUMMARY\n{'='*60}")
    peaks = []
    for proc in procs:
        text = proc._log_path.read_text(encoding="utf-8", errors="ignore")
        matches = VRAM_RE.findall(text)
        peak = max((float(m) for m in matches), default=None)
        peaks.append(peak)
        rc = proc.returncode
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        if peak is None:
            print(f"  slot {proc._slot}: {status}  (no VRAM reading parsed — check log)")
        else:
            print(f"  slot {proc._slot}: {status}  peak VRAM = {peak:.2f} GB")

    valid = [p for p in peaks if p is not None]
    if len(valid) == 4:
        total = sum(valid)
        print(f"\nTotal across 4 slots: {total:.2f} GB / 24 GB available")
        if total < 22:
            print(f"VERDICT: 4 concurrent at batch=4 FITS (~{24 - total:.1f} GB headroom).")
        elif total < 24:
            print(f"VERDICT: 4 concurrent at batch=4 fits but tight (~{24 - total:.1f} GB headroom).")
        else:
            print(f"VERDICT: 4 concurrent at batch=4 DOES NOT FIT.")


if __name__ == "__main__":
    main()
