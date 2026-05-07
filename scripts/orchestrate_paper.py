"""
Paper run orchestrator — launches N concurrent training runs, refills slots as they finish.

Usage:
    python scripts/orchestrate_paper.py            # 2 concurrent slots
    python scripts/orchestrate_paper.py --slots 3  # 3 concurrent slots
    python scripts/orchestrate_paper.py --dry-run  # print queue, don't launch

Logs for each run are written to runs/paper_576/logs/<label>.log.
Ctrl-C cleanly terminates all child processes.
"""
import argparse
import signal
import sqlite3
import subprocess
import sys
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "db" / "rtcc_experiments.db"
LOG_DIR = PROJECT_ROOT / "runs" / "paper_576" / "logs"

# Run queue in phase order (RTCC_Sweep_Plan.md)
QUEUE = [
    # Phase 1 — anchors
    ("dense_ffn",        "configs/paper_576/B1_dense_ffn.yaml",         "B1_dense_ffn"),
    ("rtcc",             "configs/paper_576/S7_rtcc_60pct.yaml",         "S7_rtcc_60pct"),
    # Phase 2 — baselines
    ("standard_moe",     "configs/paper_576/B2_standard_moe.yaml",       "B2_standard_moe"),
    ("slicemoe_flat",    "configs/paper_576/B3_slicemoe_flat.yaml",      "B3_slicemoe_flat"),
    ("flat_grid_overlap","configs/paper_576/B4_flat_grid_overlap.yaml",  "B4_flat_grid_overlap"),
    # Phase 3 — sweep
    ("rtcc",             "configs/paper_576/S1_rtcc_11pct.yaml",         "S1_rtcc_11pct"),
    ("rtcc",             "configs/paper_576/S2_rtcc_25pct_1cell.yaml",   "S2_rtcc_25pct_1cell"),
    ("rtcc",             "configs/paper_576/S3_rtcc_25pct_2cell.yaml",   "S3_rtcc_25pct_2cell"),
    ("rtcc",             "configs/paper_576/S4_rtcc_36pct_1cell.yaml",   "S4_rtcc_36pct_1cell"),
    ("rtcc",             "configs/paper_576/S5_rtcc_36pct_2cell.yaml",   "S5_rtcc_36pct_2cell"),
    ("rtcc",             "configs/paper_576/S6_rtcc_51pct.yaml",         "S6_rtcc_51pct"),
]


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def run_name_from_config(config_path: str) -> str:
    with open(PROJECT_ROOT / config_path) as f:
        return yaml.safe_load(f)["run_name"]


def is_complete(run_name: str) -> bool:
    if not DB_PATH.exists():
        return False
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute(
        "SELECT status FROM runs WHERE run_name=? ORDER BY run_id DESC LIMIT 1",
        (run_name,)
    ).fetchone()
    conn.close()
    return row is not None and row[0] == "complete"


def launch(arch: str, config: str, label: str) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{label}.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "scripts/launch_run.py", "--arch", arch, "--config", config],
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=log_file,
    )
    proc._log_file = log_file
    proc._log_path = log_path
    return proc


def format_elapsed(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    return f"{h}h {rem//60:02d}m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=2,
                        help="Number of concurrent training runs (default 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print queue without launching anything")
    args = parser.parse_args()

    queue = list(QUEUE)  # remaining runs to start
    running = []         # list of (proc, label, config, start_time)
    completed = []       # (label, elapsed, success)
    start_wall = time.time()

    # Pre-check: skip already-complete runs
    skipped = []
    filtered = []
    for arch, config, label in queue:
        rname = run_name_from_config(config)
        if is_complete(rname):
            skipped.append(label)
        else:
            filtered.append((arch, config, label))
    queue = filtered

    print(f"\n{'='*60}")
    print(f"RTCC Paper Sweep Orchestrator")
    print(f"Slots: {args.slots}  |  Runs queued: {len(queue)}  |  Skipped (complete): {len(skipped)}")
    if skipped:
        print(f"Already complete: {', '.join(skipped)}")
    print(f"Logs: {LOG_DIR}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("DRY RUN — queue order:")
        for i, (arch, config, label) in enumerate(queue, 1):
            print(f"  {i:2d}. [{arch}] {label}")
        return

    if not queue:
        print("Nothing to run.")
        return

    # Graceful Ctrl-C: terminate all children
    def _sigint(sig, frame):
        print(f"\n[{now()}] Interrupted — terminating {len(running)} running process(es)...")
        for proc, label, config, t0 in running:
            proc.terminate()
            proc._log_file.close()
            print(f"  Terminated: {label}")
        sys.exit(1)
    signal.signal(signal.SIGINT, _sigint)

    while queue or running:
        # Check for finished processes
        still_running = []
        for proc, label, config, t0 in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((proc, label, config, t0))
            else:
                proc._log_file.close()
                elapsed = time.time() - t0
                success = (rc == 0)
                completed.append((label, elapsed, success))
                status = "DONE" if success else f"FAILED (exit {rc})"
                print(f"[{now()}] {label}: {status} in {format_elapsed(elapsed)}"
                      f"  →  log: {proc._log_path.name}")
        running = still_running

        # Fill empty slots from queue
        while queue and len(running) < args.slots:
            arch, config, label = queue.pop(0)
            proc = launch(arch, config, label)
            running.append((proc, label, config, time.time()))
            print(f"[{now()}] {label}: launched (PID {proc.pid})"
                  f"  [{len(running)}/{args.slots} slots]"
                  f"  →  {LOG_DIR.name}/{label}.log")

        # Status line every 5 minutes
        if running:
            total_done = len(completed) + len(skipped)
            total_all = total_done + len(running) + len(queue)
            wall = time.time() - start_wall
            active = ", ".join(label for _, label, _, _ in running)
            eta_str = ""
            if completed:
                avg_sec = sum(e for _, e, _ in completed) / len(completed)
                remaining = len(running) + len(queue)
                eta_sec = (remaining * avg_sec) / args.slots
                eta_str = f"  |  ETA ~{format_elapsed(eta_sec)}"
            print(f"[{now()}] Running: {active}  ({total_done}/{total_all} done){eta_str}")

        if queue or running:
            time.sleep(300)  # poll every 5 minutes

    # Summary
    print(f"\n{'='*60}")
    print(f"All runs complete.  Wall time: {format_elapsed(time.time() - start_wall)}")
    n_ok = sum(1 for _, _, ok in completed if ok)
    n_fail = sum(1 for _, _, ok in completed if not ok)
    print(f"Completed: {n_ok}  Failed: {n_fail}  Skipped: {len(skipped)}")
    if n_fail:
        print("Failed runs:")
        for label, elapsed, ok in completed:
            if not ok:
                print(f"  {label}  →  check {LOG_DIR}/{label}.log")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
