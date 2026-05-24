"""
Paper run orchestrator — launches N concurrent training runs, refills slots as they finish.

Auto-discovers every YAML in configs/paper_1024_rtcc_coda/ and runs each via the
rtcc_coda_topk arch. As of 2026-05-24 this directory holds the 16-cell K x overlap
ablation; once a winner is picked, gen_winner_configs.py adds B4_<winner>.yaml plus
two seed variants, and a subsequent invocation will pick those up (already-complete
runs are skipped via DB lookup).

Usage:
    python scripts/orchestrate_paper.py             # 1 slot (RTX 3090 is bandwidth-saturated)
    python scripts/orchestrate_paper.py --slots 2   # 2 concurrent slots (not recommended)
    python scripts/orchestrate_paper.py --dry-run   # print queue, don't launch

Child stdout/stderr inherit the parent terminal — step prints land directly in
the orchestrator's window in real time (matches CART's UX). To archive the
full sweep transcript to a file, tee from PowerShell:

    python scripts/orchestrate_paper.py 2>&1 | Tee-Object -FilePath sweep.log

Ctrl-C terminates child processes. NOTE on Windows: subprocess.terminate() is a
hard kill (TerminateProcess), so an in-flight cell loses progress since its last
checkpoint. Plan around checkpoint_every (currently 1000 steps).
"""
import argparse
import re
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
CONFIG_DIR = PROJECT_ROOT / "configs" / "paper_1024_rtcc_coda"
ARCH = "rtcc_coda_topk"


def _cell_sort_key(p: Path):
    """Sort K<n>_O<m> filenames numerically by n then m so K2 comes before K16.
    B4 prefix and seed suffix order is preserved within their groups.
    """
    name = p.stem
    m = re.match(r'^(B4_)?K(\d+)_O(\d+)(?:_seed(\d+))?$', name)
    if not m:
        return (1, 99, 99, 0, name)
    is_b4   = 1 if m.group(1) else 0
    k       = int(m.group(2))
    overlap = int(m.group(3))
    seed    = int(m.group(4)) if m.group(4) else 0
    return (is_b4, k, overlap, seed, name)


def _build_queue() -> list[tuple[str, str, str]]:
    """(arch, config_path, label) for every YAML under CONFIG_DIR."""
    yamls = sorted(CONFIG_DIR.glob("*.yaml"), key=_cell_sort_key)
    queue = []
    for p in yamls:
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        label = p.stem
        queue.append((ARCH, rel, label))
    return queue


QUEUE = _build_queue()


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
    # Inherit parent stdout/stderr so child prints land in the orchestrator's
    # terminal in real time (matches CART). At --slots > 1 this would interleave;
    # this project runs --slots 1 by design.
    proc = subprocess.Popen(
        [sys.executable, "-u", "scripts/launch_run.py", "--arch", arch, "--config", config],
        cwd=str(PROJECT_ROOT),
    )
    return proc


def format_elapsed(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    return f"{h}h {rem//60:02d}m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=1,
                        help="Number of concurrent training runs (default 1; RTX 3090 is "
                             "bandwidth-saturated by one RTCC run, so concurrent slots "
                             "do not add aggregate throughput)")
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
                elapsed = time.time() - t0
                success = (rc == 0)
                completed.append((label, elapsed, success))
                status = "DONE" if success else f"FAILED (exit {rc})"
                print(f"[{now()}] {label}: {status} in {format_elapsed(elapsed)}")
        running = still_running

        # Fill empty slots from queue
        while queue and len(running) < args.slots:
            arch, config, label = queue.pop(0)
            proc = launch(arch, config, label)
            running.append((proc, label, config, time.time()))
            print(f"[{now()}] {label}: launched (PID {proc.pid})"
                  f"  [{len(running)}/{args.slots} slots]")

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
