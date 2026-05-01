"""Depth sweep — best val_perplexity vs recurrent loop count R.

Queries: eval_results joined with runs for depth_sweep tier.
Output: paper/figures/fig3_depth_sweep.pdf
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DB = ROOT / "db" / "rtcc_experiments.db"
OUT = ROOT / "paper" / "figures"


def main():
    import matplotlib.pyplot as plt

    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT r.max_loop_iters, MIN(c.val_perplexity)
        FROM runs r
        JOIN checkpoints c ON r.run_id = c.run_id
        WHERE r.tier = 'depth_sweep' AND r.architecture = 'rtcc'
          AND c.val_perplexity IS NOT NULL
        GROUP BY r.max_loop_iters
        ORDER BY r.max_loop_iters
    """).fetchall()
    conn.close()

    if not rows:
        print("No depth sweep data yet.")
        return

    r_vals = [row[0] for row in rows]
    ppl = [row[1] for row in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_vals, ppl, "o-", color="#CC3311", linewidth=2, markersize=8)
    ax.set_xlabel("Recurrent loop count R")
    ax.set_ylabel("Best validation perplexity")
    ax.set_title("RTCC depth sweep (768-dim)")
    ax.set_xticks(r_vals)
    ax.grid(True, alpha=0.3)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig3_depth_sweep.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_depth_sweep.png", bbox_inches="tight", dpi=150)
    print(f"Saved to {OUT}/fig3_depth_sweep.*")


if __name__ == "__main__":
    main()
