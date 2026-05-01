"""Privacy sweep — best val_perplexity vs privacy_pct.

Queries: privacy_sweep_summary view.
Output: paper/figures/fig2_privacy_sweep.pdf
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
        SELECT privacy_pct, best_perplexity, n_experts, patch_size, stride
        FROM privacy_sweep_summary
        ORDER BY privacy_pct
    """).fetchall()
    conn.close()

    if not rows:
        print("No privacy sweep data yet.")
        return

    privacy = [r[0] * 100 for r in rows]
    ppl = [r[1] for r in rows]
    labels = [f"ps={r[3]},st={r[4]}\n({r[2]} experts)" for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(privacy, ppl, "o-", color="#CC3311", linewidth=2, markersize=8)
    for x, y, lbl in zip(privacy, ppl, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Privacy % (private-core dims / patch dims)")
    ax.set_ylabel("Best validation perplexity")
    ax.set_title("RTCC privacy sweep (768-dim, RTX 3090)")
    ax.grid(True, alpha=0.3)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig2_privacy_sweep.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2_privacy_sweep.png", bbox_inches="tight", dpi=150)
    print(f"Saved to {OUT}/fig2_privacy_sweep.*")


if __name__ == "__main__":
    main()
