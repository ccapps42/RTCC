"""Loss curves for all Tier 1 architectures — tokens_seen vs loss.

Queries: steps table for all paper_768 tier='dev' or tier='baseline' runs.
Output: paper/figures/fig1_loss_curves.pdf
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DB = ROOT / "db" / "rtcc_experiments.db"
OUT = ROOT / "paper" / "figures"

ARCH_ORDER = ["dense_ffn", "standard_moe", "slicemoe_flat", "flat_grid_overlap", "rtcc"]
ARCH_LABELS = {
    "dense_ffn": "Dense FFN",
    "standard_moe": "Standard MoE",
    "slicemoe_flat": "SliceMoE Flat",
    "flat_grid_overlap": "Flat Grid Overlap",
    "rtcc": "RTCC (ours)",
}
ARCH_COLORS = {
    "dense_ffn": "#666666",
    "standard_moe": "#4477AA",
    "slicemoe_flat": "#EE6677",
    "flat_grid_overlap": "#228833",
    "rtcc": "#CC3311",
}


def main():
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    conn = sqlite3.connect(DB)
    fig, ax = plt.subplots(figsize=(8, 5))

    for arch in ARCH_ORDER:
        rows = conn.execute("""
            SELECT s.tokens_seen, s.loss
            FROM steps s
            JOIN runs r ON s.run_id = r.run_id
            WHERE r.architecture = ? AND r.model_dim = 768
              AND r.tier IN ('baseline', 'dev')
            ORDER BY s.tokens_seen
        """, (arch,)).fetchall()

        if not rows:
            print(f"  [skip] no data for {arch}")
            continue

        tokens = [r[0] / 1e6 for r in rows]  # millions
        loss = [r[1] for r in rows]
        ax.plot(tokens, loss, label=ARCH_LABELS[arch], color=ARCH_COLORS[arch], linewidth=1.5)

    conn.close()

    ax.set_xlabel("Tokens seen (millions)")
    ax.set_ylabel("Training loss")
    ax.set_title("Architecture comparison — loss curves (768-dim)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig1_loss_curves.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_loss_curves.png", bbox_inches="tight", dpi=150)
    print(f"Saved to {OUT}/fig1_loss_curves.*")


if __name__ == "__main__":
    main()
