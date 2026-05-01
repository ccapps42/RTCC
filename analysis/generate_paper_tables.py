"""Generate LaTeX tables from DB results.

Tables:
  1. Tier 1 architecture comparison (5 architectures)
  2. Privacy/overlap sweep (5 configs)
  3. Column depth ablation (R=2,4,8)
  4. lm-eval benchmark results

Output: paper/tables/*.tex
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DB = ROOT / "db" / "rtcc_experiments.db"
OUT = ROOT / "paper" / "tables"


ARCH_LABELS = {
    "dense_ffn": "Dense FFN",
    "standard_moe": "Standard MoE",
    "slicemoe_flat": "SliceMoE Flat",
    "flat_grid_overlap": "Flat Grid (zero-pad)",
    "rtcc": r"\textbf{RTCC (ours)}",
}


def table_tier1(conn):
    rows = conn.execute("""
        SELECT r.architecture,
               MIN(c.val_perplexity) AS best_ppl,
               MIN(c.val_loss) AS best_loss,
               SUM(r.tokens_trained) AS tokens
        FROM runs r
        JOIN checkpoints c ON r.run_id = c.run_id
        WHERE r.model_dim = 768 AND r.tier IN ('baseline', 'dev')
          AND c.val_perplexity IS NOT NULL
        GROUP BY r.architecture
        ORDER BY best_ppl
    """).fetchall()

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Architecture comparison on validation perplexity (768-dim, ~200M tokens each)}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Architecture & Val. Perplexity & Val. Loss \\",
        r"\midrule",
    ]
    for arch, ppl, loss, _ in rows:
        label = ARCH_LABELS.get(arch, arch)
        lines.append(f"{label} & {ppl:.2f} & {loss:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_privacy_sweep(conn):
    rows = conn.execute("""
        SELECT patch_size, stride, n_experts, privacy_pct, best_perplexity, best_val_loss
        FROM privacy_sweep_summary
        ORDER BY privacy_pct
    """).fetchall()

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Overlap width sweep (768-dim, 75k steps each; bold = primary config)}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Patch & Stride & Experts & Privacy\% & Val. PPL \\",
        r"\midrule",
    ]
    for ps, st, ne, priv, ppl, _ in rows:
        pct = f"{priv*100:.0f}\\%"
        lines.append(f"{ps}×{ps} & {st} & {ne} & {pct} & {ppl:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    conn = sqlite3.connect(DB)
    OUT.mkdir(parents=True, exist_ok=True)

    t1 = table_tier1(conn)
    (OUT / "table1_tier1.tex").write_text(t1)
    print("Wrote table1_tier1.tex")

    t2 = table_privacy_sweep(conn)
    (OUT / "table2_privacy_sweep.tex").write_text(t2)
    print("Wrote table2_privacy_sweep.tex")

    conn.close()


if __name__ == "__main__":
    main()
