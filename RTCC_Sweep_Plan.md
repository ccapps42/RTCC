# RTCC Sweep Plan — SUPERSEDED (2026-05-24)

This document specified a **d=576 sweep across 7 privacy configs with 4 baselines
(B1-B4) at 500M tokens each**, with ToroidalMoE in the recurrent core and all
experts always active. The project pivoted to a **d=1024 16-cell K × overlap
ablation in the coda** with top-K routing and 1B tokens per cell.

## Current sources of truth

- **`CLAUDE.md`** — architecture, hardware, key constants, launch commands.
- **`memory/project_definite_plan_toroid_coda_topk.md`** at
  `C:\Users\ccapp\.claude\projects\K--projects-RTCC-Paper-2\memory\` —
  the locked plan with the full ablation grid and rationale.
- **`README.md`** — public-facing architecture and run plan.
- **`configs/paper_1024_rtcc_coda/`** — the 16 cell YAMLs (K{2,4,8,16}_O{1,2,3,4}.yaml).
- **`scripts/orchestrate_paper.py`** — auto-discovers and runs the 16 cells.

## Current ablation grid (replaces the 7-config privacy sweep)

| K (top-K of 64 experts) | Overlap (patch_size - stride) | Cells |
|---|---|---|
| 2, 4, 8, 16 | 1, 2, 3, 4 | 4 × 4 = 16 |

`expert_hidden` auto-derives from patch_size at compression ratio 0.4 (rounded to
multiples of 8) so the overlap axis doesn't conflate geometry with per-expert capacity.

## Full 20-run plan

| Group | Count | Notes |
|---|---|---|
| K × overlap ablation cells | 16 | configs in `configs/paper_1024_rtcc_coda/K*.yaml` |
| Dense baseline | 0 (reuses CART) | CART d=1024 R=6 P=6 from `K:\projects\Model_Paper_1\results.db` is the direct comparator |
| B4 flat-grid baseline | 1 | Same geometry as the winning cell with `padding_mode: zeros` |
| Extra seeds at winner | 2 | seeds 137 and 271 for stability evidence |
| lm-eval benchmark | 1 | Inference-only on the winning checkpoint |
| **Total training runs** | **19** | Plus 1 inference-only eval |

B4 and the seed configs are generated post-winner by `scripts/gen_winner_configs.py`.

## Why the pivot

- **d=576 was an arbitrary intermediate**; d=1024 matches CART exactly, removing
  any scaling-extrapolation argument from the paper.
- **Coda placement is faster and cleaner** than recurrent-core placement:
  - Recurrent toroid ran R times per forward; coda toroid runs once. ~6× speedup.
  - Coda swap is a single-line architectural change vs CART, making the RTCC-vs-CART
    delta attributable to one mechanism.
- **Top-K routing adds an ablation axis** (sparse → dense) that the always-active
  design lacked. K becomes the second sweep dimension.
- **CART has 10 data points** at d=1024 R=6 P=6 (3 seeds + 6 ablations + 1 Stage 1
  reference), giving the RTCC-vs-CART comparison rich context across multiple angles.

## What survived from the original sweep

- Overlap as an ablation axis (1, 2, 3, 4 cells).
- B4 flat-grid baseline (same geometry, zero-pad instead of toroidal wrap).
- Extra seeds (42, 137, 271) for stability evidence at the winner.
- Inference-only benchmarks via lm-eval-harness.

The old B1 (Dense FFN), B2 (Standard MoE), B3 (SliceMoE-Flat) baselines from the
576-dim sweep are dropped — they were comparison points for a self-contained
"is RTCC better than other MoE variants?" question. The current framing is
"does swapping CART's dense coda FFN for a toroid top-K MoE change perplexity
under otherwise-identical conditions?" — for which CART itself IS the baseline.
