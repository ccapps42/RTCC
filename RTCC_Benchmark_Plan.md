# RTCC Benchmark Plan — SUPERSEDED (2026-05-24)

This document specified a **d=1600 benchmark run** compared against Pythia-410M.
The plan was replaced when the project pivoted to a **d=1024 architecture that
directly matches CART**, removing the need for a separately-trained larger
"benchmark" model — RTCC IS the benchmark, compared head-to-head against CART's
existing d=1024 R=6 P=6 results.

## Current sources of truth

- **`CLAUDE.md`** — architecture, hardware, key constants.
- **`memory/project_definite_plan_toroid_coda_topk.md`** at
  `C:\Users\ccapp\.claude\projects\K--projects-RTCC-Paper-2\memory\` —
  the locked plan (20 runs total: 16 ablation + 1 B4 + 2 extra seeds + 1 lm-eval).
- **`README.md`** — public-facing architecture and run plan.

## Why the pivot

The original benchmark plan presumed RTCC would be a different size/shape from
CART and would need its own external comparator (Pythia-410M at 1B-token checkpoint).
The current plan keeps the RTCC backbone **exactly identical to CART d=1024 R=6 P=6**,
changing only the coda FFN. This makes the RTCC-vs-CART comparison directly meaningful
without needing any external comparator — same training data (CART's stage2_train.bin),
same seed, same hyperparameters, same eval bins, same eval budget.

CART's d=1024 R=6 P=6 results are already in `K:\projects\Model_Paper_1\results.db`
across 3 seeds (42, 137, 271) plus 6 diagnostic ablation runs, providing more
comparison surface than a single Pythia checkpoint would.

## What survived

- Llama-2 32K tokenizer.
- MLA attention with KV latent = d/4.
- RMSNorm, SwiGLU, weight-tied embeddings.
- Sequence length 1024, batch=4, grad_accum=8 → 32,768 tokens/step.
- ~1B token training budget (now exactly 30,500 steps to match CART).

External benchmark suite (lm-eval-harness on the winning checkpoint) is still
planned and is the "1 benchmark eval" in the 20-run total — see the plan doc.
