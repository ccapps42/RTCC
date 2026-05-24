# DEPRECATED as the Paper 2 architecture (2026-05-23)

This directory implements the original RTCC concept: a toroidal MoE in the
**recurrent core**, with no routing — all 64 experts active every iteration.

It has been **superseded for Paper 2** by `architectures/06_rtcc_coda_topk/`,
which moves the toroid into the **coda** with top-K routing.

## Why

1. **Throughput.** The recurrent-core toroid pays the F.unfold/F.fold cost
   `R` times per forward pass (once per loop iteration). At d=1024 with R=6
   on a single RTX 3090, the planned 20-run sweep was too slow.
2. **Comparison precision.** With the toroid in the recurrent core, an
   RTCC-vs-CART delta confounds the toroid contribution with `R` rounds of
   toroidal processing. Coda placement isolates the toroid to a single
   layer, leaving the recurrent backbone identical to CART. The vs-CART
   delta then attributes cleanly to the coda swap.
3. **Routing.** No routing in the recurrent-core design means every expert
   is always active. The coda-placement design adds top-K routing as an
   ablation axis, which is one of the paper's central comparisons.

## Status

Code preserved here for reference. The Paper 2 architecture, configs, and
sweep all live under `06_rtcc_coda_topk/`. See
`memory/project_definite_plan_toroid_coda_topk.md` for the active plan.
