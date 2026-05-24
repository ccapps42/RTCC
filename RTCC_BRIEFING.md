# RTCC Architecture Briefing — SUPERSEDED (2026-05-24)

This document originally onboarded new sessions to a 576-dim sweep + 1600-dim
benchmark plan with ToroidalMoE in the **recurrent core**. The project pivoted
to a **coda-placement** design at **d=1024** with **top-K routing** and a
**16-cell K × overlap ablation**.

## Current sources of truth

For project context, read these instead:

- **`CLAUDE.md`** (project root) — architecture map, shared components, launch path,
  hardware, training data, gotchas, key constants. This is the active onboarding doc.
- **`memory/project_definite_plan_toroid_coda_topk.md`** at
  `C:\Users\ccapp\.claude\projects\K--projects-RTCC-Paper-2\memory\` —
  the locked plan with all design decisions, ablation grid, and rationale.
- **`README.md`** (project root) — public-facing architecture and run plan.
- **`architectures/06_rtcc_coda_topk/`** — the active paper architecture.
- **`configs/paper_1024_rtcc_coda/`** — the 16 ablation cells.

## Why the pivot

- **Recurrent-core toroid was too slow** at d=1024 for the planned sweep budget on a single RTX 3090.
- **Coda placement isolates the toroid contribution** to a single layer, leaving the
  recurrent backbone identical to CART. The RTCC-vs-CART delta then attributes cleanly
  to the coda swap alone — the cleanest possible single-variable comparison.
- **Top-K routing** adds an ablation axis (sparse → dense) that the original
  always-active design lacked.

## What survived from the original briefing

- CART backbone validated at R=6, P=6, d=1024.
- LTI gating, hyper-connections (n=3), LIE.
- MLA attention (DeepSeek-V2), RoPE in prelude/coda only.
- Tokenizer: NousResearch/Llama-2-7b-hf (32K vocab), matching CART.
- The toroidal geometry concept itself — just moved from recurrent core to coda.

The deprecated recurrent-core implementation is preserved at
`architectures/05_rtcc/` (see its NOTE.md).
