# RTCC Architecture Roadmap
*Chad A. Capps — May 2026*

This document tracks the architecture across the paper series:
**Paper 1 (CART)** → **Paper 2 (RTCC minimal)** → **Paper 3 (RTCC full / formerly "V1 spec")** → Future.

The paper 2 goal is minimum deviation from CART so that the toroidal FFN is the only variable.
Paper 3 is the full architecture spec, now relabeled V2.

---

## Paper 1 — CART
*Context-Anchored Recurrent Transformer. In progress (Stage 2 runs active).*

### Architecture

**Three-block design: Prelude → Recurrent Core → Coda**

```
Input tokens
  → Embedding [vocab × d]
  → Prelude (P layers, unique weights, causal MLA self-attention + SwiGLU)
  → e  ← store prelude output as fixed context
  → KV ← compute K, V from e once; reuse across all R loops
  → h = e  ← initialize hidden state from prelude output
  → Recurrent loop × R:
      h_input = hyper.combine(buffer)          ← blend last n_hyper hidden states
      h_input = h_input + LIE(r)               ← inject loop-depth signal
      block_out = CoreBlock(h_input, K, V)     ← cross-attn + SwiGLU FFN
      h = sigmoid(A) * h_input + block_out     ← LTI-stable update
      buffer = hyper.update_buffer(buffer, h)
  → Coda (1 layer, unique weights, causal MLA self-attention + SwiGLU)
  → RMSNorm → logits (tied embedding weight)
```

**Core block (shared weights, looped R times):**
- MLA cross-attention: Q from h, K/V pre-computed from e — no RoPE in core
- SwiGLU FFN: d → ffn_intermediate → d

### Parameters (d_model = 768)

| Component | Value | Notes |
|---|---|---|
| `vocab_size` | 32,000 | Llama-2 BPE tokenizer |
| `d_model` | 768 | Primary sweep dimension |
| `n_heads` | 12 | d_model / d_head = 768 / 64 |
| `d_head` | 64 | Fixed |
| `d_kv_latent` | 192 | d_model / 4 (MLA compression) |
| `ffn_intermediate` | 2,048 | ceil(8d/3 / 256) × 256 |
| `n_prelude` (P) | 4, 6, 8 | Swept; Stage 1 best = 6 |
| `n_loops` (R) | 4, 6, 8, 10 | Swept; Stage 1 best = 6 |
| `n_coda` | 1 | Fixed |
| `n_hyper` | 3 | Ring buffer depth |
| `lie_dim` | 32 | Sinusoidal loop-depth embedding |
| `lti_init_value` | 0.9 | sigmoid_inverse(0.9) stored as a_param |
| `tie_embeddings` | True | Output projection = embedding weight^T |
| `rope_base` | 10,000 | Applied in prelude and coda only |
| `dropout` | 0.0 | |
| `bias` | False | All linear projections |
| `norm` | RMSNorm, eps=1e-6 | Pre-norm throughout |

**LTI formula:** `h = sigmoid(A) * h_input + block_out`
*(A = learnable diagonal, sigmoid-parameterized → spectral radius < 1 guaranteed)*

### Training (Stage 2)

| Setting | Value |
|---|---|
| `seq_len` | 1,024 |
| `batch_size` | 8 |
| `grad_accum` | 4 |
| Effective batch | 32 seqs = 32,768 tokens/step |
| `total_steps` | 30,500 |
| Total tokens | ~1B |
| `warmup_steps` | 100 |
| `lr_max` | 3e-4 |
| `lr_min` | 3e-5 |
| LR schedule | Cosine decay with linear warmup |
| Optimizer | AdamW, β=(0.9, 0.95), wd=0.1, clip=1.0 |
| Seeds | 42, 137, 271 |

**Sweep space (d_model=768):** R ∈ {4, 6, 8, 10} × P ∈ {4, 6, 8} × 3 seeds = 36 configs

---

## Paper 2 — RTCC (Coda Top-K Toroidal MoE)
*Recurrent Toroidal Cortical Columns. Single architectural change from CART: the
dense SwiGLU FFN in the **coda layer** is replaced by an up-projected toroidal
top-K MoE. The CART backbone (prelude + recurrent core + everything else) is
unchanged.*

### What Changes from CART

Only the coda FFN slot. The prelude and recurrent core are byte-for-byte CART.

```
CodaBlock (CART):  MLA self-attn → SwiGLU(1024 → 2816 → 1024)
CodaBlock (RTCC):  MLA self-attn → up_proj(1024 → 4096)
                                → TopKToroidMoE on 64×64 sheet
                                → down_proj(4096 → 1024)
```

The TopKToroidMoE:
- Treats the up-projected residual as a 64×64 **toroidal sheet** (4096 dims)
- 8×8 = **64 position-fixed experts**, each on a `patch_size × patch_size` patch
- **Linear router** picks **top-K of 64 experts** per token
- Each selected expert: SwiGLU on its patch (`patch_dims → expert_hidden → patch_dims`)
- **Toroidal wrap** on both axes (circular padding); zero-pad as a B4 ablation
- **Wrap-add fold** ensures full toroidal symmetry post-fold

All other components — embedding, prelude (6 dense layers), recurrent core (looped
6 times, dense SwiGLU FFN), MLA attention with KV latent = d/4, RoPE in prelude/coda
only, hyper-connections (n=3), LIE, LTI gating, RMSNorm, tied embeddings, Llama-2
32K tokenizer — are inherited from CART unchanged.

### Locked Configuration (single backbone, swept coda)

| Parameter | Value | Source |
|---|---|---|
| `d_model` | 1024 | Matches CART d=1024 exactly |
| Prelude layers (P) | 6 | CART Stage 2 winner |
| Core loops (R) | 6 | CART Stage 2 winner (R* = 6) |
| Coda layers | 1 | CART |
| `n_heads` (prelude/recurrent/coda) | 16 / 16 / 16 | head_dim = 64 |
| `mla_latent_dim` | 256 | d / 4 |
| `vocab_size` | 32,000 | Llama-2 BPE (NousResearch/Llama-2-7b-hf) — CART parity |
| Sheet (in coda only) | 64 × 64 | up_dim = 4096 |
| Stride (in coda) | 8 | → 8 × 8 = 64 experts |
| Aux loss coef | 0.01 | DeepSeek-style load-balance |
| LTI formulation | `h = sigmoid(A)·h_input + transformer_out` | CART |
| Total params | ~125M | At any ablation cell |

### Ablation Grid (16 cells)

**K × overlap = {2, 4, 8, 16} × {1, 2, 3, 4} = 16 training runs**

`patch_size = stride + overlap = 8 + overlap`, so patch sizes are {9, 10, 11, 12}.
`expert_hidden` auto-derives at compression ratio 0.4, rounded to multiples of 8:

| overlap | patch | patch_dims | expert_hidden | compression |
|---|---|---|---|---|
| 1 | 9×9 | 81 | 32 | 0.40 |
| 2 | 10×10 | 100 | 40 | 0.40 |
| 3 | 11×11 | 121 | 48 | 0.40 |
| 4 | 12×12 | 144 | 56 | 0.39 |

This per-cell scaling prevents conflating geometry with per-expert capacity across
the overlap axis.

### Full 20-Run Plan

| Group | Count | Notes |
|---|---|---|
| K × overlap ablation cells | 16 | `configs/paper_1024_rtcc_coda/K{2,4,8,16}_O{1,2,3,4}.yaml` |
| Dense baseline (no new run) | 0 | CART d=1024 R=6 P=6 from `K:\projects\Model_Paper_1\results.db` is the direct comparator (3 seeds + 6 diagnostic ablations) |
| B4 flat-grid baseline | 1 | Same geometry as the winning cell with `padding_mode: zeros` (isolates the toroidal-wrap contribution) |
| Extra seeds at winner | 2 | seeds 137 and 271 for 3-seed stability evidence |
| lm-eval-harness benchmarks | 1 | Inference-only on the winning checkpoint |
| **Total training runs** | **19** | Plus 1 inference-only eval |

B4 and the seed configs are generated post-winner via
`python scripts/gen_winner_configs.py --cell K<X>_O<Y>`.

### Training (matches CART d=1024 Stage 2 exactly)

| Setting | Value |
|---|---|
| `seq_len` | 1,024 |
| `batch_size` | 4 |
| `grad_accum` | 8 |
| Effective batch | 32 sequences = 32,768 tokens/step |
| `total_steps` | 30,500 |
| Total tokens | ~1B (matches CART) |
| `warmup_steps` | 2,000 |
| `lr_max` | 3e-4 |
| `lr_min` | 3e-5 |
| LR schedule | Cosine decay with linear warmup |
| Optimizer | AdamW8bit (bitsandbytes) |
| Eval frequency | every 2,500 steps |
| Checkpoint frequency | every 1,000 steps |
| Training data | `K:\projects\Model_Paper_1\data\stage2\stage2_train.bin` (FixedOrderDataset, same seed/order as CART) |
| Validation | CART val bins (tiny/wiki/edu, 50 batches each) |
| Hardware | RTX 3090 (single, slots=1) |

### Status

Code complete, sweep launched as of 2026-05-24. First cell (K2_O1) was achieving
wiki PPL parity with CART by step 5,000 and beating CART by ~5% at step 10,000.
Expected sweep wall-clock: ~11 days at slots=1.

---

## Paper 3 — RTCC Full (formerly "V1 Spec", now "V2")
*Full RTCC architecture. Builds on Paper 2's validated toroidal claim with a larger, more expressive design.*

### Architecture Changes from Paper 2

| Component | Paper 2 (RTCC minimal) | Paper 3 (RTCC V2) |
|---|---|---|
| `d_model` | 576 (sweep) / 768 (CART cmp) | 1,024 |
| Input embedding | Direct `[32K × 576]` | Factored `[32K × 512] + [512 × 1024]` (−15.9M params) |
| Unembedding | Direct `[576 × 32K]` | Factored `[1024 × 768] + [768 × 32K]` (−7.4M params) |
| Prelude attention | MLA self-attn | GQA self-attn, 4:1 KV ratio (8Q / 2KV heads) |
| Prelude attention pattern | Full causal | Sliding window W=2,048 + 64 global sink tokens |
| Sink tokens | None | 64 learned prefix tokens |
| Prelude FFN expansion | ~2.67× (8/3×) | 6× |
| Prelude layers (P) | Swept {4,6,8} | Fixed P=6 |
| Core attention | MLA cross-attn (Q=h, KV=e) | MLA self-attn (full causal) |
| Expert sheet | In-place 24×24 grid (no proj) | Projected: 1024 → 5,184 (72×72) via proj_up |
| Expert grid | n×n = 9–64 experts, square (swept) | 8×8 = 64 experts (square symmetric) |
| Expert patch | Varies across sweep; see Sweep Plan | 10×10, stride=9, overlap=1 |
| Expert FFN hidden | 256 | 256 |
| Expert init | Uniform std=0.02 | Staggered: std + δ×expert_idx, δ=0.001 |
| Overlap aggregation | F.fold sum | Learned softmax-weighted sum (2,304 scalars) |
| Gated residual bypass | Block residual only | Explicit: `h_new = h + proj_down(expert_out)` |
| LTI formula | `A·h + block_out` | `A·h_new + B·e` (B·e injection restored; cross-attn removed) |
| Prelude output norm | None | RMSNorm on e before injection |
| LIE | Sinusoidal projected, lie_dim=32 | Lookup table [16 × 1024], sinusoidal init |
| Loop count (R) | Swept {4,6,8,10} | Fixed R=16 |
| Early exit probe | None | Confidence scalar, BCE loss λ=0.1, min_loops=4 |
| Optimizer | Pure AdamW | Hybrid Muon (2D weights) + AdamW (embeddings, norms, 1D) |
| RoPE | Standard, base=10,000 | YaRN, base=500,000, scale=4 (inference to ~32K) |
| Weight init | Uniform std=0.02 | Depth-scaled: prelude σ=0.02/√12, coda σ=0.02/√2 |
| Gradient strategy | Full backprop | Truncated BPTT K=8, stop-grad on e |
| Training tokens | ~1B (match CART) | ~15B (10B Phase 1 @ 2K, 5B Phase 2 @ 8K) |
| Training phases | 1 | 2 (Phase 2 freezes prelude, unfreezes probe) |
| Hardware target | RTX 3050 / RTX 3090 | Single RTX 3090 (24GB) |

### Paper 3 Parameter Summary (d_model = 1,024, R = 16)

| Block | Real Params | Loops | Effective Params |
|---|---|---|---|
| Input embeddings (factored 512) | 16.91M | 1 | 16.91M |
| Learned sink tokens | 0.07M | 1 | 0.07M |
| Prelude (6 layers, 6× FFN) | 129.06M | 1 | 129.06M |
| Recurrent core | 16.95M | 16 | 271.20M |
| Coda (1 layer, 4× FFN) | 13.95M | 1 | 13.95M |
| Output unembedding (768 bottleneck) | 25.37M | 1 | 25.37M |
| **Total** | **~202M** | | **~457M** |

### Paper 3 Required Ablations

| Ablation | What changes | What it answers |
|---|---|---|
| Baseline-Dense | ToroidalMoE → dense FFN (matched params) | Does the torus beat a same-size dense FFN? |
| Baseline-MoE | ToroidalMoE → vanilla top-2-of-16 MoE | Does the torus beat standard MoE? |
| NoOverlap | stride=10 (no overlap, no torus wrap) | Is overlap load-bearing? |
| No-Wrap | overlap kept, zero-pad boundary | Is torus topology specifically what helps? |
| NoResidual | Remove gated residual bypass | Quantify bypass contribution |
| Overlap-2 | 2-cell overlap, stride=8, sheet=64×64 | Is 2-cell overlap better than 1-cell? |
| Overlap-3 | 3-cell overlap, stride=7, sheet=56×56 | Does trend continue toward biology? |
| Loop-count sweep | R ∈ {4, 8, 12, 16} on fixed checkpoint | Establish loop-depth scaling curve |

Ablation budget: ~7 runs × ~3.5 days = ~25 days + 1 headline run ≈ 4 weeks on one RTX 3090.

---

## Future Work — Deferred from Paper 3 Spec (Section 11)

Items below were evaluated for Paper 3 and deferred with explicit conditions for reconsideration.

### Deferred to Paper 4 / V3

| # | Item | Brief description | Reconsider when |
|---|---|---|---|
| 11.1 | **SSM (Mamba) in core** | State-space model alongside or replacing core attention | Paper 3 landed; v3 can ablate SSM cleanly against validated toroidal baseline |
| 11.2 | **Manifold-constrained hyper-connections (mHC)** | Stiefel-manifold Riemannian SGD for hyper-connection weights | Scaling to ≥1B params where mHC gains were observed |
| 11.3 | **Depth-wise LoRA per loop** | Per-loop LoRA adapters on shared core weights | K increased to cover all loops, or LoRA restricted to loops 8–15 by design |
| 11.4 | **DeepSeek Sparse Attention (DSA)** | Lightning-indexer sparse attention for long context | v3 extends training to ≥16K; DSA becomes structurally necessary at that scale |
| 11.5 | **Engram Conditional Memory** | External key-value memory validated at 27B params | Scaling beyond 1B params, or data shifts toward >20% encyclopedic content |
| 11.6 | **16K context (Phase 3)** | Long-context extension training | Simultaneous with DSA inclusion |
| 11.7 | **Coda FFN 6× expansion** | Upsize coda FFN from 4× to 6× | Paper 3 ablation slot available; stronger prior if prelude 6× shows clear benefit |
| 11.8 | **TIE — Token Index Embedding re-injection** | Re-inject token position into every loop iteration | Paper 3 shows LIE alone provides insufficient loop differentiation |
| 11.9 | **Buffer_alpha (sliding window buffer)** | Frozen 7-token compressed buffer injected per loop | Paper 3 cleared; multi-frequency framing becomes a paper topic; or clause-level coherence underperforms |
| 11.10 | **Hierarchical State Carry (HSC bands)** | 5-band EMA states updated at multiple timescales (1→2003 tokens), injected per loop | v3 has ≥16K context and long-document data; HSC_fast may move earlier if attention insufficient for in-context running summary |
| 11.11 | **Two-Level Toroidal Hierarchy** | Second 32×32 toroidal sheet post-loop, top-k routed, 4:1 spatial compression from level-1 | Paper 3 landed — this is large enough for its own paper; geometry extends cleanly to 3-level cortical hierarchy (mini → macro → hyper column) |
| 11.12 | **Phase Coupling Gate** | Slow HSC state gates amplitude of LIE signal (cross-frequency modulation) | HSC bands stable in v3; phase coupling is the natural "completing the cortical analogy" addition |
| 11.13 | **Aggressive overlap (≥4 cells)** | Overlap=4, private core=2×2=4 dims (4%), testing the biology-dominant hypothesis | Paper 3 overlap sweep shows overlap=3 outperforming overlap=2 (trend continues into biological regime) |

### Notes on Section 11 Items

**11.11 (Two-Level Toroidal Hierarchy)** is the strongest candidate for its own standalone paper.
The geometry extends cleanly: level-1 = minicolumns (64 experts, 5184-dim sheet),
level-2 = macrocolumns (16 experts, 1024-dim sheet, 4:1 spatial compression), level-3 = hypercolumns
(4 experts, 256-dim). This three-level hierarchy maps directly onto the known cortical hierarchy
and could carry a paper in its own right.

**11.9 + 11.10 + 11.12** (Buffer_alpha, HSC bands, Phase Coupling Gate) form a coherent
multi-frequency / cortical-rhythm cluster. These are best introduced together as a unified
"multi-timescale RTCC" paper rather than incrementally.

---

## Summary Table — Paper Series

| | Paper 1 (CART) | Paper 2 (RTCC, coda Top-K Toroidal MoE) | Paper 3 (RTCC V2 full) |
|---|---|---|---|
| Novel claim | Recurrent depth > feed-forward depth at equal params | Toroidal Top-K MoE in coda > dense FFN coda under otherwise-identical CART backbone | Full RTCC with expanded sheet, GQA+SWA prelude, Muon, YaRN |
| What changes | — | Coda FFN only (prelude + recurrent core = CART exactly) | Multi-axis (recurrent core sheet, prelude attn, optimizer, RoPE) |
| FFN placement | SwiGLU in prelude/core/coda | SwiGLU in prelude/core; **TopKToroidMoE in coda** | TopKToroidMoE in recurrent core (full V2) |
| d_model | 256–1024 (swept) | **1024** (locked to CART) | 1,024 |
| Vocab | 32,000 | 32,000 (Llama-2 BPE) | 32,000 |
| Loop count | Swept | **R = 6** (locked to CART Stage 2 winner) | 16 |
| Coda sheet | — | Up-projected 1024 → 4096 = 64×64 toroidal, 8×8 = 64 experts | Recurrent-core 1024 → 5184 |
| Ablation grid | — | **K × overlap = {2,4,8,16} × {1,2,3,4} = 16 cells** | NoOverlap / No-Wrap / NoResidual / overlap sweep / R sweep |
| Routing | — | **Linear router, top-K of 64 experts** | Top-K (K TBD) |
| Optimizer | AdamW | AdamW8bit (bitsandbytes) | Hybrid Muon + AdamW |
| RoPE | Standard | Standard | YaRN |
| Training tokens | ~1B | **~1B (30,500 steps, matches CART exactly)** | ~15B |
| Hardware | RTX 3050 / 3090 | RTX 3090 (slots=1) | RTX 3090 |
| Status | Complete (Stage 2 + diagnostic ablations) | **Sweep launched 2026-05-24, ~11 days wall-clock** | Planned post Paper 2 |

*— Chad A. Capps, updated 2026-05-24*
