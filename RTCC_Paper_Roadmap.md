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

## Paper 2 — RTCC (Minimal)
*Recurrent Toroidal Cortical Columns. Single architectural change from CART: SwiGLU FFN in the recurrent core is replaced by ToroidalMoE. Everything else is identical.*

### What Changes from CART

Only the FFN module inside the recurrent core block changes.

```
CoreBlock (CART):    cross-attn(h, K, V) → SwiGLU FFN(d → 2048 → d)
CoreBlock (RTCC):    cross-attn(h, K, V) → ToroidalMoE(d → d)
```

All other blocks (prelude, coda), all training parameters, and the full loop mechanics
(hyper-connections, LIE, LTI, K/V pre-computation from e) are carried over unchanged.

### Parameters — All Identical to CART Except:

| Component | CART | RTCC Paper 2 |
|---|---|---|
| Core FFN | SwiGLU, hidden=2,048 | ToroidalMoE (see below) |
| vocab_size | 32,000 | 32,000 ← *must fix in current code (currently 50,257)* |
| Core attention | MLA cross-attention | MLA cross-attention ← *must fix in current code (currently self-attention)* |
| LTI formula | `A·h + block_out` | `A·h + block_out` ← *must fix in current code (currently adds B·e term)* |
| Everything else | — | Identical |

### Sweep — d=576 (24×24 square grid) — 11 runs

The 576-dim hidden vector reshapes in-place to a **24×24 = 576** square grid.

d=576 is chosen for three reasons:
- 24×24 is a perfect square → symmetric torus, no axis bias, all experts topologically equivalent
- GCD(24, 24) = 24; after the private_core ≥ 3×3 floor, strides {4, 6, 8} survive → 7 valid sweep configurations across the 11%–61% privacy range
- head_dim=64 maintained: n_heads = 576/64 = 9 (valid; CART itself sweeps non-power-of-2 head counts)

**Fixed across all sweep runs:**

| Parameter | Value | Notes |
|---|---|---|
| `d_model` | 576 | |
| Grid | 24 × 24 | Square — symmetric torus |
| `n_heads` | 9 | 576 / 64 |
| `d_kv_latent` | 144 | d / 4 |
| `vocab_size` | 32,000 | Llama-2 BPE — matches CART |
| `expert_hidden` | TBD (512 recommended) | See Sweep Plan open questions |
| All other params | Identical to CART | |

**Sweep configurations (all on 24×24 grid):**

| Run | Stride | Overlap | Patch | Experts | Privacy | Role |
|---|---|---|---|---|---|---|
| S2 | 3 | 1-cell | 4×4 | 8×8 = 64 | 25.0% | Pair A |
| S1 | 4 | 2-cell | 6×6 | 6×6 = 36 | 11.1% | Low-privacy anchor |
| S4 | 4 | 1-cell | 5×5 | 6×6 = 36 | 36.0% | Pair B |
| S3 | 6 | 2-cell | 8×8 | 4×4 = 16 | 25.0% | Pair A |
| S6 | 6 | 1-cell | 7×7 | 4×4 = 16 | 51.0% | Mid-high anchor |
| S5 | 8 | 2-cell | 10×10 | 3×3 = 9 | 36.0% | Pair B |
| S7 | 8 | 1-cell | 9×9 | 3×3 = 9 | 60.5% | High-privacy anchor (primary) |

Expert range: 9–64, patch range: 4×4–10×10. Pairs A and B each hold privacy % constant while varying
overlap depth — embedded ablations at no extra training cost. Full specification in **RTCC_Sweep_Plan.md**.

### CART Comparison Run — d=768 — 2 runs

One pair of runs at CART's exact dimension so perplexity curves can be directly compared.
Uses the best-performing privacy setting from the sweep.

| Parameter | Value | Notes |
|---|---|---|
| `d_model` | 768 | Matches CART Stage 2 exactly |
| Grid | 32 × 24 | Non-square — axis bias noted in paper |
| Stride | 8 | Only valid stride at d=768 |
| Patch | 9×9, overlap=1 | 1-cell; mirrors S7 privacy level (60.5%) |
| Experts | 4×3 = 12 | Non-square layout |
| Privacy | 60.5% | Like-for-like with sweep winner |
| Sweep R, P | Same as CART Stage 2 | Direct curve comparison |

Runs: 1 RTCC-768 + 1 Dense-FFN-768 baseline. Total: 2 additional runs beyond the sweep.
These validate that sweep findings at d=576 generalize to the CART-comparable dimension and
allow a direct perplexity number comparison against CART's published Stage 2 results.

### Ablation Set (5 architectures, all identical except core FFN)

| Arch | Core FFN | Tests |
|---|---|---|
| 01 Dense FFN | SwiGLU (identical to CART) | Validates implementation — should reproduce CART perplexity |
| 02 Standard MoE | Top-k routed MoE | Does the torus beat routing-based MoE? |
| 03 SliceMoE flat | Non-overlapping slices, no torus | No overlap, no wrap |
| 04 Flat grid overlap | Overlapping patches, zero-pad boundary | Overlap yes, torus wrap no |
| 05 RTCC | Overlapping patches, circular-pad boundary | Full toroidal — the paper claim |

The 04 vs. 05 comparison directly isolates the torus wrap contribution: identical code, only
`F.pad(mode='constant', value=0)` vs. `_toroidal_pad` (circular).

### Changes Required in Current Code

The current RTCC_Paper_2 codebase has three deviations from CART that must be fixed before paper runs:

1. **`shared/config.py`**: `vocab_size` is 50,257 (GPT-2). **Locked to 32,000** (Llama-2 BPE, matches CART).
2. **Core attention**: Current `RTCCBlock` and all arch 01–05 core blocks use `MLASelfAttention`.
   Change to `MLACrossAttention` (Q from h, K/V pre-computed from e) matching CART exactly.
   This requires adding `MLAKVProjection` to each model and passing K, V through the loop,
   same pattern as CART's `cart.py`.
3. **LTI**: Current `LTIInjection` adds a `B·e` term: `A·h + B·e + block_out`.
   Remove the `B·e` term. CART's formula is `A·h + block_out`; prelude injection is handled
   by cross-attention, not a separate additive path.

### Training — Identical to CART Stage 2

Same seq_len, batch, tokens, optimizer, LR schedule. Sweep same R and P values so that
perplexity curves are directly comparable across (R, P) pairs.

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

| | Paper 1 (CART) | Paper 2 (RTCC minimal) | Paper 3 (RTCC V2 full) |
|---|---|---|---|
| Novel claim | Recurrent depth > feed-forward depth at equal params | Toroidal FFN > SwiGLU in RDT | Full RTCC with expanded sheet, GQA+SWA prelude, Muon, YaRN |
| Core FFN | SwiGLU | ToroidalMoE | ToroidalMoE (projected, 64 experts) |
| d_model | 256–1024 (swept) | 576 (sweep) + 768 (CART cmp) | 1,024 |
| Vocab | 32,000 | 32,000 | 32,000 |
| Loop count | Swept | Swept | 16 |
| Expert grid | — | n×n square, 9–64 experts (swept) | 8×8 = 64 (square symmetric) |
| Sheet space | — | In-place (24×24) | Projected (1024 → 5,184) |
| Prelude attn | MLA self + cross-attn core | MLA cross-attn core (matches CART) | GQA + SWA + sink tokens |
| Optimizer | AdamW | AdamW | Hybrid Muon + AdamW |
| RoPE | Standard | Standard | YaRN |
| Training tokens | ~1B | ~1B | ~15B |
| Hardware | RTX 3050 / 3090 | RTX 3050 / 3090 | RTX 3090 |
| Status | **Stage 2 in progress** | Needs 2 code fixes, then runs | Planned post Paper 2 |

*— Chad A. Capps, May 2026*
