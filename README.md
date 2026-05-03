# Recurrent Toroidal Cortical Columns (RTCC)

**Paper:** *Recurrent Toroidal Cortical Columns: A Neocortically-Inspired Architecture for Efficient Language Modeling*
**Author:** Chad Capps ([@ccapps42](https://github.com/ccapps42))
**Status:** Experiments in progress — paper forthcoming
**License:** MIT

---

## Overview

RTCC is a language model architecture that replaces the feed-forward layer in a recurrent-depth transformer with a **Toroidal Sheet Mixture of Experts (TS-MoE)** — a spatially organized expert system in which:

* The model's embedding dimensions are reshaped into a **2D toroidal grid**
* Fixed-position experts process **overlapping patches** of that grid
* Shared border dimensions between adjacent experts create **implicit inter-expert communication** through gradient flow — no explicit lateral connections needed
* Recurrent depth nests these toroidal sheets into **cortical columns**, one column per expert position, across all loop iterations

The result is an architecture with a precise structural analog to the neocortical minicolumn — the fundamental computational unit of the mammalian brain.

---

## Architecture

RTCC builds on the **CART** framework ([ccapps42/CART](https://github.com/ccapps42/CART)) and differs in exactly one way: the feed-forward layer in the recurrent core is replaced by the Toroidal Sheet MoE.

```
Input Tokens
    ↓
[Embedding]
    ↓
[Prelude]  — 4× MLA self-attention blocks (RoPE, causal), run once
    ↓        (sensory cortex analog)
    ↓
[Recurrent Core]  — looped R times (shared weights)
    ↓  ┌──────────────────────────────────────────────┐
    ↓  │  hyper.combine(buffer) → h_input             │
    ↓  │    (softmax-weighted blend of last 3 states)  │
    ↓  │  LIE: sinusoidal loop-index signal            │
    ↓  │    → projected to model_dim → added to h_input│
    ↓  │  MLA self-attention (12-head, causal)         │
    ↓  │  Toroidal Sheet MoE               ← novel     │
    ↓  │    • reshape 768-dim → 32×24 grid             │
    ↓  │    • circular (toroidal) padding              │
    ↓  │    • unfold → 12 position-fixed experts       │
    ↓  │    • per-expert SwiGLU                        │
    ↓  │    • overlap-add fold                         │
    ↓  │  LTI: h = sigmoid(A)·h_input + B·e + block_out│
    ↓  │  hyper.update_buffer(buffer, h)               │
    ↓  │    (push h to front of ring buffer,           │
    ↓  │     drop oldest state)                        │
    ↓  └──────────────────────────────────────────────┘
    ↓
[Coda]  — 1× MLA self-attention block (RoPE, causal), run once
    ↓     (motor cortex analog)
    ↓
[RMSNorm → LM Head]  (weight-tied to embedding)
    ↓
Output Logits
```

### Toroidal Sheet MoE

The key innovation. At each recurrent loop iteration:

1. The 768-dim token embedding is reshaped into a **32×24 toroidal grid**
2. **Circular (toroidal) padding** wraps both axes — eliminating edge effects, making every expert position structurally identical
3. **N×N patches** slide over the grid with configurable stride and overlap
4. Each expert is **position-fixed** — no routing, no load-balancing loss, each expert always processes its assigned patch
5. Shared border dimensions between adjacent patches accumulate gradients from both experts during backprop — **implicit communication through architecture, not explicit lateral connections**
6. The output is folded back via **overlap-add** to the full 32×24 grid

### Cortical Column Emergence

Each expert position on the toroidal grid, stacked across all R recurrent loop iterations, forms a **cortical column**:

| Neocortex | RTCC |
| --- | --- |
| Cortical sheet (2D) | Toroidal grid (32×24) |
| Minicolumn (depth) | Expert position across R loops |
| ~6 cortical layers | R loop iterations |
| Private intracolumn wiring | Private core dims (never shared) |
| Lateral intercolumn connections | Shared border dims between neighbors |
| Local neighborhood only | Toroidal adjacency — neighbors only |
| Columnar feedback | Hyper-connections at loop boundary |

### Model Configuration (Paper Runs)

```
model_dim:       768        # 768 = 32×24 toroidal grid
grid:            32×24
n_heads:         12         # head_dim = 64 (power of 2)
mla_latent_dim:  192        # KV compression (~dim/4)
prelude_layers:  4
recurrent_layers: 6         # looped R times
coda_layers:     1
max_loop_iters:  8
vocab_size:      50,257     # GPT-2 tokenizer

# Primary RTCC config (Tier 1 comparison)
patch_size:      10×10      # 100 dims per expert
stride:          8
overlap:         2 cells
n_experts:       4×3 = 12  # ≥3 per axis on toroidal grid
private_dims:    36         # 6×6 core, never shared
privacy_pct:     36%
```

### Privacy Ratio

The ratio of private to total expert dimensions is a key architectural hyperparameter. The paper ablates five configurations:

| Config | Patch | Stride | Overlap | Experts | Private dims | Privacy% |
| --- | --- | --- | --- | --- | --- | --- |
| sweep\_11 | 3×3 | 2 | 1-cell | 192 | 1 | 11% |
| sweep\_36 | 5×5 | 4 | 1-cell | 48 | 9 | 36% |
| sweep\_44 | 6×6 | 4 | 2-cell | 48 | 4 | 11%\* |
| sweep\_60 | 9×9 | 8 | 1-cell | 12 | 49 | 60% |
| **sweep\_64** | **10×10** | **8** | **2-cell** | **12** | **36** | **36%**\* |

\* Controlled pairs at the same expert count isolate the effect of overlap width independently of expert count.

### Grid Dimension Selection

The toroidal grid places a hard constraint on which model dimensions are usable: a stride `S` is only valid if it divides **both** grid axes cleanly. This means the choice of `d` determines the entire patch/stride design space, and most values of `d` have very few valid configurations.

**Three additional constraints apply:**

1. **Head dimension.** For efficient attention, `d / n_heads` should be a power of 2. This strongly favors `d` values with large powers-of-2 factors (e.g. 768 = 2⁸ × 3, giving head\_dim = 64 with 12 heads).

2. **Stride threshold.** Strides below 6 produce private cores of 9 dimensions or fewer (a 3×3 region), which are too small for meaningful expert specialization. Only configurations with stride ≥ 6 are architecturally interesting.

3. **Minimum narrow dimension.** Every axis of the expert grid must have at least **3 experts**. This is a hard topological requirement of the toroidal communication mechanism, not a soft preference.

   On a torus, each expert's shared border dims accumulate gradient from its neighbors on all four sides. For this lateral communication to be meaningful, the neighbors must be **distinct**: every expert must have a unique neighbor to its left, a unique neighbor to its right, a unique neighbor above, and a unique neighbor below — all four different from each other and different from the expert itself.

   With 2 experts on an axis, the toroidal wrap makes each expert its own second-order neighbor: expert A's left neighbor is B, and expert A's right neighbor is also B. The shared border dims then receive gradient from the same source in both directions, providing no lateral diversity — the communication degenerates to a self-loop through one intermediary. With 3 experts, expert A has B on one side and C on the other — four unique neighbors across both axes.

   This constraint eliminates any configuration where either grid axis has fewer than 3 expert positions, i.e. where `min(rows/S, cols/S) < 3`.

   Note that at 768-dim, stride 8 gives a 4×3 layout — the short axis sits exactly at the minimum. The constraint is satisfied but not comfortably. Larger dimensions provide more headroom above the floor on both axes simultaneously.

#### Why 768

768 = 32 × 24. GCD(32, 24) = 8, giving valid strides {2, 4, 8}. After applying the stride ≥ 6 threshold, only **stride 8** remains — which is exactly the paper's primary configuration. The dimension is chosen precisely because it concentrates all the useful design space into one well-characterized stride, and because 768 / 12 = 64 satisfies the power-of-2 head\_dim constraint cleanly.

| Stride | Overlap | Patch | Experts | Total dims | Private | Shared | Privacy% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 1-cell | 9×9 | 4×3 = 12 | 81 | 49 | 32 | 60% |
| **8** | **2-cell** | **10×10** | **4×3 = 12** | **100** | **36** | **64** | **36%** ← primary |

The 10×10 patch at stride 8 also produces a notable biological correspondence: **100 dimensions per expert** matching the ~100-neuron count of a biological cortical minicolumn. This correspondence was not designed in — it is a consequence of the patch geometry.

#### Candidates Above 768

Larger experiments require a dimension with stride 9 available (to preserve the 10×10 / 100-dim biological match) and at least three useful stride ≥ 6 configurations for a meaningful privacy sweep. A systematic search over all `d` from 769 to 2200 where both grid axes are divisible by 9 identifies two candidates:

---

**d = 1296 — 36×36 (square grid) — recommended for follow-up**

GCD(36, 36) = 36. Valid strides ≥ 6: **6, 9, 12**. Square grid — every expert position is structurally identical, full toroidal symmetry. Head dim: 1296 / 12 = 108 (not a power of 2; a separate attention projection is required).

| Stride | Overlap | Patch | Experts | Total dims | Private | Shared | Privacy% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 1-cell | 7×7 | 6×6 = 36 | 49 | 25 | 24 | 51% |
| 6 | 2-cell | 8×8 | 6×6 = 36 | 64 | 16 | 48 | 25% |
| **9** | **1-cell** | **10×10** | **4×4 = 16** | **100** | **64** | **36** | **64%** ★ |
| 9 | 2-cell | 11×11 | 4×4 = 16 | 121 | 49 | 72 | 40% |
| 12 | 1-cell | 13×13 | 3×3 = 9 | 169 | 121 | 48 | 72% |
| 12 | 2-cell | 14×14 | 3×3 = 9 | 196 | 100 | 96 | 51% |

★ Stride 9 with 1-cell overlap reproduces the biological match exactly: 10×10 patch, 100 dims per expert, 64 private — and the 4×4 = 16 expert layout on a square grid preserves full toroidal symmetry.

1296 was chosen over 1728 (36×48), the next multiple-of-81 value, because 1728 does not support stride 9 — 9 divides 36 but not 48. This eliminates the biological match configuration entirely at 1728.

---

**d = 1944 — 36×54 (rectangular 2:3 grid) — superseded by 2592**

GCD(36, 54) = 18. Valid strides ≥ 6: **6, 9, 18**. Head dim: 1944 / 12 = 162 (not a power of 2).

The minimum narrow-dimension constraint (≥ 3 experts per axis on the toroidal sheet, so no expert wraps around to become its own neighbor) eliminates stride 18 here: the 2×3 layout has a narrow dimension of 2. This reduces 1944 to only 2 valid strides, which is insufficient for a meaningful privacy sweep. 1944 is superseded by 2592, which adds stride 12 and achieves 3 valid strides at smaller d.

| Stride | Overlap | Patch | Layout | Narrow | Experts | Total dims | Private | Privacy% | Valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 1-cell | 7×7 | 6×9 | 6 | 54 | 49 | 25 | 51% | ✓ |
| 6 | 2-cell | 8×8 | 6×9 | 6 | 54 | 64 | 16 | 25% | ✓ |
| **9** | **1-cell** | **10×10** | **4×6** | **4** | **24** | **100** | **64** | **64%** ★ | **✓** |
| 9 | 2-cell | 11×11 | 4×6 | 4 | 24 | 121 | 49 | 40% | ✓ |
| ~~18~~ | ~~1-cell~~ | ~~19×19~~ | ~~2×3~~ | ~~2~~ | ~~6~~ | ~~361~~ | ~~289~~ | ~~80%~~ | ✗ |
| ~~18~~ | ~~2-cell~~ | ~~20×20~~ | ~~2×3~~ | ~~2~~ | ~~6~~ | ~~400~~ | ~~256~~ | ~~64%~~ | ✗ |

★ Biological match preserved at stride 9.

---

**d = 2592 — 36×72 (rectangular 2:1 grid) — richest valid stride set**

GCD(36, 72) = 36. Valid strides ≥ 6 after narrow constraint: **6, 9, 12**. Stride 18 eliminated (2×4, narrow=2). Head dim: 2592 / 12 = 216 (not a power of 2). The unique property of 2592 is stride 12, which no other candidate supports — giving a privacy range from 51% up to 72% at the primary (1-cell) config that no other candidate reaches with 3+ experts on both axes.

| Stride | Overlap | Patch | Layout | Narrow | Experts | Total dims | Private | Privacy% | Valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 1-cell | 7×7 | 6×12 | 6 | 72 | 49 | 25 | 51% | ✓ |
| 6 | 2-cell | 8×8 | 6×12 | 6 | 72 | 64 | 16 | 25% | ✓ |
| **9** | **1-cell** | **10×10** | **4×8** | **4** | **32** | **100** | **64** | **64%** ★ | **✓** |
| 9 | 2-cell | 11×11 | 4×8 | 4 | 32 | 121 | 49 | 40% | ✓ |
| 12 | 1-cell | 13×13 | 3×6 | 3 | 18 | 169 | 121 | 72% | ✓ |
| 12 | 2-cell | 14×14 | 3×6 | 3 | 18 | 196 | 100 | 51% | ✓ |
| ~~18~~ | ~~1-cell~~ | ~~19×19~~ | ~~2×4~~ | ~~2~~ | ~~8~~ | ~~361~~ | ~~289~~ | ~~80%~~ | ✗ |
| ~~18~~ | ~~2-cell~~ | ~~20×20~~ | ~~2×4~~ | ~~2~~ | ~~8~~ | ~~400~~ | ~~256~~ | ~~64%~~ | ✗ |

★ Biological match preserved at stride 9.

---



GCD(54, 54) = 54. Valid strides ≥ 6: **6, 9, 18**. Square grid — every expert position is structurally identical. Head dim: 2916 / 12 = 243 (not a power of 2; a separate attention projection is required, as with 1296).

| Stride | Overlap | Patch | Expert layout | Experts | Total dims | Private | Shared | Privacy% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 1-cell | 7×7 | 9×9 sq | 81 | 49 | 25 | 24 | 51% |
| 6 | 2-cell | 8×8 | 9×9 sq | 81 | 64 | 16 | 48 | 25% |
| **9** | **1-cell** | **10×10** | **6×6 sq** | **36** | **100** | **64** | **36** | **64%** ★ |
| 9 | 2-cell | 11×11 | 6×6 sq | 36 | 121 | 49 | 72 | 40% |
| 18 | 1-cell | 19×19 | 3×3 sq | 9 | 361 | 289 | 72 | 80% |
| 18 | 2-cell | 20×20 | 3×3 sq | 9 | 400 | 256 | 144 | 64% |

★ Stride 9 with 1-cell overlap reproduces the biological match exactly: 10×10 patch, 100 dims per expert, 64 private. The expert layout is 6×6 = 36 experts in a square grid — every expert position on the toroidal sheet is structurally identical at every useful stride simultaneously.

2916 is architecturally notable for a property no smaller candidate possesses: **every useful stride produces a square expert layout on a square toroidal sheet**. At stride 6: 9×9 = 81 experts. At stride 9: 6×6 = 36 experts. At stride 18: 3×3 = 9 experts. Full structural equivalence is maintained across the entire privacy sweep, not just at the primary configuration.

---

**d = 3888 — 54×72 (rectangular 4:3 grid) — largest valid expert counts, all strides pass**

GCD(54, 72) = 18. Valid strides ≥ 6 after narrow constraint: **6, 9, 18** — all pass with no eliminations. Head dim: 3888 / 12 = 324 (not a power of 2). 3888 is the only candidate where every stride ≥ 6 meets the narrow-dimension constraint without exception. It also produces the highest expert counts of any candidate at each stride level.

| Stride | Overlap | Patch | Layout | Narrow | Experts | Total dims | Private | Privacy% | Valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 1-cell | 7×7 | 9×12 | 9 | 108 | 49 | 25 | 51% | ✓ |
| 6 | 2-cell | 8×8 | 9×12 | 9 | 108 | 64 | 16 | 25% | ✓ |
| **9** | **1-cell** | **10×10** | **6×8** | **6** | **48** | **100** | **64** | **64%** ★ | **✓** |
| 9 | 2-cell | 11×11 | 6×8 | 6 | 48 | 121 | 49 | 40% | ✓ |
| 18 | 1-cell | 19×19 | 3×4 | 3 | 12 | 361 | 289 | 80% | ✓ |
| 18 | 2-cell | 20×20 | 3×4 | 3 | 12 | 400 | 256 | 64% | ✓ |

★ Biological match preserved at stride 9.

---

The square-grid candidates follow a clean geometric scaling law: grid axes grow as multiples of 9 (36 → 45 → 54), and expert counts at stride 9 grow as perfect squares (4² → 5² → 6²). Each step adds one expert per axis at the biological-match stride. 2025 is omitted from practical consideration because it supports only two useful strides (9 and 15) after applying the narrow constraint, making a meaningful privacy sweep impossible.

#### Dimension Comparison Summary

All candidates after applying the minimum narrow-dimension constraint (≥ 3 experts per axis). Invalid configs excluded. For each stride the entry shows: **experts (layout) @ 1-cell% / 2-cell%**. Biological-match stride (9 or 8 for 768) marked ★.

| d | Grid | Head /12 | Stride 6 | Stride 8/9 ★ | Stride 12 | Stride 18 | Valid configs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **768** | 32×24 | **64 ✓** | — | 12 (4×3) @ 60%/36% | — | — | 2 |
| **1296** | 36×36 | 108 | 36 (6×6) @ 51%/25% | 16 (4×4) @ 64%/40% | 9 (3×3) @ 72%/51% | — | 6 |
| ~~1728~~ | ~~36×48~~ | ~~144~~ | ~~no stride 9~~ | ~~—~~ | — | — | — |
| 1944 | 36×54 | 162 | 54 (6×9) @ 51%/25% | 24 (4×6) @ 64%/40% | — | ~~narrow=2~~ | 4 |
| **2592** | **36×72** | **216** | **72 (6×12) @ 51%/25%** | **32 (4×8) @ 64%/40%** | **18 (3×6) @ 72%/51%** | ~~narrow=2~~ | **6** |
| **2916** | **54×54** | **243** | **81 (9×9) @ 51%/25%** | **36 (6×6) @ 64%/40%** | — | **9 (3×3) @ 80%/64%** | **6** |
| **3888** | **54×72** | **324** | **108 (9×12) @ 51%/25%** | **48 (6×8) @ 64%/40%** | — | **12 (3×4) @ 80%/64%** | **6** |

**Notes:**
- 1728 eliminated — no stride 9 available.
- 1944 reduced to 4 valid configs after stride 18 eliminated (narrow=2). Superseded by 2592.
- 2592 is the only candidate with stride 12, adding an intermediate 18-expert config at 72% privacy unavailable elsewhere.
- 2592 and 2916 both reach 6 valid configs but through different strides — 2592 via stride 12, 2916 via stride 18.
- 3888 is the only candidate where every stride ≥ 6 passes the narrow constraint with no eliminations. Also has the highest expert counts at every stride level.

---

## Relationship to CART (Paper 1)

RTCC is Paper 2 in a two-paper sequence. **CART** ([ccapps42/CART](https://github.com/ccapps42/CART), Capps 2026) establishes the recurrent framework through original ablation experiments.

The general direction of increasing effective depth through recurrence is supported by recent independent work. Oncescu et al. (arXiv 2604.21215, 2026) introduce a structurally distinct recurrent transformer in which each layer attends to KV pairs computed from its own outputs rather than the previous layer — a per-layer self-referential mechanism that differs from CART's loop-based weight sharing. Their empirical results show cross-entropy improvement over parameter-matched baselines at 150M–300M parameters on C4, independently corroborating that recurrent depth is a productive direction at scales comparable to CART's experiments. Their theoretical proofs, however, concern their specific architecture and do not directly cover CART's mechanism.

CART's recurrent core is architecturally distinct: the same block of layers is executed R times with fully shared weights, conditioned per loop by a sinusoidal loop-index embedding (LIE), stabilized by LTI injection, and blended across iterations by hyper-connections. This design, and the specific empirical findings below, are original to CART.

**Borrowed components assembled in CART:**

* LTI injection for loop stability (Parcae, Prairie et al. 2026)
* Loop index embedding LIE (OpenMythos, kyegomez)
* Hyper-connections at loop boundaries (Hyperloop Transformer, arXiv 2604.21254)
* MLA attention (DeepSeek-V2, arXiv 2405.04434)
* RoPE, RMSNorm, SwiGLU, weight tying (standard components)

**Original findings from CART ablations:**

* **4+6+1 prelude/recurrent/coda balance** — empirically optimal at both dim=256 and dim=768; all prior RDT work uses symmetric configs
* **Asymmetric MLA head counts** — 16-head prelude, 12-head recurrent core, 8-head coda
* **R=8 loop ceiling** — validated as the optimal training ceiling
* **Phased sequence-length and loop-count curriculum** — critical for stable training, validated across four phases

RTCC inherits all of the above from CART without modification and contributes exactly one change: the feed-forward layer in the recurrent core is replaced by the Toroidal Sheet MoE. This makes the paper's claim precise and fully controlled.

---

## Novelty

RTCC makes two distinct original contributions:

### Contribution 1: The CART Framework (established in Paper 1)

The recurrent framework underlying RTCC was designed and validated through original ablation experiments in CART ([ccapps42/CART](https://github.com/ccapps42/CART), Capps 2026). CART's loop-based weight-sharing architecture — in which the same block executes R times with shared weights, conditioned per loop by LIE, stabilized by LTI injection, and blended across iterations by hyper-connections — is architecturally distinct from prior recurrent transformer work including Oncescu et al. (arXiv 2604.21215, 2026), whose per-layer KV self-reference mechanism is a different approach to the same general goal of increasing effective depth. CART's original contributions are:

* **4+6+1 layer balance** — asymmetric prelude/recurrent/coda configuration, empirically optimal across dim=256 and dim=768; all prior published RDT work uses symmetric configs
* **Asymmetric MLA head counts** — 16-head prelude, 12-head recurrent core, 8-head coda; heavier prelude conditions the signal, lighter coda is sufficient for output projection
* **R=8 loop ceiling** — validated as optimal training ceiling; lower ceilings constrain low-loop performance, higher ceilings degrade it
* **Phased curriculum** — sequence length and loop count ramped jointly across four phases; independently validated as critical for stable training
* **Component assembly** — LTI injection (Parcae), loop index embedding (OpenMythos), hyper-connections (Hyperloop Transformer), MLA (DeepSeek-V2), RoPE, RMSNorm, SwiGLU assembled into a coherent loop-based architecture and validated at scale

These findings are the contribution of CART. RTCC inherits this framework exactly, with no modifications, making the paper's claim precise: *the ToroidalMoE is the only variable.*

### Contribution 2: The Toroidal Sheet MoE (this work)

To our knowledge, RTCC is the first architecture to:

1. Impose a **topological structure on expert dimensional ownership** — experts own specific subspaces of the embedding, not the full embedding
2. Create **implicit inter-expert communication through overlapping dimensional ownership** — shared border dims receive gradient from multiple experts simultaneously, forcing agreement without explicit lateral connections
3. Use a **toroidal boundary condition** on the embedding grid — eliminating edge effects and making every expert position structurally identical
4. Achieve **cortical column structure as an emergent property** of combining 2D toroidal expert layout with recurrent depth — no biological structure is explicitly programmed

### Closest Prior Work

* **SliceMoE** (Vejendla, 2024) — dimensional partitioning without overlap, without topology, without position-fixed experts
* **MoGE** (Kang et al., 2025) — 2D structure applied to routing inputs, not dimensional ownership

---

## Experiments

All paper experiments use:

* **Hardware:** RTX 3090 24GB
* **Model dim:** 768
* **Tokenizer:** GPT-2 (`gpt2` via HuggingFace `transformers`), vocab size 50,257
* **Training data:** TinyStories, Wikipedia, FineWeb-Edu, FineWeb (via HuggingFace datasets)
* **Evaluation:** Held-out validation set + lm-eval-harness benchmarks
* **All results logged to SQLite** — exported to `results/` for reproducibility

### Run Plan

| Tier | Runs | Purpose |
| --- | --- | --- |
| Tier 1 | 5 | Architecture comparison: Dense FFN, Standard MoE, SliceMoE-flat, Flat Grid Overlap, RTCC |
| Tier 2 | 4+1 | Privacy/overlap sweep (5 configs) |
| Tier 3 | 2 | Inference-only communication ablations (zero shared dims, expert shuffle) |
| Tier 4 | 3 | Column depth sweep (R=2, 4, 6, 8) |
| Tier 5 | 0 | Specialization analysis on saved checkpoints |
| Tier 6 | 1 | 1B token benchmark run, full lm-eval suite |

---

## Repository Structure

```
rtcc/
  config.py              — RTCCConfig with computed privacy% and guard assertions
  model.py               — Top-level RTCC model
  toroidal_moe.py        — Toroidal Sheet MoE (the core contribution)
  column_block.py        — One recurrent block with ToroidalMoE

architectures/
  01_dense_ffn/          — Standard transformer baseline
  02_standard_moe/       — Coarse-grained MoE baseline
  03_slicemoe_flat/      — SliceMoE-style dimensional partition, no overlap
  04_flat_grid_overlap/  — Overlapping grid without toroidal boundary
  05_rtcc/               — Full RTCC architecture

shared/
  components/            — MLA attention, RMSNorm, RoPE, LTI injection, LIE, hyper-connections
  training/              — Trainer, optimizer, checkpoint, DB logger
  data/                  — Curriculum scheduler, dataset loader, document packing
  eval/                  — Perplexity, lm-eval wrapper, specialization probing

configs/
  paper_768/             — All paper run configs (YAML)
  dev_512/               — Development/debugging configs

scripts/
  prepare_validation.py  — ONE-TIME: carve held-out validation set
  launch_run.py          — Queue a training run
  resume_run.py          — Resume from latest checkpoint
  run_inference_ablation.py  — Zero/shuffle tests on saved checkpoints
  generate_paper_tables.py   — Pull from DB, output LaTeX tables

analysis/
  plot_loss_curves.py
  plot_privacy_sweep.py
  plot_depth_sweep.py
  expert_activation_entropy.py
  representational_similarity.py

db/
  rtcc_experiments.db    — SQLite database, all training and eval results

results/                 — CSV/JSON exports for reproducibility
paper/
  figures/               — Generated paper figures
  tables/                — Generated LaTeX tables
```

---

## Installation

```
git clone https://github.com/ccapps42/RTCC
cd RTCC
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, CUDA 11.8+

---

## Reproducing Paper Results

> ⚠️ **Note:** Paper experiments are still in progress. This section will be completed at submission time with exact commands, config paths, and checkpoint links.

```
# Prepare validation set (run once before any training)
python scripts/prepare_validation.py

# Launch a training run
python scripts/launch_run.py --config configs/paper_768/rtcc_full.yaml

# Run inference ablations on a saved checkpoint
python scripts/run_inference_ablation.py --checkpoint runs/paper_768/05_rtcc/checkpoints/best

# Generate paper tables from DB
python scripts/generate_paper_tables.py --output paper/tables/
```

Model weights will be available on HuggingFace Hub at submission time.

---

## Citation

If you use this work, please cite:

```bibtex
@article{capps2026rtcc,
  title   = {Recurrent Toroidal Cortical Columns: A Neocortically-Inspired Architecture for Efficient Language Modeling},
  author  = {Capps, Chad},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/ccapps42/RTCC}
}
```

*arXiv ID and venue will be added at submission.*

Also cite CART (Paper 1):

```bibtex
@article{capps2026cart,
  title   = {CART: Context-Anchored Recurrent Transformer},
  author  = {Capps, Chad},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/ccapps42/CART}
}
```

---

## Key References

* Geiping et al., *Huginn* (2025) — recurrent depth at scale
* Hyperloop Transformers, arXiv 2604.21254 (2026) — hyper-connections
* Oncescu et al., arXiv 2604.21215 (2026) — a structurally distinct recurrent transformer using per-layer KV self-reference (different mechanism from CART's loop-based weight sharing); provides independent empirical evidence that recurrent depth improves cross-entropy over parameter-matched baselines at 150M–300M scale on C4
* Gomez, *OpenMythos* (2026) — loop index embedding (LIE)
* Prairie et al., *Parcae* (2026) — LTI injection, spectral radius stability
* Dai et al., *DeepSeek-V2*, arXiv 2405.04434 (2024) — MLA attention
* Vejendla, *SliceMoE* (2024) — closest prior work on dimensional partitioning
* Hawkins et al., *Thousand Brains Theory* (2019) — columnar cortical computation
* Gardner et al., *Science* (2022) — toroidal topology of grid cells in entorhinal cortex
* Dehghani et al., *Universal Transformer* (2018) — recurrent depth foundations

---

## License

MIT License — see [LICENSE](https://github.com/ccapps42/RTCC/blob/master/LICENSE)
