# RTCC Architecture Briefing
## Context for New Conversation

**Author:** Chad Capps ([@ccapps42](https://github.com/ccapps42))  
**GitHub:** https://github.com/ccapps42/RTCC  
**Related:** CART (Paper 1) at https://github.com/ccapps42/CART  
**Date:** 2026-05-03

---

## What RTCC Is

Recurrent Toroidal Cortical Columns (RTCC) is a language model architecture that replaces the feed-forward layer in a recurrent-depth transformer with a **Toroidal Sheet Mixture of Experts (TS-MoE)**:

- Model embedding dimensions reshaped into a **2D toroidal grid**
- Fixed-position experts process **overlapping patches** of that grid (no routing)
- Shared border dimensions between adjacent experts create **implicit inter-expert communication** via overlap-add gradient accumulation — no explicit lateral connections
- Recurrent depth stacks toroidal sheets into **cortical columns** — each expert position across all R loop iterations forms one column
- Private core dims are never shared with any other column at any loop depth

RTCC builds on **CART** (Context-Anchored Recurrent Transformer, Paper 1) and differs in exactly one way: the FFN in the recurrent core is replaced by ToroidalMoE.

---

## CART Validated Findings (from ongoing runs at d=256, 512, 768, 1024)

- **P=6 prelude layers** is optimal across all tested dims
- **1 coda layer** is sufficient
- **R=8 loop ceiling** is optimal for training
- **LTI injection** (from OpenMythos): `h = sigmoid(a)*h + transformer_out`
- **Phased curriculum**: seq_len 128→256→512, loop count 2-4→2-6→2-8
- All of the above are inherited by RTCC unchanged

---

## Key Formula

```
overlap      = patch_size - stride          (number of shared border cells each side)
private_side = patch_size - 2 × overlap
private_dims = private_side²
privacy%     = private_dims / patch_size²

# Guards (enforced in RTCCConfig):
assert private_side >= 1
assert grid_rows % stride == 0
assert grid_cols % stride == 0
assert grid_rows // stride >= 3   # ≥3 experts per axis (toroidal neighborhood)
assert grid_cols // stride >= 3
```

**All paper runs use 1-cell overlap only** (overlap=1, stride=patch-1). This gives the cleanest gradient signal — shared border dims receive gradient from exactly 2 experts on edges, 4 on corners. This is true for both 1-cell and 2-cell overlap (corners always get 4), but 1-cell keeps the private core larger and the communication bandwidth narrower.

---

## Architecture Stack

```
Input Tokens
    ↓
[Embedding]  vocab_size × dim
    ↓
[Prelude]  6× MLA blocks + Dense FFN  (runs ONCE — validated P=6)
    ↓
[Recurrent Core]  ×R loops
    ┌─────────────────────────────────────┐
    │  MLA Attention                      │
    │  ToroidalMoE (replaces FFN)         │
    │  LTI injection                      │
    │  Hyper-connections                  │
    └──────────────── loop ───────────────┘
    ↓
[Coda]  1× MLA block + Dense FFN  (runs ONCE)
    ↓
[RMSNorm → LM Head (weight-tied to embedding)]
```

**Attention:** MLA (Multi-head Latent Attention, DeepSeek-V2). KV cache stores compressed latent (dim/4), not full K/V. Decoupled RoPE.

---

## The Two Model Scales

### Scale 1: 576-dim — SWEEP MODEL (paper ablations)

```
model_dim:    576
grid:         24×24  (576 = 24×24)  ★ SQUARE
n_heads:      9   (576/9 = 64)  head_dim=64 ✓
mla_latent:   144  (dim/4)
vocab_size:   50,000
prelude:      P=6
recurrent:    1 shared block, ×R loops
coda:         1 block
max_loops:    R=8 (training ceiling)
```

**Valid strides for 24×24:** 2, 3, 4, 6, 8, 12  
**All use 1-cell overlap (stride = patch - 1)**

**Five-Point Privacy Sweep:**

| Patch | Stride | Experts | MinAxis | PatchDims | PrivDims | Privacy% |
|-------|--------|---------|---------|-----------|----------|---------|
| 3×3   | 2      | 12×12=144 | 12    | 9         | 1        | 11.1%   |
| 4×4   | 3      | 8×8=64    | 8     | 16        | 4        | 25.0%   |
| 5×5   | 4      | 6×6=36    | 6     | 25        | 9        | 36.0%   |
| 7×7   | 6      | 4×4=16    | 4     | 49        | 25       | 51.0%   |
| **9×9** | **8** | **3×3=9** | **3** | **81**    | **49**   | **60.5%** |

- Max private dims: **49** (7×7 core)
- Min axis experts: **3** at the top config (exactly the minimum — stated in paper)
- Stride 8 on 24×24: 24÷8=3 exactly ✓

**Parameter Breakdown (576-dim, primary config 9×9/stride-8, expert_hidden=256):**

| Component | Params |
|---|---|
| Embedding (50K×576) | 28.800M |
| Prelude MLA ×6 | 4.479M |
| Prelude Dense FFN ×6 | 23.888M |
| Recurrent MLA ×1 (shared) | 0.746M |
| Recurrent MoE ×1 (shared) | 0.560M |
| LTI matrices A+B | 0.664M |
| Hyper-connections (n=3) | 0.002M |
| Coda MLA ×1 | 0.746M |
| Coda Dense FFN ×1 | 3.981M |
| RMSNorms | 0.007M |
| **TOTAL PARAMS** | **~63.9M** |
| **EFFECTIVE PARAMS (×R=6)** | **~72.3M** |

Dense FFN reference: ~67.9M total / ~90.8M effective  
RTCC effective as % of dense: **~79.6%**

**VRAM (576-dim training, seq=512, batch=4, grad_accum=8, 8-bit Adam):** ~1.0-1.5GB  
**Training speed on RTX 3090:** ~8,000-12,000 tokens/sec estimated

---

### Scale 2: 1600-dim — BENCHMARK MODEL (single run, no sweep)

```
model_dim:    1600
grid:         40×40  (1600 = 40×40)  ★ SQUARE
n_heads:      25  (1600/25 = 64)  head_dim=64 ✓
mla_latent:   400  (dim/4)
vocab_size:   50,000
prelude:      P=6
recurrent:    1 shared block, ×R loops
coda:         1 block
max_loops:    R=8
```

**Primary config (single benchmark config, not swept):**

| Patch | Stride | Experts | MinAxis | PatchDims | PrivDims | Privacy% |
|-------|--------|---------|---------|-----------|----------|---------|
| **11×11** | **10** | **4×4=16** | **4** | **121** | **81** | **66.9%** |

- patch=11, stride=10, overlap=1-cell
- private_side = 11-2 = 9, private_dims = 81
- 24÷10 is NOT integer — wait: 40÷10=4 ✓ (grid is 40×40 not 24×24)
- expert_hidden: TBD (see hidden dim discussion below)

**Parameter Breakdown (1600-dim, P=6 prelude, expert_hidden=256):**

| Component | Params |
|---|---|
| Embedding (50K×1600) | 80.000M |
| Prelude MLA ×6 | 34.560M |
| Prelude Dense FFN ×6 | 184.320M |
| Recurrent MLA ×1 (shared) | 5.760M |
| Recurrent MoE ×1 (shared) | 1.487M |
| LTI matrices A+B | 5.120M |
| Hyper-connections (n=3) | 0.005M |
| Coda MLA ×1 | 5.760M |
| Coda Dense FFN ×1 | 30.720M |
| RMSNorms | 0.019M |
| **TOTAL PARAMS** | **~347.8M** |
| **EFFECTIVE PARAMS (×R=6)** | **~409.7M** |

Dense FFN reference: ~378.8M total / ~584.4M effective  
RTCC effective as % of dense: **~70.1%**

**VRAM (1600-dim training, seq=512, 8-bit Adam):** ~3-4GB estimated (comfortable on 3090)

**Note on total/effective ratio:** The ratio is ~1.18× because the P=6 prelude with dense FFN dominates (184M+ params running once). The recurrent MoE (1.5M) is tiny relative to the prelude. The efficiency story is better told as: recurrent MoE uses 1.5M params vs 30.7M for dense FFN in the same slot — a 20× reduction in the looped component.

---

## Hidden Dim Decision (OPEN — needs resolution before coding)

Expert MLP structure (SwiGLU, 3 weight matrices):
```
patch_dims → expert_hidden → patch_dims
```

At 576-dim primary config (9×9 patch): patch_dims=81, hidden=256 → ratio=3.2× (close to standard 4×)  
At 1600-dim primary config (11×11 patch): patch_dims=121, hidden=256 → ratio=2.1× (below standard)

**Options:**
- **256:** Current assumption. Below standard for 1600-dim.
- **512:** 4× of 128, 6.3× patch_dims at 576. Above standard but reasonable.
- **1024:** 8× of 128. Generous. Grows recurrent core meaningfully.
- **Ablation:** Run 576-dim primary config at hidden=256, 512, 1024 (3 short runs, 100M tokens each) before committing. Adds a paper contribution. Recommended.

**If ablating:** Include as Tier 0 before any other runs. Result propagates to all sweep runs and the 1600-dim benchmark.

---

## Comparison Architectures (Five Baselines)

All baselines use the same CART recurrent framework. Only the FFN/MoE component changes.

### 1. Dense FFN Baseline
Standard SwiGLU FFN in the recurrent core. No MoE, no grid.
```
dim → 4×dim → dim  (SwiGLU, 3 matrices)
```
At 576-dim: 23.9M params in recurrent FFN per loop body (vs 0.56M for RTCC MoE)  
Purpose: absolute floor. All architectures compared against this.

### 2. Standard Sparse MoE
Full-embedding MoE with top-k routing and load-balance aux loss.
```
16 routed experts + 2 shared, top-2 routing
Each expert: dim → expert_hidden → dim
```
Experts see full 576-dim or 1600-dim embedding. Routing via softmax top-k.  
Purpose: MoE baseline. RTCC must beat this for the paper to have a claim.

### 3. SliceMoE-Flat (closest prior art)
Non-overlapping dimensional partition, position-fixed, no torus.  
At 576-dim: 576 / 81 = 7.1 — not clean. Use 576/9=64 experts × 9 dims each (no overlap, 1D slicing).  
At 1600-dim: 1600/121 = 13.2 — not clean. Use 1600/100=16 experts × 100 dims each.  
Purpose: isolates the overlap contribution specifically.  
Prior art: SliceMoE (Vejendla, arXiv 2510.04286, 2024).

### 4. Flat Grid Overlap
Same 2D grid and same patch/stride as RTCC primary config. Zero-padding at boundaries instead of circular toroidal wrap.  
Purpose: isolates the toroidal boundary contribution specifically.  
Implementation: `F.unfold` with `padding_mode='zeros'` instead of circular pad.

### 5. Full RTCC
The complete architecture. Toroidal grid + overlapping patches + position-fixed experts + overlap-add fold.

---

## Sweep Plan

### Tier 0: Hidden Dim Ablation (3 runs — do first if ablating)

| Run | Dim | Config | Hidden | Tokens | Purpose |
|-----|-----|--------|--------|--------|---------|
| H1 | 576 | 9×9/s8 | 256 | 100M | Baseline |
| H2 | 576 | 9×9/s8 | 512 | 100M | 2× hidden |
| H3 | 576 | 9×9/s8 | 1024 | 100M | 4× hidden |

Winner propagates to all subsequent runs.

### Tier 1: Architecture Comparison (5 runs)

All at 576-dim, primary config (9×9/stride-8), 200M tokens, same everything except FFN/MoE component.

| Run | Architecture | Purpose |
|-----|-------------|---------|
| 1 | Dense FFN | Floor baseline |
| 2 | Standard Sparse MoE | MoE baseline |
| 3 | SliceMoE-Flat | Does overlap help? |
| 4 | Flat Grid Overlap | Does torus help? |
| 5 | Full RTCC | The architecture |

### Tier 2: Privacy Sweep (4 new runs + reuse Run 5)

All RTCC at 576-dim, 200M tokens. Vary patch/stride config only.

| Run | Patch | Stride | Experts | PrivDims | Privacy% |
|-----|-------|--------|---------|----------|---------|
| 6  | 3×3   | 2      | 144     | 1        | 11.1%   |
| 7  | 4×4   | 3      | 64      | 4        | 25.0%   |
| 8  | 5×5   | 4      | 36      | 9        | 36.0%   |
| 9  | 7×7   | 6      | 16      | 25       | 51.0%   |
| 5  | 9×9   | 8      | 9       | 49       | 60.5% ← reuse |

### Tier 3: Communication Ablations (inference-only, no training)

Run on saved Run 5 checkpoint. Zero compute cost.

| Test | Method | Tests |
|------|--------|-------|
| Zero shared dims | Zero border dims at inference | Is communication real? |
| Expert shuffle | Permute expert positions at inference | Is spatial specialization real? |

### Tier 4: Column Depth Sweep (3 runs)

Vary R (loop count) at best config from Tier 1-2.

| Run | R loops | Purpose |
|-----|---------|---------|
| 10 | R=2 | Shallow |
| 11 | R=4 | Medium |
| 5  | R=6 | Target ← reuse |
| 12 | R=8 | Training ceiling |

### Tier 5: Expert Specialization Analysis (inference-only)

On Run 5 checkpoint:
- Expert activation entropy (low = more specialized)
- RSA similarity analysis (toroidal neighbors should be more similar)

### Tier 6: Benchmark Run (1 run)

576-dim, best config, 1B tokens, full lm-eval suite.

| Run | Dim | Tokens | Benchmarks |
|-----|-----|--------|-----------|
| 13 | 576 | 1B | WikiText-103, LAMBADA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, PIQA |

### Scale Run: 1600-dim Benchmark (1 run)

Single run, no sweep, primary config only.

| Run | Dim | Config | Tokens | Purpose |
|-----|-----|--------|--------|---------|
| 14 | 1600 | 11×11/s10 | 1B+ | Benchmark comparison at larger scale |

**Total training runs: 13 (Tier 0 ablation) or 10 (skip Tier 0) + 1 scale run**

---

## Benchmark Comparison Strategy

### At 576-dim (~63-72M params depending on hidden dim)

Direct comparison is difficult at this scale — few published models exist at exactly this parameter count. Compare primarily against:
- Your own internal baselines (Dense FFN, Standard MoE) at matched compute
- Pythia-70M and Pythia-160M (EleutherAI) as bracketing references
- Report FLOPs-normalized results, not raw benchmark scores

### At 1600-dim (~347M total params, ~410M effective)

This is the publishable comparison scale. Comparable published models:

| Model | Params | Trained | Notes |
|-------|--------|---------|-------|
| GPT-2 Medium | 345M | 40B tokens | Universal baseline |
| OPT-350M | 350M | 180B tokens | Meta, fully open |
| Pythia-410M | 410M | 300B tokens | Purpose-built for comparison, published at many token counts |
| Cerebras-GPT-256M | 256M | 26B tokens | Clean compute-matched study |

**Critical caveat:** These models were trained on 26-300B tokens. Your 1600-dim run targets 1B tokens. This is a 26-300× gap in training data. Raw benchmark comparison is unfair.

**Recommended framing:**
1. Compare against Pythia-410M at its 1B-token checkpoint specifically (they publish intermediate checkpoints)
2. Report compute-normalized: FLOPs = 6 × params × tokens (Chinchilla formula)
3. Primary comparison is always RTCC-1600 vs Dense-FFN-1600 at matched budget — your own controlled baseline

**Benchmarks to run via lm-eval-harness:**
- WikiText-103 perplexity
- LAMBADA accuracy
- HellaSwag 0-shot accuracy
- WinoGrande 0-shot accuracy
- ARC-Easy 0-shot accuracy
- ARC-Challenge 0-shot accuracy
- PIQA 0-shot accuracy

---

## Key Architectural Decisions Still Open

1. **expert_hidden dim** — 256, 512, or 1024? Ablate at Tier 0 or pick 512 and commit.
2. **P=6 prelude** — confirmed from CART. Use in all RTCC runs.
3. **LTI formulation** — use CART's exact formulation (sigmoid gating). Do not modify.
4. **Hyper-connections** — match CART exactly (placement and n=3).
5. **LIE (Loop Index Embedding)** — match CART exactly (include if CART uses it).
6. **SSM** — omit from RTCC paper runs. ToroidalMoE is the only variable vs baselines.

---

## Component Attribution

| Component | Source |
|---|---|
| Prelude/Core/Coda structure | Huginn (Geiping et al., 2025) + Hyperloop (arXiv 2604.21254) |
| LTI injection | OpenMythos / Claude Mythos (kyegomez, arXiv 2604.21215) |
| Hyper-connections | Hyperloop (arXiv 2604.21254) |
| P=6/R=8 optimal config | CART original ablation (Capps, 2026) |
| MLA attention | DeepSeek-V2 (Dai et al., arXiv 2405.04434) |
| RoPE | Su et al., 2021 |
| SwiGLU | Shazeer, 2020 |
| RMSNorm | Zhang & Sennrich, 2019 |
| Toroidal Sheet MoE | **This work (Capps, 2026)** |
| Cortical column emergence | **This work (Capps, 2026)** |

---

## Project Locations

```
K:\projects\RTCC_Paper_2\          ← project root
K:\projects\RTCC_Paper_2\data\validation\  ← held-out val set (created ONCE before training)
K:\projects\Model_Paper_1\data\hf_cache   ← training data (already downloaded)
K:\projects\Model_Paper_1\                ← CART codebase (reference for porting)
```

**GitHub:** https://github.com/ccapps42/RTCC (MIT license)  
**CART:** https://github.com/ccapps42/CART  
**Hardware:** RTX 3050 8GB (dev/debug only), RTX 3090 24GB (ALL paper runs)  
**All paper runs on 3090** — single hardware disclosure sentence in paper.

---

## What Claude Code Should Do First

1. Read `K:\projects\Model_Paper_1\` codebase — understand CART's exact LTI, hyper-connection, and LIE implementations
2. Create directory structure under `K:\projects\RTCC_Paper_2\`
3. Initialize SQLite database with full schema
4. Run `prepare_validation.py` ONCE — carve held-out val set before any training
5. Port shared components from CART (MLA attention, RMSNorm, RoPE, LTI, data pipeline)
6. Implement ToroidalMoE with unit tests for:
   - Toroidal padding correctness (circular wrap)
   - Overlap-add fold is exact inverse of unfold
   - Gradient flows through shared border dims
   - RTCCConfig assertions fire on invalid configs
7. Smoke test at dim=288 (small, fast) before 576-dim
8. Queue Tier 0 hidden dim ablation OR commit to expert_hidden=512
