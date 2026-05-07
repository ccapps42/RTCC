# RTCC Benchmark Plan — d=1600
**Paper 2: Recurrent Toroidal Cortical Columns (Capps, 2026)**  
Last updated: 2026-05-03

---

## Purpose

This document defines the benchmark model for Paper 2. The benchmark model is a single larger-scale run whose job is to compare RTCC against published models at a parameter count that reviewers recognize. It is not a sweep — config choices are locked before training starts.

The sweep model (d=576, covered in RTCC_Sweep_Plan.md) should be complete before the benchmark run begins so that sweep findings can inform any final config decisions.

---

## Locked architecture

| Parameter | Value | Source |
|---|---|---|
| model_dim | 1600 | This paper |
| grid | 40×40 | 1600 = 40×40 |
| n_heads | 25 | 1600 / 64 = 25, head_dim = 64 |
| mla_latent | 400 | dim / 4 |
| vocab_size | 32,000 | Llama-2 BPE — matches CART exactly |
| Prelude layers (P) | 6 | CART-validated |
| Core loops (R) | 10 | See rationale below |
| Coda layers | 1 | CART-validated |
| expert_hidden | 512 | ~4× standard ratio at primary config |
| LTI formulation | sigmoid gating | From CART / OpenMythos |
| Hyper-connections | n=3 | From CART / Hyperloop |
| Attention | MLA (DeepSeek-V2) — decoupled RoPE, KV latent = dim/4 | |
| FFN (prelude + coda) | Dense SwiGLU, width = 4 × dim = 6400 | |
| Recurrent FFN | ToroidalMoE (this work) | |
| Normalization | RMSNorm | |

**Rationale for R=10:** R=10 pushes the effective parameter count into the ~474M range, which makes the comparison against Pythia-410M at its 1B-token checkpoint more credible. R=8 (~446M effective) is acceptable if training time is prohibitive. Decision should be made after profiling training speed on the 3090 at d=1600.

---

## Configs

### Primary — 11×11 patch, 1-cell overlap (preferred)

```
stride:        10
patch:         11×11
overlap:       1-cell  (patch - stride = 1)
experts:       4×4 = 16
patch_dims:    121
private_side:  9  (11 - 2×1)
private_dims:  81  (9²)
privacy:       66.9%  (81/121)
expert_hidden: 512
h/patch ratio: 512/121 = 4.23×  ≈ standard 4× FFN expansion
```

**Config validation:**
- 40 % 10 = 0 ✓
- 40 / 10 = 4 experts per axis ✓ (above minimum of 3)
- private_side = 9 ≥ 1 ✓

### Optional — 12×12 patch, 2-cell overlap

```
stride:        10
patch:         12×12
overlap:       2-cell  (patch - stride = 2)
experts:       4×4 = 16
patch_dims:    144
private_side:  8  (12 - 2×2)
private_dims:  64  (8²)
privacy:       44.4%  (64/144)
expert_hidden: 512
h/patch ratio: 512/144 = 3.56×
```

Run only if the primary config trains successfully and compute budget allows. The 12×12 config provides a meaningful comparison — same stride, same expert count, same model dim, different privacy level — which could become a compact ablation within the benchmark section of the paper.

---

## Parameter counts

Fixed base (P=6 prelude, coda=1, d=1600):

| Component | Params |
|---|---|
| Embedding (50K × 1600) | 80.000M |
| Prelude MLA ×6 | 34.560M |
| Prelude Dense FFN ×6 | 184.320M |
| Coda MLA ×1 | 5.760M |
| Coda Dense FFN ×1 | 30.720M |
| RMSNorms | 0.019M |
| **Non-recurrent subtotal** | **335.379M** |

Recurrent per loop (non-MoE):

| Component | Params / loop |
|---|---|
| Recurrent MLA | 5.760M |
| LTI matrices A+B | 5.120M |
| Hyper-connections (n=3) | 0.005M |
| **Recurrent non-MoE subtotal** | **10.885M / loop** |

### Primary config (11×11, h=512, R=10)

| Metric | Value |
|---|---|
| MoE params | 3 × 16 × 121 × 512 / 1M = **2.974M** |
| **Total (real) params** | **349.2M** |
| **Effective params (R=10)** | **474.0M** |
| Real → effective delta | **+35.7%** |
| h / patch_dims ratio | 512 / 121 = **4.23×** |

Effective = 335.379 + 10 × (10.885 + 2.974) = 335.379 + 138.590 = **473.97M**

### Optional config (12×12, h=512, R=10)

| Metric | Value |
|---|---|
| MoE params | 3 × 16 × 144 × 512 / 1M = **3.539M** |
| **Total (real) params** | **349.8M** |
| **Effective params (R=10)** | **479.6M** |
| Real → effective delta | **+37.1%** |
| h / patch_dims ratio | 512 / 144 = **3.56×** |

### Dense FFN baseline (1600-dim, R=10)

| Metric | Value |
|---|---|
| Dense FFN per loop (4× width) | 3 × 1600 × 6400 / 1M = **30.720M** |
| **Total (real) params** | **377.0M** |
| **Effective params (R=10)** | **751.4M** |
| Real → effective delta | **+99.3%** |
| Recurrent MoE vs Dense FFN | **2.974M vs 30.720M — 10.3× reduction** |

The 10.3× reduction in the looped component is the core efficiency story of the paper.

---

## Required baseline

One baseline is required for the benchmark model. This is the same Dense FFN baseline described above, run at d=1600, same training setup as the benchmark run.

The sweep model baselines (B1–B4 at d=576) are for the sweep paper section. The d=1600 Dense FFN is for the benchmark section specifically. Without it, you cannot make any claim about RTCC quality at this scale.

---

## Comparable published models

### Parameter-matched comparisons (~349M total params)

| Model | Params | Tokens trained | Notes |
|---|---|---|---|
| GPT-2 Medium | 345M | ~40B (WebText) | Different arch dim (d=1024). Token gap: 40× |
| OPT-350M | 350M | 180B (The Pile) | Closest param match. Token gap: 180× |
| Pythia-410M | 410M | 300B (The Pile) | Slightly larger. **Has 1B-token checkpoint.** Token gap at 1B: none |

### Effective-parameter comparisons (~474M effective, R=10)

| Model | Params | Tokens at comparison | Notes |
|---|---|---|---|
| Pythia-410M | 410M | 1B checkpoint | **Best external comparison — matched token budget** |
| Pythia-1B | 1B | 1B checkpoint | Upper reference; RTCC approaches this scale via recurrent reuse |

### Comparison strategy

**Primary (always use):** RTCC-1600 vs Dense-FFN-1600 at matched compute. Same architecture, same data, same hardware, only the recurrent body differs. This comparison is unimpeachable and is the central result of the benchmark section.

**Secondary (use with framing):** RTCC-1600 vs Pythia-410M at Pythia's 1B-token checkpoint. Token budget matches; parameter count is close. This is a fair external comparison. Cite the specific Pythia checkpoint, not the final model.

**Reference only (mention with caveats):** GPT-2 Medium and OPT-350M. Note explicitly that these were trained on 40–180× more tokens. Present as upper-bound references, not head-to-head results. Do not include these in benchmark score tables — mention them in the discussion section only.

### Chinchilla context

Chinchilla-optimal token count for ~350M params is approximately 7B tokens. At 1B tokens this model is trained to roughly 14% of compute-optimal. This is not unusual for an architecture paper — the goal is to show the architecture works and scales, not to produce a state-of-the-art checkpoint. Frame accordingly in the paper.

If training budget allows 2B tokens, absolute benchmark scores will improve meaningfully and the Pythia comparison becomes stronger. Worth profiling time cost before committing.

---

## Training setup

### Hardware

| Use | Machine |
|---|---|
| Smoke tests, debug | RTX 3050 (8GB) |
| All paper runs | RTX 3090 (24GB) |

Estimated VRAM (d=1600, seq=512, 8-bit Adam): ~3–4GB. Comfortable on the 3090.  
**Profile actual tokens/sec at d=1600 before committing to token budget.**

### File locations

| Path | Contents |
|---|---|
| `K:\projects\RTCC_Paper_2\` | Project root |
| `K:\projects\RTCC_Paper_2\data\validation\` | Held-out validation set — shared with sweep model |
| `K:\projects\Model_Paper_1\data\hf_cache\` | Training data |

The validation set is created once for the entire paper. Both sweep and benchmark models are evaluated on the same held-out set.

### Sequence length curriculum (recommended)

| Phase | seq_len | Loops | Tokens | Notes |
|---|---|---|---|---|
| 1 | 128 | 2–5 | ~100M | Warm-up, verify stability |
| 2 | 256 | 4–8 | ~200M | Build context capacity |
| 3 | 512 | 6–10 | ~300M | Full R ceiling reached |
| 4 | 1024 | 8–10 | ~400M | Long-context, benchmark-relevant |
| **Total** | | | **~1B tokens** | |

Extend Phase 4 if compute allows. Benchmark scores (especially LAMBADA and HellaSwag) improve with longer sequence training.

### Data mixture

Confirm before training. Recommended starting point — same proportional mixture as CART:

| Source | Fraction | ~Tokens at 1B budget |
|---|---|---|
| Wikipedia | 30% | 300M |
| FineWeb-Edu | 40% | 400M |
| FineWeb (general) | 20% | 200M |
| Books (PG19 or similar) | 10% | 100M |

Do not include TinyStories in the benchmark run. It is appropriate for language modeling validation at small scale (CART) but will hurt benchmark scores on commonsense reasoning tasks.

If books corpus is unavailable, reallocate its share to FineWeb-Edu.

---

## Evaluation

### Benchmarks via lm-eval-harness

Run on the final checkpoint. All 0-shot unless noted.

| Benchmark | Metric | Notes |
|---|---|---|
| WikiText-103 | Perplexity | Primary language modeling metric |
| LAMBADA | Accuracy (0-shot) | Long-range dependency — favors recurrent models |
| HellaSwag | Accuracy (0-shot) | Commonsense completion |
| WinoGrande | Accuracy (0-shot) | Commonsense reasoning |
| ARC-Easy | Accuracy (0-shot) | Factual QA |
| ARC-Challenge | Accuracy (0-shot) | Harder factual QA |
| PIQA | Accuracy (0-shot) | Physical commonsense |

Compare all results against:
1. Dense-FFN-1600 (internal baseline, same token budget) — primary comparison
2. Pythia-410M at 1B-token checkpoint — external comparison

### Reporting format

Report three numbers for RTCC-1600: total params, effective params (R=10), and token count. This is non-standard and must be explained clearly in the paper. Suggested framing:

> "RTCC-1600 has 349M unique parameters. At R=10 recurrent loops, the model applies 474M parameter-equivalents per forward pass — a 35.7% increase in effective capacity at zero memory cost beyond a single loop body."

### Intermediate checkpoints

Save checkpoints at minimum every 100M tokens. This allows:
- Comparison against Pythia at its specific checkpoint intervals
- Training curve analysis (are benchmark scores improving smoothly?)
- Recovery point if training destabilizes at longer sequences

---

## Run order

1. **Verify sweep model is trained** — sweep findings should be available before this run.
2. **Profile training speed at d=1600** — tokens/sec at seq=128 and seq=512. Use this to set realistic token budget before committing.
3. **Dense-FFN-1600 baseline** — run in parallel with or before the RTCC run if hardware permits. Establishes the floor at this scale.
4. **RTCC-1600 primary** — 11×11 patch, 1-cell, R=10, full curriculum.
5. **RTCC-1600 optional** — 12×12 patch, 2-cell, run only if primary is successful and budget remains.
6. **lm-eval benchmarks** — on final checkpoint of each completed run.

**Total training runs: 2 required** (Dense-FFN baseline + RTCC primary)  
**Optional: 1** (RTCC secondary 12×12 config)

---

## Key formulas (reference)

```
Primary config:
  overlap      = 11 - 10 = 1  (1-cell)
  private_side = 11 - 2×1 = 9
  private_dims = 9² = 81
  privacy%     = 81/121 = 66.9%

Optional config:
  overlap      = 12 - 10 = 2  (2-cell)
  private_side = 12 - 2×2 = 8
  private_dims = 8² = 64
  privacy%     = 64/144 = 44.4%

Effective params = non_recurrent + R × (recurrent_non_MoE + MoE_params)
                 = 335.379 + 10 × (10.885 + 2.974)
                 = 473.97M  (primary config, h=512, R=10)

Dense FFN recurrent body = 3 × 1600 × 6400 = 30.720M/loop
RTCC recurrent body      = 2.974M/loop  (primary config)
Reduction                = 10.3×
```

---

## Component attribution (for paper)

| Component | Source |
|---|---|
| Prelude/Core/Coda structure | Huginn (Geiping et al., 2025) + Hyperloop (arXiv 2604.21254) |
| LTI injection | OpenMythos / Claude Mythos (kyegomez, arXiv 2604.21215) |
| Hyper-connections | Hyperloop (arXiv 2604.21254) |
| P=6 / R=8–10 optimal config | CART original ablation (Capps, 2026) |
| MLA attention | DeepSeek-V2 (Dai et al., arXiv 2405.04434) |
| RoPE | Su et al., 2021 |
| SwiGLU | Shazeer, 2020 |
| RMSNorm | Zhang & Sennrich, 2019 |
| **Toroidal Sheet MoE** | **This work (Capps, 2026)** |
| **Cortical column emergence** | **This work (Capps, 2026)** |
