# RTCC Sweep Plan — d=576
**Paper 2: Recurrent Toroidal Cortical Columns (Capps, 2026)**  
Last updated: 2026-05-07

---

## Purpose

This document defines the complete sweep experiment plan for Paper 2. The sweep model (d=576) characterizes how privacy percentage and overlap depth affect model quality. The benchmark model (d=TBD) will be covered in a separate document.

All runs in this document are at d=576. No benchmark comparisons against published models are made here — that is the benchmark doc's job. The primary claims this sweep supports:

1. Privacy percentage has a measurable effect on quality (the privacy curve).
2. Overlap depth (1-cell vs 2-cell) has an independent effect at matched privacy.
3. The toroidal boundary contributes meaningfully (vs flat zero-padding).
4. Position-fixed dimensional experts specialize spatially.

---

## Locked architecture

These values are fixed across all runs. Do not vary them.

| Parameter | Value | Source |
|---|---|---|
| model_dim | 576 | This paper |
| grid | 24×24 | 576 = 24×24 |
| n_heads | 9 | 576 / 64 = 9, head_dim = 64 |
| mla_latent | 144 | dim / 4 |
| vocab_size | 32,000 | Llama-2 BPE — matches CART exactly |
| Prelude layers (P) | 6 | CART-validated across all tested dims |
| Max loops (R) | **TBD** | Waiting for CART Stage 2 results — R ∈ {6, 8, 10} being tested at d=512 and d=768; inherit winner before starting any RTCC run |
| Coda layers | 1 | CART-validated |
| LTI formulation | sigmoid gating `h = sigmoid(a)·h + transformer_out` | From CART / OpenMythos |
| Hyper-connections | n=3 | From CART / Hyperloop |
| Attention | MLA (DeepSeek-V2) — decoupled RoPE, KV latent = dim/4 | |
| FFN (prelude + coda) | Dense SwiGLU, width = 4 × dim | |
| Recurrent FFN | ToroidalMoE (this work) | |
| Normalization | RMSNorm | |

The recurrent core contains exactly one shared block (MLA + ToroidalMoE + LTI + hyper-connections) applied ×R loops. This is the only component that differs between RTCC and baselines.

---

## Pre-training decisions

Lock these before starting any training run. They propagate to all runs.

### 1. expert_hidden dimension

The ToroidalMoE expert MLP is: `patch_dims → expert_hidden → patch_dims` (SwiGLU, 3 matrices).

| Option | Notes |
|---|---|
| 256 | Ratio varies 2.1×–12.6× across sweep configs. Below standard for large patches. |
| **512** | **Recommended.** Lands at ~4× standard ratio for mid-range configs. Clean paper justification. |
| 1024 | Generous. Recurrent core grows meaningfully, potentially confounding sweep results. |

**Recommendation: commit to 512.** Alternatively, run a 3-shot hidden-dim ablation at 100M tokens on config S7 (256 / 512 / 1024) before the main sweep. If ablating, do it first — result propagates to all 11 training runs.

### 2. Token budget per sweep run

**Locked: 500M tokens per sweep run.** 200M is borderline for detecting subtle differences (e.g., toroidal vs. flat boundary, adjacent privacy configs). At 500M, the privacy curve ordering stabilizes, pair comparisons are meaningful, and expert specialization has time to emerge.

**Profiled time estimates (RTX 3090, d=576 R=8, seq=1024, batch=4, grad_accum=8):**
- Measured throughput: **10,811 tok/s** (smoke test, 100 steps, kernels pre-warmed)
- Per run: 15,259 steps × 32,768 tok/step ÷ 10,811 tok/s = **~12.8 hours**
- Sequential (1 slot): 11 runs × 12.8h = **~141 hours (~5.9 days)**
- 2 concurrent slots: ~38 hours (~1.6 days) — peak VRAM 2 × 4.95 GB = 9.9 GB
- 3 concurrent slots: ~26 hours (~1.1 days) — peak VRAM 3 × 4.95 GB = 14.85 GB, ~9 GB headroom

Use `python scripts/orchestrate_paper.py --slots 3` to run 3 concurrent with safe headroom on the 3090 (24 GB total).

### 3. Sequence length

**Locked: seq_len=1024**, matching CART Stage 2. CART's stage2_train.bin was interleaved in 1024-token chunks; using 512 would split documents at wrong boundaries. Same effective batch as CART: 4 × 8 × 1024 = 32,768 tokens/step.

### 4. Data mixture

CART uses: 300M TinyStories + 300M Wikipedia + 400M FineWeb-Edu = 1B tokens total.  
For sweep runs at 500M tokens each, confirm whether to use a proportional slice of the same mixture or a modified mixture (e.g., drop TinyStories for quality-focused runs).

---

## The 7 sweep configurations

All runs: d=576, 24×24 grid, locked architecture above.  
Sorted by privacy percentage ascending.

| Run | Overlap | Stride | Patch | Experts | Patch dims | Private dims | Privacy % | Role |
|-----|---------|--------|-------|---------|------------|--------------|-----------|------|
| S1 | 2-cell | 4 | 6×6 | 6×6 = 36 | 36 | 4 | 11.1% | Low-privacy anchor |
| S2 | 1-cell | 3 | 4×4 | 8×8 = 64 | 16 | 4 | 25.0% | Pair A |
| S3 | 2-cell | 6 | 8×8 | 4×4 = 16 | 64 | 16 | 25.0% | Pair A |
| S4 | 1-cell | 4 | 5×5 | 6×6 = 36 | 25 | 9 | 36.0% | Pair B |
| S5 | 2-cell | 8 | 10×10 | 3×3 = 9 | 100 | 36 | 36.0% | Pair B |
| S6 | 1-cell | 6 | 7×7 | 4×4 = 16 | 49 | 25 | 51.0% | Mid-high anchor |
| S7 | 1-cell | 8 | 9×9 | 3×3 = 9 | 81 | 49 | 60.5% | High-privacy anchor |

**Configuration constraints verified:**

| Run | grid % stride check | Experts per axis | private_side | Valid |
|-----|---------------------|------------------|--------------|-------|
| S1 | 24 % 4 = 0 ✓ | 6 × 6 (min=6) ✓ | 4-2=2 ✓ | ✓ |
| S2 | 24 % 3 = 0 ✓ | 8 × 8 (min=8) ✓ | 3-1=2 ✓ | ✓ |
| S3 | 24 % 6 = 0 ✓ | 4 × 4 (min=4) ✓ | 6-2=4 ✓ | ✓ |
| S4 | 24 % 4 = 0 ✓ | 6 × 6 (min=6) ✓ | 4-1=3 ✓ | ✓ |
| S5 | 24 % 8 = 0 ✓ | 3 × 3 (min=3) ✓ | 8-2=6 ✓ | ✓ ⚠ |
| S6 | 24 % 6 = 0 ✓ | 4 × 4 (min=4) ✓ | 6-1=5 ✓ | ✓ |
| S7 | 24 % 8 = 0 ✓ | 3 × 3 (min=3) ✓ | 8-1=7 ✓ | ✓ ⚠ |

⚠ S5 and S7 use exactly the minimum 3 experts per axis. Valid per RTCCConfig assertions but note in paper.

**Privacy formula by overlap:**
- 1-cell: `privacy = (stride−1)² / (stride+1)²`
- 2-cell: `privacy = (stride−2)² / (stride+2)²`

---

## Embedded ablations

These ablations are built into the sweep at no extra cost. They answer questions the main privacy curve cannot.

### Pair A — overlap depth at 25% privacy

| | S2 | S3 |
|---|---|---|
| Overlap | 1-cell | 2-cell |
| Patch | 4×4 | 8×8 |
| Experts | 64 (8×8) | 16 (4×4) |
| Private dims | 4 | 16 |
| Privacy % | 25.0% | 25.0% |

S2 and S3 have identical privacy percentages but differ in overlap depth, patch size, and expert count. The communication border in S3 is wider (2 cells vs 1), meaning each shared dimension receives gradient from more neighboring experts.

**Interpretation:**
- If S3 outperforms S2: wider overlap (deeper inter-expert communication) adds value beyond what privacy% alone captures.
- If S2 ≈ S3: privacy% is the dominant variable; overlap width is secondary.

### Pair B — overlap depth at 36% privacy

| | S4 | S5 |
|---|---|---|
| Overlap | 1-cell | 2-cell |
| Patch | 5×5 | 10×10 |
| Experts | 36 (6×6) | 9 (3×3) |
| Private dims | 9 | 36 |
| Privacy % | 36.0% | 36.0% |

Same question as Pair A, at higher privacy. S5 has 4× more private dims than S4 for the same privacy percentage, because the larger patch allocates the same *fraction* to a larger absolute core.

**Note:** The Pair A and B results together can identify whether the overlap effect is consistent across privacy levels — a potentially publishable finding in itself.

---

## Required baselines

These 4 runs are required for the paper. The RTCC sweep results are uninterpretable without them.

| Run | Architecture | Config | Purpose |
|-----|-------------|--------|---------|
| B1 | Dense FFN | Standard recurrent SwiGLU, hidden = 4 × 576 = 2304 | Absolute floor. Establishes what a plain recurrent-depth transformer achieves. |
| B2 | Standard Sparse MoE | 16 experts, top-2 routing, full 576-dim per expert, load-balance aux loss | MoE baseline. RTCC must match or beat this to have a publishable claim. |
| B3 | SliceMoE-Flat | 64 experts × 9 dims each, non-overlapping 1D partition, no torus | Closest prior art (Vejendla, arXiv 2510.04286, 2024). Isolates whether overlap contributes over simple dimensional slicing. |
| B4 | Flat Grid Overlap | Same as S7 (9×9 patch, stride 8, 3×3 = 9 experts) but zero-padding instead of circular wrap | Isolates the toroidal boundary contribution specifically. Implementation: `F.unfold` with `padding_mode='zeros'`. |

All baselines at d=576, same training setup as sweep runs (same token budget, same curriculum, same data).

### What each baseline claim supports

- **B1 beats nothing:** Expected. Dense FFN with 23.9M recurrent params vs RTCC's ~0.6–1.1M. The efficiency story.
- **RTCC matches or beats B2:** The central MoE claim.
- **RTCC beats B3:** The overlap contribution is real.
- **RTCC beats B4:** The toroidal boundary is doing real work, not just the overlapping patches.

If RTCC does not beat B3 or B4, those are honest findings that still contribute to the field.

---

## Training setup

### Hardware

| Use | Machine |
|---|---|
| Smoke tests, debug | RTX 3050 (8GB) |
| All paper runs | RTX 3090 (24GB) |

Single hardware disclosure sentence required in paper.

### File locations

| Path | Contents |
|---|---|
| `K:\projects\RTCC_Paper_2\` | Project root |
| `K:\projects\RTCC_Paper_2\data\validation\` | Held-out validation set — create ONCE before any training |
| `K:\projects\Model_Paper_1\data\hf_cache\` | Training data (already downloaded) |
| `K:\projects\Model_Paper_1\` | CART codebase — reference for porting shared components |

### Validation set

Create the held-out validation set once, before any training run begins. All 11 training runs (7 sweep + 4 baselines) are evaluated on the same set. Do not regenerate.

### Per-run estimated VRAM (d=576)

- Seq=512, batch=4, grad_accum=8, 8-bit Adam: ~1.0–1.5GB (comfortable on 3090)
- Multiple concurrent runs may be possible on the 3090 depending on batch size

---

## Evaluation

### Primary metric

Validation perplexity on the held-out set, evaluated at the end of training and at intermediate checkpoints (save at minimum every 25M tokens).

### Inference-only analyses

Run on the best-performing RTCC checkpoint after the sweep is complete. Zero additional training compute.

**Communication ablation (run 1):**
- At inference, zero all shared border dimensions before each expert forward pass.
- Compare perplexity to the unmodified checkpoint.
- If perplexity degrades significantly: the inter-expert communication via overlap is real and doing measurable work.
- If perplexity is unchanged: the communication is not contributing — a negative result worth reporting.

**Communication ablation (run 2):**
- At inference, permute expert positions randomly on the torus (shuffle which expert sees which patch location).
- Compare perplexity to the unmodified checkpoint.
- If perplexity degrades: experts have specialized spatially to their positions — spatial organization emerged from training.
- If perplexity is unchanged: position-fixed routing did not produce spatial specialization.

**Expert specialization analysis:**
- Compute per-expert activation entropy over the validation set. Low entropy = more specialized.
- Run RSA (Representational Similarity Analysis) between adjacent experts on the torus. Toroidal neighbors should be more representationally similar than non-adjacent experts if spatial structure emerged.

These analyses are section-level contributions to the paper independent of benchmark scores.

---

## Run order

Dependencies and rationale.

### Phase 0 — Verify before committing

1. **Smoke test at d=288** — Small, fast. Verify that:
   - ToroidalMoE forward pass produces correct output shape
   - Toroidal padding wraps correctly (circular, not zero-pad)
   - Overlap-add fold is the exact inverse of unfold
   - Gradient flows through shared border dimensions
   - RTCCConfig assertions fire on invalid configurations (e.g., private_side < 1)
   - Training loss decreases

2. **Lock pre-training decisions** — expert_hidden, token budget, curriculum, data mixture.

### Phase 1 — Establish anchors (run in parallel if possible)

3. **B1 (Dense FFN)** — Establishes the floor. Run this first so you know where baseline performance lands before interpreting RTCC results.
4. **S7 (1-cell, stride 8, 60.5%)** — Primary RTCC config. Run early to verify stable training at the target architecture.

### Phase 2 — Complete baselines

5. **B2 (Standard Sparse MoE)**
6. **B3 (SliceMoE-Flat)**
7. **B4 (Flat Grid Overlap)**

### Phase 3 — Remaining sweep runs

8. **S1** — Low-privacy anchor (2-cell, stride 4)
9. **S2, S3** — Pair A (25% privacy)
10. **S4, S5** — Pair B (36% privacy)
11. **S6** — Mid-high anchor (1-cell, stride 6)

### Phase 4 — Inference-only analyses

12. Communication ablations on best RTCC checkpoint.
13. Expert specialization analysis on best RTCC checkpoint.

**Total training runs: 11** (7 sweep + 4 baselines)  
**Additional inference-only: 4** (2 communication ablations + 2 specialization analyses, zero training compute)

---

## Open questions

Resolve before Phase 1.

| # | Question | Recommendation |
|---|---|---|
| 1 | expert_hidden: 256, 512, or 1024? | Commit to 512, or ablate first (3 runs × 100M tokens on S7) |
| 2 | Token budget per sweep run? | **Locked: 500M** |
| 3 | Sequence length curriculum for sweep? | Confirm max seq_len |
| 4 | Data mixture for sweep runs? | Confirm proportional slice of CART mixture or modified |
| 5 | **R (max loops)?** | **Blocked on CART Stage 2.** Check `K:/projects/Model_Paper_1/results.db` for d=512 and d=768 Stage 2 results across R ∈ {6, 8, 10}. Inherit the winning R before starting any RTCC run. |

---

## Key formulas (reference)

```
overlap      = patch_size - stride
private_side = patch_size - 2 × overlap
private_dims = private_side²
privacy%     = private_dims / patch_size²

1-cell overlap: stride = patch - 1 → privacy = (stride-1)² / (stride+1)²
2-cell overlap: stride = patch - 2 → privacy = (stride-2)² / (stride+2)²

RTCCConfig guards:
  assert private_side >= 1
  assert grid_rows % stride == 0
  assert grid_cols % stride == 0
  assert grid_rows // stride >= 3
  assert grid_cols // stride >= 3
```

---

## Component attribution (for paper)

| Component | Source |
|---|---|
| Prelude/Core/Coda structure | Huginn (Geiping et al., 2025) + Hyperloop (arXiv 2604.21254) |
| LTI injection | OpenMythos / Claude Mythos (kyegomez, arXiv 2604.21215) |
| Hyper-connections | Hyperloop (arXiv 2604.21254) |
| P=6 / R=8 optimal config | CART original ablation (Capps, 2026) |
| MLA attention | DeepSeek-V2 (Dai et al., arXiv 2405.04434) |
| RoPE | Su et al., 2021 |
| SwiGLU | Shazeer, 2020 |
| RMSNorm | Zhang & Sennrich, 2019 |
| SliceMoE prior art | Vejendla, arXiv 2510.04286, 2024 |
| **Toroidal Sheet MoE** | **This work (Capps, 2026)** |
| **Cortical column emergence** | **This work (Capps, 2026)** |
