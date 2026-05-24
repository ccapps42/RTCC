# Recurrent Toroidal Cortical Columns (RTCC)

**Paper:** *Recurrent Toroidal Cortical Columns: A Neocortically-Inspired Architecture for Efficient Language Modeling*
**Author:** Chad Capps ([@ccapps42](https://github.com/ccapps42))
**Status:** Experiments in progress — paper forthcoming
**License:** MIT

---

## Overview

RTCC is a language model architecture that replaces the dense feed-forward layer in a single transformer block — the **coda** of a recurrent-depth transformer — with a **Top-K Toroidal Sheet Mixture of Experts (TS-MoE)**. The rest of the model is byte-for-byte identical to the CART backbone ([ccapps42/CART](https://github.com/ccapps42/CART)), so the RTCC-vs-CART perplexity delta is attributable to exactly one architectural change.

The Top-K Toroidal Sheet MoE:

* Up-projects the residual stream into a fixed-size **2D toroidal grid** (64×64 = 4096 dims)
* Tiles the grid with **position-fixed experts** on overlapping patches
* A **linear router** picks the **top-K of 64 experts** per token
* Each selected expert processes its fixed patch via a small SwiGLU MLP
* Outputs are routing-weighted, scattered to per-expert positions, and **overlap-add folded** back to the grid
* **Toroidal wrap-add** ensures every pixel of the grid receives the full set of expert contributions, including from wraparound neighbors

The toroidal geometry creates implicit inter-expert communication through overlapping dimensional ownership — shared border dimensions receive gradient from multiple experts simultaneously, with no explicit lateral connections.

---

## Architecture

RTCC builds on **CART** ([ccapps42/CART](https://github.com/ccapps42/CART)) and differs in exactly one place: the FFN slot of the single coda layer.

```
Input Tokens
    ↓
[Embedding]  vocab × dim   (32K Llama-2 BPE, dim=1024, weight-tied to LM head)
    ↓
[Prelude]  ×6   MLA self-attn + SwiGLU FFN   (CART exactly)
    ↓
[Recurrent Core]  ×R=6 loops, shared weights
    ┌──────────────────────────────────────────────┐
    │  hyper.combine(buffer)  → h_input            │  ← softmax-weighted last 3 states
    │  LIE(r)                 → h_input            │  ← sinusoidal loop-index signal
    │  MLA cross-attn(h, K, V from prelude)        │  ← K, V precomputed from prelude e
    │  SwiGLU FFN             → block_out          │  ← CART exactly
    │  h = sigmoid(A)·h_input + block_out          │  ← LTI gating, ρ < 1 guaranteed
    │  hyper.update_buffer(buffer, h)              │
    └──────────────────────────────────────────────┘
    ↓
[Coda]  ×1   MLA self-attn + (up_proj → TopKToroidMoE → down_proj)   ← the novel layer
    ↓
[RMSNorm → LM Head]
    ↓
Output Logits
```

### The Coda Top-K Toroidal MoE

```
x [B, S, 1024]
  → norm + MLA self-attention (residual)
  → up_proj 1024 → 4096               ← project to 64×64 sheet space
  → TopKToroidMoE on 64×64 sheet:
      • toroidal pad (circular wrap on both axes)
      • F.unfold → 64 patches of (patch_size × patch_size)
      • linear router (input 4096 → logits 64) → softmax → top-K
      • gather selected K patches and K expert weights per token
      • per-expert SwiGLU on patches
      • routing-weighted scatter into [N, 64, patch_dims]
      • F.fold → padded 64+2p × 64+2p output
      • toroidal wrap-add of edge bands + corners back into original 64×64
  → down_proj 4096 → 1024
  → residual
```

Per-cell `patch_size = 8 + overlap` (stride is fixed at 8, giving 8×8 = 64 experts on the 64×64 sheet). `expert_hidden` auto-derives from `patch_size` at a constant compression ratio (0.4, rounded to multiples of 8) so the overlap axis tests geometry independently of per-expert capacity.

The aux loss for load balancing follows the DeepSeek-MoE formulation: `L_aux = n_experts · Σ (frac_i · mean_prob_i)` with `coef = 0.01`.

### Cortical Column Emergence

Each expert position on the toroidal sheet, stacked across all R recurrent loop iterations, forms a cortical column. The biological analogy is exact:

| Neocortex | RTCC |
| --- | --- |
| Cortical sheet (2D) | 64×64 toroidal grid (inside the coda) |
| Minicolumn (vertical) | Expert position × R loops of the shared recurrent core |
| ~6 cortical layers | R = 6 loop iterations |
| Private intracolumn wiring | Private patch dims (never overlap with neighbors) |
| Lateral intercolumn connections | Shared border dims between neighboring patches |
| Local neighborhood only | Toroidal adjacency — neighbors-of-neighbors only |
| Columnar feedback | Hyper-connections at the loop boundary |

---

## Model Configuration

All paper runs use a single fixed backbone — locked to match CART d=1024 R=6 P=6 exactly so the RTCC-vs-CART comparison is direct.

| Parameter | Value | Source |
| --- | --- | --- |
| `model_dim` | 1024 | CART parity |
| Prelude layers (P) | 6 | CART Stage 2 winner |
| Recurrent loops (R) | 6 | CART Stage 2 winner (R*=6) |
| Coda layers | 1 | CART |
| `head_dim` | 64 | All blocks |
| `n_heads` (prelude / recurrent / coda) | 16 / 16 / 16 | 1024 / 64 = 16 |
| `mla_latent_dim` | 256 | d / 4 (MLA compression) |
| `vocab_size` | 32,000 | NousResearch/Llama-2-7b-hf — matches CART exactly |
| `max_seq_len` | 1,024 | CART Stage 2 |
| Coda up-projection | 1024 → 4096 | New for RTCC |
| Coda sheet | 64 × 64 toroidal | 4096 = 64² |
| Coda stride | 8 (fixed) | → 8×8 = 64 experts |
| Coda `patch_size` | 8 + overlap | Per-cell |
| Coda `expert_hidden` | auto-derived (~0.4 × patch_dims) | Per-cell |
| Coda `top_k` | Ablated: 2, 4, 8, 16 | Per-cell |
| Aux loss coef | 0.01 | DeepSeek-style load-balance |
| Total params | ~125M | At any cell |

### The 16-Cell K × Overlap Ablation

The single ablation axis pair: `K ∈ {2, 4, 8, 16} × overlap ∈ {1, 2, 3, 4}`. `expert_hidden` is per-cell:

| overlap | patch | patch_dims | expert_hidden | compression | coda FFN params |
| --- | --- | --- | --- | --- | --- |
| 1 | 9×9 | 81 | 32 | 0.40 | 8.88M |
| 2 | 10×10 | 100 | 40 | 0.40 | 9.16M |
| 3 | 11×11 | 121 | 48 | 0.40 | 9.50M |
| 4 | 12×12 | 144 | 56 | 0.39 | 9.93M |

CART's dense coda FFN reference at d=1024: **8.65M params**. The lightest cell is at parameter parity (+3%); the heaviest is +15% over CART. The total-param drift across the overlap axis is ~12%, footnoted in the paper.

---

## Relationship to CART (Paper 1)

RTCC is Paper 2 in a two-paper sequence. **CART** ([ccapps42/CART](https://github.com/ccapps42/CART), Capps 2026) establishes the recurrent backbone through original ablation experiments — both papers share the prelude / recurrent core / coda design, MLA attention, hyper-connections, LIE, LTI gating, RoPE-only-in-prelude-and-coda, RMSNorm, SwiGLU, tied embeddings, and Llama-2 32K tokenization.

The general direction of increasing effective depth through recurrence is supported by recent independent work. Oncescu et al. (arXiv 2604.21215, 2026) introduce a structurally distinct recurrent transformer in which each layer attends to KV pairs computed from its own outputs rather than the previous layer — a per-layer self-referential mechanism that differs from CART's loop-based weight sharing. Their empirical results show cross-entropy improvement over parameter-matched baselines at 150M–300M parameters on C4, independently corroborating that recurrent depth is a productive direction at scales comparable to CART's.

CART's recurrent core is architecturally distinct: the same block of layers is executed R times with fully shared weights, conditioned per loop by a sinusoidal loop-index embedding (LIE), stabilized by LTI injection, and blended across iterations by hyper-connections. These specific design and empirical findings are original to CART.

**Borrowed components assembled in CART (inherited unchanged by RTCC):**

* LTI injection for loop stability (Parcae, Prairie et al. 2026)
* Loop index embedding LIE (OpenMythos, kyegomez)
* Hyper-connections at loop boundaries (Hyperloop Transformer, arXiv 2604.21254)
* MLA attention (DeepSeek-V2, arXiv 2405.04434)
* RoPE, RMSNorm, SwiGLU, weight tying (standard components)

**Original findings from CART ablations:**

* **P=6 prelude / R=6 recurrent / 1 coda layer balance** at d=1024
* **Asymmetric MLA head counts** validated across scales
* **R=6 is the d=1024 Stage 2 winner** — RTCC inherits this without searching
* **Phased sequence-length and loop-count curriculum** critical for stable training

RTCC inherits all of this and contributes exactly one change: **the dense SwiGLU FFN in the coda layer is replaced by the Top-K Toroidal MoE**. This makes the paper's claim precisely falsifiable.

---

## Novelty

### Contribution 1: The CART Framework (established in Paper 1)

The recurrent backbone underlying RTCC was designed and validated through original ablation experiments in CART. CART's loop-based weight-sharing architecture is architecturally distinct from prior recurrent transformer work and the closest contemporary alternatives (Oncescu et al. 2026, Hyperloop, Parcae). CART's original contributions:

* Asymmetric prelude/recurrent/coda layer balance, validated as optimal across multiple scales
* R*=6 at d=1024, validated against R ∈ {6, 8, 10}
* Component assembly (LTI, LIE, hyper-connections, MLA) into a stable looped architecture
* Six diagnostic ablations at d=1024 (frozen-KV, R=1, unshared-core, self-attention, no-hyper, no-LTI) decomposing where CART's behavior comes from

These findings are CART's contribution. RTCC inherits the framework exactly, making the Paper 2 claim precise: *the Top-K Toroidal MoE coda is the only variable.*

### Contribution 2: The Top-K Toroidal Sheet MoE (this work)

To our knowledge, RTCC is the first architecture to combine all four of these:

1. **Topological structure on expert dimensional ownership** — experts own specific subspaces of an up-projected residual stream, not the full embedding
2. **Implicit inter-expert communication through overlapping dimensional ownership** — shared border dims receive gradient from multiple experts simultaneously, forcing agreement without explicit lateral connections
3. **Toroidal boundary condition** on the embedding grid — every expert position is structurally identical, eliminating edge effects; the wrap-add fold makes the geometry true to the toroid name
4. **Top-K routing across spatially-fixed experts** — sparse to dense behavior is itself an ablation axis (K = 2, 4, 8, 16)

### Closest Prior Work

* **SliceMoE** (Vejendla, 2024) — 1D dimensional partitioning without overlap, without topology, without position-fixed experts
* **MoGE** (Kang et al., 2025) — 2D structure applied to routing inputs, not to dimensional ownership
* **Mixtral-style top-K MoE** — routing-based MoE without spatial structure on the expert layout

---

## Experiments

All paper experiments use:

* **Hardware:** RTX 3090 24GB, single GPU
* **Model dim:** 1024 (matching CART exactly)
* **Tokenizer:** NousResearch/Llama-2-7b-hf (vocab 32,000)
* **Training data:** CART's `stage2_train.bin` (~1B tokens, 30/30/40 TinyStories/Wikipedia/FineWeb-Edu mix)
* **Validation:** CART's per-source val bins (tiny/wiki/edu, 50 batches per source — matches CART eval exactly)
* **All results logged to SQLite** at `db/rtcc_experiments.db` — see schema in `db/init_db.py`

### Full Run Plan (20 total)

| Group | Count | Notes |
| --- | --- | --- |
| K × overlap ablation cells | 16 | `configs/paper_1024_rtcc_coda/K{2,4,8,16}_O{1,2,3,4}.yaml` |
| Dense baseline (no new run) | 0 | CART d=1024 R=6 P=6 from `K:\projects\Model_Paper_1\results.db` (3 seeds + 6 diagnostic ablations already available) |
| B4 flat-grid baseline | 1 | Same geometry as the winning cell with `padding_mode: zeros` (isolates the toroidal-wrap contribution from the overlap-add fold) |
| Extra seeds at winner | 2 | seeds 137 and 271, for 3-seed stability evidence |
| lm-eval-harness benchmarks | 1 | Inference-only on the winning checkpoint (HellaSwag, ARC-Easy, ARC-Challenge, WinoGrande, PIQA, LAMBADA, WikiText-103) |
| **Total training runs** | **19** | Plus 1 inference-only eval |

B4 and the seed configs are generated post-winner via `scripts/gen_winner_configs.py`.

### Training Settings (matches CART d=1024 Stage 2 exactly)

| Setting | Value |
| --- | --- |
| `seq_len` | 1,024 |
| `batch_size` | 4 |
| `grad_accum_steps` | 8 |
| Effective batch | 32,768 tokens/step |
| `total_steps` | 30,500 |
| Total tokens | ~1B |
| `warmup_steps` | 2,000 |
| `lr_max` / `lr_min` | 3e-4 / 3e-5 (cosine) |
| Optimizer | AdamW8bit (bitsandbytes) |
| Precision | BF16 autocast (no GradScaler) |
| Eval frequency | every 2,500 steps |
| Checkpoint frequency | every 1,000 steps |

### Throughput and Wall-Clock

Measured 2026-05-24, RTX 3090:

| Cell | tok/sec | 1B-token run time |
| --- | --- | --- |
| K2_O1 (lightest) | 19,210 | 14.5 h |
| K16_O4 (heaviest) | 14,339 | 19.4 h |

Full 16-cell sweep at `--slots 1`: **~261 h ≈ 11 days**.

---

## Repository Structure

```
RTCC_Paper_2/
├── architectures/
│   ├── 01_dense_ffn/          ← legacy (not used in Paper 2)
│   ├── 02_standard_moe/       ← legacy
│   ├── 03_slicemoe_flat/      ← legacy
│   ├── 04_flat_grid_overlap/  ← legacy
│   ├── 05_rtcc/               ← legacy recurrent-core toroid, see its NOTE.md
│   └── 06_rtcc_coda_topk/     ← THE PAPER ARCHITECTURE
│       ├── config.py          — TopKToroidCodaConfig
│       ├── model.py           — TopKToroidCodaModel
│       ├── top_k_toroid.py    — TopKToroidMoE (the novel module)
│       └── trainer.py         — TopKToroidTrainer (handles aux loss)
├── configs/
│   ├── paper_1024_rtcc_coda/  ← the 16 ablation cells (K*_O*.yaml)
│   └── dev_1024_rtcc_coda/    ← smoke configs
├── shared/
│   ├── components/            ← MLA attention, RMSNorm, RoPE, LTI, LIE, hyper-connections
│   ├── training/              ← Trainer, optimizer, checkpoint, DB logger
│   ├── data/                  ← FixedOrderDataset, curriculum
│   └── eval/                  ← perplexity_bin (CART-compatible)
├── scripts/
│   ├── launch_run.py          — single-run launcher (used by orchestrator)
│   ├── orchestrate_paper.py   — auto-discovers configs/paper_1024_rtcc_coda/*.yaml
│   ├── gen_winner_configs.py  — emits B4 + 2-seed configs post-winner
│   └── resume_run.py          — resume from latest checkpoint
├── db/
│   ├── init_db.py             — SQLite schema
│   └── rtcc_experiments.db    — runs, steps, checkpoints, eval_results
├── CLAUDE.md                  ← architecture map + key constants for new sessions
├── README.md                  ← (this file)
├── RTCC_Paper_Roadmap.md      ← Paper 1 → Paper 2 → Paper 3 architectural progression
└── Notes for writing paper.md ← literature notes, prior-work analysis
```

The five `architectures/01_dense_ffn` through `architectures/05_rtcc` directories are legacy implementations from an earlier project phase. `05_rtcc` is the original recurrent-core toroid (no routing, all 64 experts active per loop iteration); it was deprecated when the project pivoted to coda placement with top-K routing. Code is preserved for reference. See `architectures/05_rtcc/NOTE.md`.

---

## Installation

```
git clone https://github.com/ccapps42/RTCC
cd RTCC
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, CUDA 11.8+. Optional: `bitsandbytes` for `AdamW8bit`.

---

## Reproducing Paper Results

> ⚠️ Paper experiments are in progress (sweep launched 2026-05-24, ~11 days wall-clock). This section will be updated at submission time with exact checkpoint links.

```powershell
# Verify the data path is correct (training reads CART's bins; see shared/data/loader.py)

# Launch a single ablation cell:
python scripts/launch_run.py --arch rtcc_coda_topk --config configs/paper_1024_rtcc_coda/K2_O1.yaml

# Or launch the whole 16-cell sweep with auto-skip on resume:
python scripts/orchestrate_paper.py

# Dry-run (print queue, no launch):
python scripts/orchestrate_paper.py --dry-run

# After the sweep completes and a winner is picked, generate post-winner configs:
python scripts/gen_winner_configs.py --cell K2_O1
# (emits B4_K2_O1.yaml + K2_O1_seed137.yaml + K2_O1_seed271.yaml)
```

Model weights will be available on HuggingFace Hub at submission time.

---

## Citation

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
* Dai et al., *DeepSeek-V2*, arXiv 2405.04434 (2024) — MLA attention; DeepSeek-MoE auxiliary load-balance loss
* Vejendla, *SliceMoE* (2024) — closest prior work on dimensional partitioning
* Mixtral (Jiang et al., 2024) — top-K routing reference
* Hawkins et al., *Thousand Brains Theory* (2019) — columnar cortical computation
* Gardner et al., *Science* (2022) — toroidal topology of grid cells in entorhinal cortex
* Dehghani et al., *Universal Transformer* (2018) — recurrent depth foundations

---

## License

MIT License — see [LICENSE](https://github.com/ccapps42/RTCC/blob/master/LICENSE)
