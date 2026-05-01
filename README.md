# Recurrent Toroidal Cortical Columns (RTCC)

**Paper:** *Recurrent Toroidal Cortical Columns: A Neocortically-Inspired Architecture for Efficient Language Modeling*  
**Author:** Chad Capps ([@ccapps42](https://github.com/ccapps42))  
**Status:** Experiments in progress — paper forthcoming  
**License:** MIT  

---

## Overview

RTCC is a language model architecture that replaces the feed-forward layer in a recurrent-depth transformer with a **Toroidal Sheet Mixture of Experts (TS-MoE)** — a spatially organized expert system in which:

- The model's embedding dimensions are reshaped into a **2D toroidal grid**
- Fixed-position experts process **overlapping patches** of that grid
- Shared border dimensions between adjacent experts create **implicit inter-expert communication** through gradient flow — no explicit lateral connections needed
- Recurrent depth stacks these toroidal sheets into **cortical columns**, one column per expert position, across all loop iterations

The result is an architecture with a precise structural analog to the neocortical minicolumn — the fundamental computational unit of the mammalian brain.

---

## Architecture

RTCC builds on the **CART** framework ([ccapps42/CART](https://github.com/ccapps42/CART)) and differs in exactly one way: the feed-forward layer in the recurrent core is replaced by the Toroidal Sheet MoE.

```
Input Tokens
    ↓
[Embedding]
    ↓
[Prelude]  — 4× MLA attention blocks, run once
    ↓        (sensory cortex analog)
    ↓
[Recurrent Core]  — looped R times
    ↓  ┌────────────────────────────────┐
    ↓  │  MLA Attention (12-head)       │
    ↓  │         ↓                      │
    ↓  │  Toroidal Sheet MoE            │  ← the novel component
    ↓  │    • reshape 768-dim → 32×24   │
    ↓  │    • toroidal circular padding │
    ↓  │    • 12 position-fixed experts │
    ↓  │    • overlap-add fold          │
    ↓  │         ↓                      │
    ↓  │  LTI injection                 │
    ↓  │  Hyper-connections             │
    ↓  └────────────────────────────────┘
    ↓
[Coda]  — 1× MLA attention block, run once
    ↓     (motor cortex analog)
    ↓
[RMSNorm → LM Head]
    ↓
Output Logits
```

### Toroidal Sheet MoE

The key innovation. At each recurrent loop iteration:

1. The 768-dim token embedding is reshaped into a **32×24 toroidal grid**
2. **Circular padding** wraps both axes — eliminating edge effects, making every expert position structurally identical
3. **N×N patches** slide over the grid with configurable stride and overlap
4. Each expert is **position-fixed** — no routing, no load-balancing loss, each expert always processes its assigned patch
5. Shared border dimensions between adjacent patches accumulate gradients from both experts during backprop — **implicit communication through architecture, not explicit lateral connections**
6. The output is folded back via **overlap-add** to the full 32×24 grid

### Cortical Column Emergence

Each expert position on the toroidal grid, stacked across all R recurrent loop iterations, forms a **cortical column**:

| Neocortex | RTCC |
|---|---|
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
vocab_size:      50,000

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
|--------|-------|--------|---------|---------|-------------|---------|
| sweep_11 | 3×3 | 2 | 1-cell | 192 | 1 | 11% |
| sweep_36 | 5×5 | 4 | 1-cell | 48 | 9 | 36% |
| sweep_44 | 6×6 | 4 | 2-cell | 48 | 4 | 11%* |
| sweep_60 | 9×9 | 8 | 1-cell | 12 | 49 | 60% |
| **sweep_64** | **10×10** | **8** | **2-cell** | **12** | **36** | **36%*** |

\* Controlled pairs at the same expert count isolate the effect of overlap width independently of expert count.

---

## Relationship to CART (Paper 1)

RTCC is Paper 2 in a two-paper sequence. **CART** ([ccapps42/CART](https://github.com/ccapps42/CART), Capps 2026) establishes the recurrent framework through original ablation experiments:

**Borrowed components assembled in CART:**
- LTI injection for loop stability (formulation from OpenMythos / Claude Mythos)
- Hyper-connections at loop boundaries (from Hyperloop, arXiv 2604.21254)
- MLA attention (from DeepSeek-V2, arXiv 2405.04434)
- RoPE, RMSNorm, SwiGLU, weight tying (standard components)

**Original findings from CART ablations:**
- **4+6+1 prelude/recurrent/coda balance** — empirically optimal at both dim=256 and dim=768; all prior RDT work uses symmetric configs
- **Asymmetric MLA head counts** — 16-head prelude, 12-head recurrent core, 8-head coda
- **R=8 loop ceiling** — validated as the optimal training ceiling
- **Phased sequence-length and loop-count curriculum** — critical for stable training, validated across four phases

RTCC inherits all of the above from CART without modification and contributes exactly one change: the feed-forward layer in the recurrent core is replaced by the Toroidal Sheet MoE. This makes the paper's claim precise and fully controlled.

---

## Novelty

RTCC makes two distinct original contributions:

### Contribution 1: The CART Framework (established in Paper 1)

The recurrent framework underlying RTCC was designed and validated through original ablation experiments in CART ([ccapps42/CART](https://github.com/ccapps42/CART), Capps 2026). Specific original findings include:

- **4+6+1 layer balance** — asymmetric prelude/recurrent/coda configuration, empirically optimal across dim=256 and dim=768; all prior published RDT work uses symmetric configs
- **Asymmetric MLA head counts** — 16-head prelude, 12-head recurrent core, 8-head coda; heavier prelude conditions the signal, lighter coda is sufficient for output projection
- **R=8 loop ceiling** — validated as optimal training ceiling; lower ceilings constrain low-loop performance, higher ceilings degrade it
- **Phased curriculum** — sequence length and loop count ramped jointly across four phases; independently validated as critical for stable training
- **Component assembly** — LTI injection (from OpenMythos), hyper-connections (from Hyperloop), MLA (from DeepSeek-V2), RoPE, RMSNorm, SwiGLU assembled into a coherent architecture and validated at scale

These findings are the contribution of CART. RTCC inherits this framework exactly, with no modifications, making the paper's claim precise: *the ToroidalMoE is the only variable.*

### Contribution 2: The Toroidal Sheet MoE (this work)

To our knowledge, RTCC is the first architecture to:

1. Impose a **topological structure on expert dimensional ownership** — experts own specific subspaces of the embedding, not the full embedding
2. Create **implicit inter-expert communication through overlapping dimensional ownership** — shared border dims receive gradient from multiple experts simultaneously, forcing agreement without explicit lateral connections
3. Use a **toroidal boundary condition** on the embedding grid — eliminating edge effects and making every expert position structurally identical
4. Achieve **cortical column structure as an emergent property** of combining 2D toroidal expert layout with recurrent depth — no biological structure is explicitly programmed

### Closest Prior Work

- **SliceMoE** (Vejendla, 2024) — dimensional partitioning without overlap, without topology, without position-fixed experts
- **MoGE** (Kang et al., 2025) — 2D structure applied to routing inputs, not dimensional ownership

---

## Experiments

All paper experiments use:
- **Hardware:** RTX 3090 24GB
- **Model dim:** 768
- **Training data:** TinyStories, Wikipedia, FineWeb-Edu, FineWeb (via HuggingFace datasets)
- **Evaluation:** Held-out validation set + lm-eval-harness benchmarks
- **All results logged to SQLite** — exported to `results/` for reproducibility

### Run Plan

| Tier | Runs | Purpose |
|------|------|---------|
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
  components/            — MLA attention, RMSNorm, RoPE, LTI injection
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

```bash
git clone https://github.com/ccapps42/RTCC
cd RTCC
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, CUDA 11.8+

---

## Reproducing Paper Results

> ⚠️ **Note:** Paper experiments are still in progress. This section will be completed at submission time with exact commands, config paths, and checkpoint links.

```bash
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
  title   = {Context-Anchored Recurrent Transformer},
  author  = {Capps, Chad},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/ccapps42/CART}
}
```

---

## Key References

- Geiping et al., *Huginn* (2025) — recurrent depth at scale
- Hyperloop Transformers, arXiv 2604.21254 (2026) — hyper-connections
- Oncescu et al., arXiv 2604.21215 (2026) — recurrent transformer theory
- Gomez, *OpenMythos* (2026) — LTI injection formulation
- Dai et al., *DeepSeek-V2*, arXiv 2405.04434 (2024) — MLA attention
- Vejendla, *SliceMoE* (2024) — closest prior work on dimensional partitioning
- Hawkins et al., *Thousand Brains Theory* (2019) — columnar cortical computation
- Gardner et al., *Science* (2022) — toroidal topology of grid cells in entorhinal cortex
- Dehghani et al., *Universal Transformer* (2018) — recurrent depth foundations

---

## License

MIT License — see [LICENSE](LICENSE)
