# RTCC Paper 2 — Codebase Guide

Recurrent Toroidal Cortical Columns (RTCC). A language-model architecture paper that swaps the recurrent FFN in CART (Paper 1, sibling project at `K:\projects\Model_Paper_1\`) for a **Toroidal Sheet MoE** — a 2D toroidal grid of position-fixed experts with overlapping patches. RTCC is the *only* variable vs the CART framework, so the paper's claim is precise: this one change.

## Architecture map

Five architectures share one trainer (`shared/training/trainer.py`) and one launch path. They differ only in the recurrent FFN component.

```
architectures/
  01_dense_ffn/          — Standard recurrent SwiGLU FFN (floor baseline, B1)
  02_standard_moe/       — Sparse MoE w/ top-k routing + aux loss (B2). Custom MoETrainer subclass.
  03_slicemoe_flat/      — Dimensional partitioning, no overlap (B3, closest prior art)
  04_flat_grid_overlap/  — RTCC geometry with zero-padding instead of circular wrap (B4)
  05_rtcc/               — Full RTCC: toroidal padding + overlapping experts (the paper)
```

Each architecture directory has `config.py` (subclass of `shared.config.BaseConfig`), `model.py`, and optionally a custom trainer. Launching loads them via `importlib` because the `NN_arch` numeric prefix prevents Python package imports — see `scripts/launch_run.py:ARCH_MAP`.

## Shared components

`shared/components/`: small, stable, mostly ported from CART. Don't modify without strong reason.
- `attention.py` — MLASelfAttention, MLACrossAttention, MLAKVProjection (fused k_up/v_up). Uses `F.scaled_dot_product_attention`.
- `rope.py` — RotaryEmbedding with seq_len cache.
- `norm.py` — RMSNorm via `F.rms_norm` (PyTorch ≥ 2.1) with weight dtype cast for fused kernel.
- `lti.py` — `h = sigmoid(a) * h_input + transformer_out`. `forward()` accepts optional precomputed `A` to hoist sigmoid out of recurrent loops.
- `lie.py` — Loop Index Embedding. `precompute_signals(n_loops)` for batched projection.
- `hyper.py` — Hyper-connections ring buffer (n=3 slots). No `.clone()` (LTI returns fresh tensors). `combine()` uses `torch.stack + einsum`.

`shared/training/`:
- `trainer.py` — base `Trainer` class. **bfloat16 autocast, no GradScaler** (fp16-only). TF32 enabled. Device-side loss accumulation (one sync per optimizer step). Prints every 50 steps including VRAM. Skips eval on Ctrl-C interrupt; second Ctrl-C force-exits.
- `optimizer.py` — `AdamW8bit` from bitsandbytes if available, else `torch.optim.AdamW`.
- `db_logger.py` — buffered SQLite writes via WAL. Flushes every 100 entries.
- `checkpoint.py` — save/load + prune-to-last-N.

`shared/data/`:
- `loader.py` — `FixedOrderDataset` reads CART's `stage2_train.bin` (uint16, Llama-2 32K tokenizer). No shuffling — same token order across all runs for sweep comparability.
- `validation.py` — `ValidationDataset` reads `data/validation/val.parquet`. **No longer used** in eval (we switched to CART val bins for parity).
- `curriculum.py` — phase/loop-count schedule.

`shared/eval/perplexity.py`:
- `evaluate_perplexity_bin(model, bin_path, ...)` — per-source PPL on CART val bins (tiny/wiki/edu). `max_batches=50` matches CART exactly.
- `evaluate_perplexity(model, parquet, ...)` — legacy, not used in trainer anymore.

## Configs

```
configs/
  paper_576/             — 11 paper runs at d=576 (B1-B4 + S1-S7)
  paper_1024/            — 1 scaling run at d=1024 (S7 geometry on 32x32 grid)
  dev_576/               — smoke tests, early signal runs, VRAM tests
  templates/             — base.yaml template
```

Config = YAML loaded by `BaseConfig.from_yaml()`. Each arch's `Config` adds its own fields with `__post_init__` assertions (e.g., `RTCCConfig` computes privacy% and asserts ≥3 experts per axis).

## How to launch runs

```powershell
# Single run
python scripts/launch_run.py --arch rtcc --config configs/paper_576/S7_rtcc_60pct.yaml

# Resume from latest checkpoint (same config, same run_name)
python scripts/resume_run.py --arch rtcc --config configs/paper_576/S7_rtcc_60pct.yaml

# Full paper sweep — see project memory for slot recommendation
python scripts/orchestrate_paper.py --slots 1

# VRAM measurement (100 steps, no eval)
python scripts/launch_run.py --arch rtcc --config configs/dev_576/vram_test_b16.yaml

# 4-concurrent VRAM stress test
python scripts/test_4_concurrent.py
```

**Interrupting a run:** Ctrl-C once → saves a checkpoint at the current step then exits (skips final eval). Ctrl-C twice → immediate force exit, no checkpoint. **Do not use `Stop-Process`** — it bypasses the SIGINT handler and you lose the checkpoint.

## Database

`db/rtcc_experiments.db` — SQLite, WAL mode. Schema in `db/init_db.py`. Three tables:
- `runs` — one row per run (config snapshot, status, timestamps)
- `steps` — per-optimizer-step training metrics (loss, grad_norm, lr, sec_per_step, tokens_seen)
- `checkpoints` — checkpoint paths, val_loss, val_perplexity, file_size_mb
- `eval_results` — flexible key/value rows for ppl per source, rho, vram, loop_delta, etc.

To query latest run state, see queries in conversation history — basic pattern:
```python
import sqlite3
conn = sqlite3.connect("db/rtcc_experiments.db")
conn.row_factory = sqlite3.Row
# latest run
conn.execute("SELECT * FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
```

## Logging output

Every 50 optimizer steps the trainer prints:
```
step  N/total (X%) | loss X.XXXX | lr X.XXe-04 | norm X.XX | rho 0.9001 | vram X.XX GB | XXXX tok/s | eta Xh YYm
```

Every `eval_every` steps:
```
[eval] step N | ppl_tiny=X ppl_wiki=X ppl_edu=X rho_max=X rho_mean=X vram=XGB
[loop_delta] r0:X r1:X ... r7:X     (RTCC only)
```

## Hardware

- **RTX 3090 (24 GB)** — all paper runs
- **RTX 3050 (8 GB)** — dev/debug only
- Single GPU, no multi-GPU code paths
- Windows 11. **No Triton.** No `torch.compile` (unreliable on Windows).

## Training data

External, not in this repo:
- Training: `K:\projects\Model_Paper_1\data\stage2\stage2_train.bin` — CART's pre-tokenized 1B token bin (NousResearch/Llama-2-7b-hf tokenizer, 32K vocab)
- Validation: `K:\projects\Model_Paper_1\data\val\{tinystories,wikipedia,fineweb_edu}_val.bin` — CART val bins, used for the three per-source PPLs (matches CART eval exactly)

Both are referenced by absolute path in `shared/config.py` defaults.

## Discipline: preserve CART parity

The paper's central comparison is RTCC (this project) vs CART at d=1024. The comparison is uniquely clean because **the only thing that differs is the coda FFN swap** — same training data (CART's tokenized bins, same order via FixedOrderDataset), same prelude, same recurrent core, same hyperparameters, same training schedule.

Every "while we're at it" tweak to the prelude, recurrent core, optimizer, scheduler, data pipeline, or curriculum weakens this comparison. If reviewers can attribute any RTCC delta to something other than the coda change, the contribution becomes ambiguous.

**Rule:** when implementing the new toroid-top-K-coda architecture, copy CART's config and code exactly; modify ONLY the coda layer. Any change to anything else needs explicit justification AND a matched-modification of CART's d=1024 reference run (or explicit acknowledgement in the paper).

This rule applies even when a small tweak looks obviously beneficial. The comparison's value comes from its precision — that's the methodological asset to protect.

## Key constants (do not casually change)

- **Tokenizer:** NousResearch/Llama-2-7b-hf (vocab 32,000) — pinned by training data; matches CART exactly
- **R (max_loop_iters):** TBD, see `memory/open_question_R_value.md`. CART Stage 2 sweeping R ∈ {6, 8, 10}; inherit winner before paper runs.
- **P (prelude_layers):** 6 — validated by CART
- **Coda layers:** 1 — validated by CART
- **seq_len:** 1024 — matches CART Stage 2; the stage2 bin was interleaved at 1024-token chunks
- **batch_size × grad_accum_steps:** 4 × 8 = 32,768 tokens/step

## Gotchas

- **The `NN_arch` package names are not importable.** The launcher loads them via `importlib.util.spec_from_file_location`. Don't try `from architectures.05_rtcc import model`.
- **GradScaler is removed.** Don't reintroduce it for bfloat16 paths — it's fp16-only and adds overhead.
- **No gradient checkpointing.** `torch.utils.checkpoint` in CART/RTCC recurrent loops aliases K/V/h_input and corrupts gradients with `use_reentrant=False`. See `memory/negative_grad_checkpoint.md`.
- **Print stdout is buffered when piped to a file.** When launching subprocesses, set `PYTHONUNBUFFERED=1` or use `python -u` if you need real-time logs. The orchestrator captures stdout to files and suffers from this.
- **PowerShell wildcard delete on `checkpoints/*` is blocked.** Delete individual `step_NNN` subdirs explicitly.
- **`Stop-Process` skips the SIGINT handler.** Use Ctrl-C to checkpoint cleanly.

## Cross-references

- **CART (Paper 1):** `K:\projects\Model_Paper_1\` — reference implementation for shared components, source of training data and val bins, source of validated hyperparameters (P=6, R, seq curriculum)
- **Project memory:** `C:\Users\ccapp\.claude\projects\K--projects-RTCC-Paper-2\memory\` — operational notes, fallback plans, throughput data, open questions
- **Sweep plan:** `RTCC_Sweep_Plan.md` — the authoritative experiment plan
- **Briefing:** `RTCC_BRIEFING.md` — architectural reference

## Performance notes (measured, May 2026)

- Solo throughput at d=576 batch=4: ~11,000 tok/s
- The RTX 3090 is **bandwidth-saturated by one RTCC run** at this batch size. Adding concurrent slots does not increase aggregate throughput. See `memory/project_sweep_orchestration.md` — full sweep is ~5.8 days at `--slots 1`.
- Bottleneck is `F.unfold`/`F.fold` in `ToroidalMoE`. Cannot be fused without Triton (Linux only).
