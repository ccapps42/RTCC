"""Base Trainer — tracks optimizer steps throughout. total_steps = optimizer steps."""
import math
import signal
import sys
import time
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from shared.config import BaseConfig
from shared.training.db_logger import DBLogger
from shared.training.checkpoint import save_checkpoint, load_checkpoint, get_checkpoint_size_mb
from shared.training.optimizer import build_optimizer, build_scheduler
from shared.data.curriculum import get_phase, get_loop_count
from shared.eval.perplexity import evaluate_perplexity, evaluate_perplexity_bin


class Trainer:
    def __init__(self, model: torch.nn.Module, config: BaseConfig, run_id: int):
        self.model = model
        self.cfg = config
        self.run_id = run_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.optimizer = build_optimizer(model, config.lr_max, config.weight_decay)
        self.scheduler = build_scheduler(
            self.optimizer, config.warmup_steps, config.total_steps,
            config.lr_max, config.lr_min,
        )
        self.logger = DBLogger(config.db_path, run_id)
        # No GradScaler — we use bfloat16 (fp32 dynamic range), scaler is fp16-only

        self._opt_step = 0       # optimizer steps (primary counter)
        self._tokens_seen = 0
        self._interrupted = False
        self._diag_batch = None  # one stored batch for diagnostic forward passes
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, *_):
        if self._interrupted:
            print("\nForce exit.")
            sys.exit(1)
        print("\nInterrupt received — will checkpoint at next opportunity (Ctrl-C again to force quit).")
        self._interrupted = True

    def train(self, data_iter):
        self.model.to(self.device)
        self.model.train()

        # Resume from checkpoint — returns optimizer step count
        self._opt_step = load_checkpoint(self.cfg.checkpoint_dir, self.model,
                                         self.optimizer, self.scheduler)
        self._resume_step = self._opt_step  # ETA is relative to this session's start
        # Fast-forward scheduler to match resumed step
        for _ in range(self._opt_step):
            self.scheduler.step()

        self.logger.set_status("running")
        print(f"REMINDER: Disable Windows Update auto-restart before a long run.")
        print(f"Settings > Windows Update > Advanced > Pause updates for 5 weeks")
        print(f"Starting from step {self._opt_step}, target {self.cfg.total_steps}")

        # Device-side accumulators — defer GPU->CPU sync to once per optimizer step
        accum_loss_t = torch.zeros((), device=self.device)
        accum_aux_t  = torch.zeros((), device=self.device)
        accum_count = 0
        step_start = time.perf_counter()
        run_start = time.perf_counter()
        n_loops = self.cfg.max_loop_iters

        data_loader = data_iter
        data_iterator = iter(data_loader)

        while True:
            if self._opt_step >= self.cfg.total_steps or self._interrupted:
                break
            try:
                x, y = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                x, y = next(data_iterator)

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if self._diag_batch is None:
                self._diag_batch = (x[:1].clone(), y[:1].clone())
            phase = get_phase(self._opt_step)
            n_loops = get_loop_count(self._opt_step)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, aux_loss = self._forward(x, y, n_loops)
                total = loss if aux_loss is None else loss + aux_loss
                (total / self.cfg.grad_accum_steps).backward()

            # Accumulate on device — no GPU->CPU sync until optimizer step
            accum_loss_t += loss.detach()
            if aux_loss is not None:
                accum_aux_t += aux_loss.detach()
            accum_count += 1

            if accum_count == self.cfg.grad_accum_steps:
                # Queue up all GPU work first, then sync once at the end
                grad_norm_t = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                self._opt_step += 1
                sec = time.perf_counter() - step_start
                lr = self.scheduler.get_last_lr()[0]
                tokens_this_step = x.numel() * self.cfg.grad_accum_steps
                self._tokens_seen += tokens_this_step

                # Single sync point — pulls loss, aux, grad_norm in one barrier
                loss_val = accum_loss_t.item() / accum_count
                aux_val  = accum_aux_t.item()  / accum_count
                grad_norm = grad_norm_t.item()
                accum_loss_t.zero_()
                accum_aux_t.zero_()

                self.logger.log_step(
                    step=self._opt_step,
                    loss=loss_val,
                    aux_loss=aux_val or None,
                    grad_norm=grad_norm,
                    lr=lr,
                    n_loops=n_loops,
                    seq_len=x.shape[1],
                    tokens_seen=self._tokens_seen,
                    phase=phase.phase_num,
                    sec_per_step=sec,
                )

                if self._opt_step % 50 == 0:
                    steps_done = self._opt_step - self._resume_step
                    steps_left = self.cfg.total_steps - self._opt_step
                    elapsed = time.perf_counter() - run_start
                    secs_left = (elapsed / steps_done) * steps_left if steps_done > 0 else 0
                    h, rem = divmod(int(secs_left), 3600)
                    eta = f"{h}h {rem//60:02d}m"
                    pct = 100 * self._opt_step / self.cfg.total_steps
                    rho_str = ""
                    if hasattr(self.model, 'lti'):
                        with torch.no_grad():
                            rho_str = f"| rho {torch.sigmoid(self.model.lti.a_param).max().item():.4f} "
                    print(f"step {self._opt_step:6d}/{self.cfg.total_steps} ({pct:5.1f}%) "
                          f"| loss {loss_val:.4f} "
                          f"| lr {lr:.2e} | norm {grad_norm:.2f} "
                          f"{rho_str}| {tokens_this_step/sec:.0f} tok/s | eta {eta}")

                accum_count = 0
                step_start = time.perf_counter()

                if self._opt_step % self.cfg.checkpoint_every == 0:
                    self._do_checkpoint(self._opt_step)

                if self._opt_step % self.cfg.eval_every == 0:
                    self._do_eval(self._opt_step)

        # Final checkpoint; skip eval on interrupt to exit promptly
        self._do_checkpoint(self._opt_step)
        if not self._interrupted:
            self._do_eval(self._opt_step)
        self.logger.flush()

        status = "interrupted" if self._interrupted else "complete"
        now = datetime.now(timezone.utc).isoformat()
        self.logger.set_status(status, completed_at=now)

        conn = sqlite3.connect(self.cfg.db_path)
        conn.execute("UPDATE runs SET tokens_trained=? WHERE run_id=?",
                     (self._tokens_seen, self.run_id))
        conn.commit()
        conn.close()
        self.logger.close()

    def _forward(self, x, y, n_loops):
        """Override in subclasses that need aux_loss (e.g. MoE)."""
        logits = self.model(x, n_loops=n_loops)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        return loss, None

    def _do_checkpoint(self, step: int):
        ckpt_dir = save_checkpoint(
            step, self.model, self.optimizer, self.scheduler,
            self.cfg.checkpoint_dir, self.cfg.keep_last_n_checkpoints,
        )
        size_mb = get_checkpoint_size_mb(ckpt_dir)
        self.logger.log_checkpoint(step, str(ckpt_dir), None, None, size_mb)

    def _do_eval(self, step: int):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Eval matches CART exactly — tiny/wiki/edu bin files at seq_len=1024,
        # 50-batch cap per source.
        wiki_loss, ppl_wiki = evaluate_perplexity_bin(
            self.model, self.cfg.val_wiki_bin, self.device, seq_len=self.cfg.max_seq_len)
        _, ppl_tiny = evaluate_perplexity_bin(
            self.model, self.cfg.val_tiny_bin, self.device, seq_len=self.cfg.max_seq_len)
        _, ppl_edu  = evaluate_perplexity_bin(
            self.model, self.cfg.val_edu_bin,  self.device, seq_len=self.cfg.max_seq_len)

        # Rho (LTI spectral radius)
        rho_max, rho_mean = 0.0, 0.0
        if hasattr(self.model, 'lti'):
            with torch.no_grad():
                a = torch.sigmoid(self.model.lti.a_param)
                rho_max = a.max().item()
                rho_mean = a.mean().item()

        # Loop delta — one diagnostic forward pass on a stored batch
        loop_deltas = []
        if self._diag_batch is not None:
            dx, _ = self._diag_batch
            dx = dx.to(self.device)
            self.model.eval()
            try:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, diag = self.model(dx, n_loops=self.cfg.max_loop_iters,
                                        return_diagnostics=True)
                loop_deltas = diag["loop_deltas"]
            except TypeError:
                pass  # model doesn't support return_diagnostics
            self.model.train()

        peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9
                        if torch.cuda.is_available() else 0.0)

        # DB — wiki as canonical val_loss/val_ppl in checkpoints table (matches
        # convention of wiki as the standard validation reference in this family)
        self.logger.log_checkpoint(step, str(Path(self.cfg.checkpoint_dir) / f"step_{step:08d}"),
                                   wiki_loss, ppl_wiki, None)
        self.logger.log_eval(step, "perplexity", "ppl_tiny", ppl_tiny, self.cfg.hardware)
        self.logger.log_eval(step, "perplexity", "ppl_wiki", ppl_wiki, self.cfg.hardware)
        self.logger.log_eval(step, "perplexity", "ppl_edu",  ppl_edu,  self.cfg.hardware)
        if hasattr(self.model, 'lti'):
            self.logger.log_eval(step, "rho", "rho_max", rho_max, self.cfg.hardware)
            self.logger.log_eval(step, "rho", "rho_mean", rho_mean, self.cfg.hardware)
        self.logger.log_eval(step, "vram", "peak_vram_gb", peak_vram_gb, self.cfg.hardware)
        for r, delta in enumerate(loop_deltas):
            self.logger.log_eval(step, "loop_delta", f"loop_{r}", delta,
                                 self.cfg.hardware, inference_loops=r)

        # Console — matches CART format
        rho_str = f"  rho_max={rho_max:.4f}  rho_mean={rho_mean:.4f}" if hasattr(self.model, 'lti') else ""
        print(f"  [eval] step {step} | "
              f"ppl_tiny={ppl_tiny:.2f}  ppl_wiki={ppl_wiki:.2f}  ppl_edu={ppl_edu:.2f}"
              f"{rho_str}  vram={peak_vram_gb:.2f}GB")
        if loop_deltas:
            delta_str = "  ".join(f"r{r}:{d:.3f}" for r, d in enumerate(loop_deltas))
            print(f"  [loop_delta] {delta_str}")
