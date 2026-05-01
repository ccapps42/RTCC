"""Buffered SQLite logger. Writes every step; flushes every 100."""
import sqlite3


class DBLogger:
    def __init__(self, db_path: str, run_id: int):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.run_id = run_id
        self.buffer: list = []
        self.flush_every = 100

    def log_step(self, step: int, loss: float, aux_loss: float | None,
                 grad_norm: float, lr: float, n_loops: int | None,
                 seq_len: int, tokens_seen: int, phase: int, sec_per_step: float | None):
        self.buffer.append((
            self.run_id, step, loss, aux_loss, grad_norm,
            lr, n_loops, seq_len, tokens_seen, phase, sec_per_step,
        ))
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        self.conn.executemany(
            """INSERT OR IGNORE INTO steps
               (run_id, step, loss, aux_loss, grad_norm, lr, n_loops,
                seq_len, tokens_seen, phase, sec_per_step)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            self.buffer,
        )
        self.conn.commit()
        self.buffer.clear()

    def log_checkpoint(self, step: int, path: str, val_loss: float | None,
                       val_perplexity: float | None, file_size_mb: float | None):
        self.conn.execute(
            """INSERT OR REPLACE INTO checkpoints
               (run_id, step, path, val_loss, val_perplexity, file_size_mb)
               VALUES (?,?,?,?,?,?)""",
            (self.run_id, step, path, val_loss, val_perplexity, file_size_mb),
        )
        self.conn.commit()

    def log_eval(self, checkpoint_step: int, eval_type: str, metric: str, value: float,
                 hardware: str, benchmark: str | None = None, inference_loops: int | None = None,
                 modifier: str | None = None, notes: str | None = None):
        self.conn.execute(
            """INSERT INTO eval_results
               (run_id, checkpoint_step, eval_type, benchmark, metric, value,
                inference_loops, modifier, hardware, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (self.run_id, checkpoint_step, eval_type, benchmark, metric, value,
             inference_loops, modifier, hardware, notes),
        )
        self.conn.commit()

    def set_status(self, status: str, completed_at: str | None = None):
        if completed_at:
            self.conn.execute(
                "UPDATE runs SET status=?, completed_at=? WHERE run_id=?",
                (status, completed_at, self.run_id),
            )
        else:
            self.conn.execute(
                "UPDATE runs SET status=? WHERE run_id=?",
                (status, self.run_id),
            )
        self.conn.commit()

    def close(self):
        self.flush()
        self.conn.close()
