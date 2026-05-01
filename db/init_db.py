"""One-time script to initialize rtcc_experiments.db with full schema."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "rtcc_experiments.db"


def init():
    if DB_PATH.exists():
        print(f"DB already exists at {DB_PATH}. Skipping.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE runs (
            run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name        TEXT NOT NULL UNIQUE,
            architecture    TEXT NOT NULL,
            tier            TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'queued',
            hardware        TEXT NOT NULL,
            model_dim       INTEGER NOT NULL,
            grid_rows       INTEGER,
            grid_cols       INTEGER,
            patch_size      INTEGER,
            stride          INTEGER,
            n_experts       INTEGER,
            private_dims    INTEGER,
            privacy_pct     REAL,
            r_loops         INTEGER,
            n_prelude       INTEGER,
            n_recurrent     INTEGER,
            n_coda          INTEGER,
            vocab_size      INTEGER NOT NULL DEFAULT 50000,
            max_seq_len     INTEGER NOT NULL,
            total_steps     INTEGER NOT NULL,
            batch_size      INTEGER NOT NULL,
            grad_accum      INTEGER NOT NULL,
            lr_max          REAL NOT NULL,
            lr_min          REAL NOT NULL,
            seed            INTEGER NOT NULL DEFAULT 42,
            tokens_target   INTEGER NOT NULL,
            tokens_trained  INTEGER DEFAULT 0,
            config_path     TEXT NOT NULL,
            checkpoint_dir  TEXT NOT NULL,
            notes           TEXT,
            started_at      TEXT,
            completed_at    TEXT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE steps (
            step_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER NOT NULL REFERENCES runs(run_id),
            step            INTEGER NOT NULL,
            loss            REAL NOT NULL,
            aux_loss        REAL,
            grad_norm       REAL NOT NULL,
            lr              REAL NOT NULL,
            n_loops         INTEGER,
            seq_len         INTEGER NOT NULL,
            tokens_seen     INTEGER NOT NULL,
            phase           INTEGER NOT NULL,
            sec_per_step    REAL,
            UNIQUE(run_id, step)
        );
        CREATE INDEX idx_steps_run_step ON steps(run_id, step);
        CREATE INDEX idx_steps_run_tokens ON steps(run_id, tokens_seen);

        CREATE TABLE checkpoints (
            checkpoint_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER NOT NULL REFERENCES runs(run_id),
            step            INTEGER NOT NULL,
            path            TEXT NOT NULL,
            val_loss        REAL,
            val_perplexity  REAL,
            file_size_mb    REAL,
            saved_at        TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(run_id, step)
        );

        CREATE TABLE eval_results (
            eval_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER NOT NULL REFERENCES runs(run_id),
            checkpoint_step INTEGER NOT NULL,
            eval_type       TEXT NOT NULL,
            benchmark       TEXT,
            metric          TEXT NOT NULL,
            value           REAL NOT NULL,
            inference_loops INTEGER,
            modifier        TEXT,
            hardware        TEXT NOT NULL,
            eval_at         TEXT NOT NULL DEFAULT (datetime('now')),
            notes           TEXT
        );
        CREATE INDEX idx_eval_run ON eval_results(run_id, eval_type);

        CREATE TABLE expert_analysis (
            analysis_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id               INTEGER NOT NULL REFERENCES runs(run_id),
            checkpoint_step      INTEGER NOT NULL,
            expert_row           INTEGER NOT NULL,
            expert_col           INTEGER NOT NULL,
            activation_entropy   REAL,
            mean_activation      REAL,
            activation_frequency REAL,
            neighbor_similarity  REAL,
            private_dim_variance REAL,
            shared_dim_variance  REAL,
            analyzed_at          TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX idx_expert_run ON expert_analysis(run_id, checkpoint_step);

        CREATE TABLE hardware_profile (
            profile_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER NOT NULL REFERENCES runs(run_id),
            phase           INTEGER NOT NULL,
            seq_len         INTEGER NOT NULL,
            avg_loops       REAL NOT NULL,
            sec_per_step    REAL NOT NULL,
            tokens_per_sec  REAL NOT NULL,
            vram_peak_gb    REAL,
            measured_at     TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE VIEW best_checkpoints AS
        SELECT c.*, r.architecture, r.tier, r.hardware, r.model_dim
        FROM checkpoints c
        JOIN runs r ON c.run_id = r.run_id
        WHERE c.val_perplexity = (
            SELECT MIN(val_perplexity) FROM checkpoints c2
            WHERE c2.run_id = c.run_id AND c2.val_perplexity IS NOT NULL
        );

        CREATE VIEW loss_windows AS
        SELECT
            run_id,
            (step / 1000) * 1000 AS window_start,
            COUNT(*) AS n_steps,
            AVG(loss) AS avg_loss,
            MIN(loss) AS min_loss,
            AVG(tokens_seen) AS avg_tokens
        FROM steps
        GROUP BY run_id, window_start;

        CREATE VIEW privacy_sweep_summary AS
        SELECT
            r.run_name,
            r.patch_size,
            r.stride,
            r.n_experts,
            r.private_dims,
            r.privacy_pct,
            r.model_dim,
            bc.val_perplexity AS best_perplexity,
            bc.val_loss AS best_val_loss,
            bc.step AS best_step
        FROM runs r
        JOIN best_checkpoints bc ON r.run_id = bc.run_id
        WHERE r.tier = 'privacy_sweep'
        ORDER BY r.privacy_pct;
    """)
    conn.commit()
    conn.close()
    print(f"Database initialized: {DB_PATH}")


if __name__ == "__main__":
    init()
