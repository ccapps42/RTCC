"""lm-eval-harness wrapper. Only used for Tier 6 benchmark run."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


BENCHMARKS = [
    "wikitext",
    "lambada_openai",
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
]


def run_lm_eval(model_wrapper, device: "torch.device", checkpoint_step: int,
                db_logger, hardware: str):
    """Run lm-eval on all BENCHMARKS and write results to DB."""
    import lm_eval

    results = lm_eval.simple_evaluate(
        model=model_wrapper,
        tasks=BENCHMARKS,
        num_fewshot=0,
        device=str(device),
    )

    for task, task_results in results["results"].items():
        for metric, value in task_results.items():
            if metric.endswith("_stderr") or not isinstance(value, (int, float)):
                continue
            db_logger.log_eval(
                checkpoint_step=checkpoint_step,
                eval_type="lm_eval",
                metric=metric,
                value=float(value),
                hardware=hardware,
                benchmark=task,
            )
