"""Quick end-to-end smoke test of ``train`` with a tiny configuration.

Runs every stage (ENN rollouts/training, forecasters, failure rows, classifier,
multi-line t+12 prediction) on 2 lines and a few steps, writing to
artifacts_smoke/ so real artifacts are untouched.  Results are not meaningful.

    python main.py smoke                          # without LLM explanations
    python main.py smoke --llm-config PATH.json   # plus SHAP/LLM rules

Do not import ``project_config`` (directly or indirectly) at module level:
``run`` must set the overrides below before the configuration is read.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

SMOKE = {
    "ARTIFACTS_DIR": "artifacts_smoke",
    "ROLLOUT_EPISODES": "1",
    "ENN_ROLLOUT_MAX_STEPS": "150",
    "ENN_EPOCHS": "2",
    "ENN_ANNEAL_EPOCHS": "1",
    "FORECAST_EPISODES": "2",
    "FORECAST_MAX_STEPS": "150",
    "FORECAST_MEAN_TRIALS": "1",
    "FAILURE_EPISODES": "2",
    "FAILURE_MAX_STEPS": "500",
    "FAILURE_CLASSIFIER_TRIALS": "2",
}


def pipeline_arguments(
    lines: str, llm_config: Path | None, force_stage: str
) -> list[str]:
    """Build ``train`` arguments for the compact smoke run."""
    arguments = ["--force-stage", force_stage, "--smoke-line", lines]
    if llm_config is not None:
        arguments.extend([
            "--with-explainability",
            "--llm-config", str(llm_config),
            "--llm-iterations", "1",
        ])
    return arguments


def run(llm_config: Path | None = None) -> None:
    os.environ.update(SMOKE)

    import pandas as pd

    from project_config import FAILURE_LINES
    from src.workflows import train

    lines = ",".join(FAILURE_LINES.split(",")[:2])
    os.environ["FAILURE_LINES"] = lines
    train.FAILURE_LINES = lines
    parser = train.build_parser()

    try:
        train.run_pipeline(parser.parse_args(pipeline_arguments(lines, llm_config, "all")))
    except subprocess.CalledProcessError:
        rows_csv = train.failure_dir(train.AGENT_NAME) / "failure_forecast_rows.csv"
        rows = pd.read_csv(rows_csv) if rows_csv.is_file() else pd.DataFrame()
        if rows.empty or rows["failed"].nunique() == 2:
            raise
        # A tiny run rarely contains a real agent blackout, so the classifier
        # cannot be fitted. Add clearly marked synthetic positives (2 per
        # episode) so the classifier and inference stages can still be exercised.
        print("\n[smoke] no agent-blackout sample in the tiny run; adding "
              "synthetic positives (label_reason=smoke_synthetic)")
        synthetic = pd.concat([
            rows[rows["episode"] == episode].head(2)
            for episode in rows["episode"].unique()
        ]).assign(failed=1, label_reason="smoke_synthetic")
        pd.concat([rows, synthetic]).to_csv(rows_csv, index=False)
        train.run_pipeline(
            parser.parse_args(pipeline_arguments(lines, llm_config, "classifier"))
        )
