"""Quick end-to-end smoke test: run_pipeline.py with a tiny configuration.

Runs every stage (ENN rollouts/training, forecasters, failure rows, classifier,
multi-line t+12 prediction) on 2 lines and a few steps, writing to
artifacts_smoke/ so real artifacts are untouched.  Results are not meaningful.

Run without LLM explanations:
    python smoke_test.py

Run every stage including LLM rule generation and explained inference:
    python smoke_test.py --llm-config path/to/llm_config.json
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Must be set before project_config is imported (it reads the environment).
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
os.environ.update(SMOKE)

import run_pipeline  # noqa: E402
from project_config import FAILURE_LINES  # noqa: E402

lines = ",".join(FAILURE_LINES.split(",")[:2])
os.environ["FAILURE_LINES"] = lines
run_pipeline.FAILURE_LINES = lines

def pipeline_arguments(llm_config: Path | None, force_stage: str) -> list[str]:
    """Build root-pipeline arguments for the compact smoke run."""
    arguments = [
        sys.argv[0],
        "--force-stage", force_stage,
        "--smoke-line", lines,
    ]
    if llm_config is not None:
        arguments.extend([
            "--with-explainability",
            "--llm-config", str(llm_config),
            "--llm-iterations", "1",
        ])
    return arguments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=None,
        help="include SHAP ranking, LLM rule generation, and explained inference",
    )
    smoke_args = parser.parse_args()

    sys.argv = pipeline_arguments(smoke_args.llm_config, "all")
    try:
        run_pipeline.main()
    except subprocess.CalledProcessError:
        rows_csv = run_pipeline.failure_dir(run_pipeline.AGENT_NAME) / "failure_forecast_rows.csv"
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
        sys.argv = pipeline_arguments(smoke_args.llm_config, "classifier")
        run_pipeline.main()
