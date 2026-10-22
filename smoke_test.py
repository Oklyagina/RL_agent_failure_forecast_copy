"""Quick end-to-end smoke test: run_pipeline.py with a tiny configuration.

Runs every stage (ENN rollouts/training, forecasters, failure rows, classifier,
multi-line t+12 prediction) on 2 lines and a few steps, writing to
artifacts_smoke/ so real artifacts are untouched.  Results are not meaningful.

Run:  python smoke_test.py
"""
import os
import subprocess
import sys

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

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--force-stage", "all", "--smoke-line", lines]
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
        sys.argv = [sys.argv[0], "--force-stage", "classifier", "--smoke-line", lines]
        run_pipeline.main()
