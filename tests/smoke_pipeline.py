"""Run the complete pipeline with deliberately tiny training datasets."""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_config import AGENT_NAME, ENV_DIR  # noqa: E402


REQUIRED_SCENARIO_FILES = ("config.py", "grid.json")
REQUIRED_CHRONIC_SERIES = ("load_p", "load_q", "prod_p")
SMOKE_LINES = "32_36_112,34_35_110"
PREDICTION_LINE = "34_35_110"
SMOKE_ARTIFACTS = ROOT / "artifacts" / "pipeline_smoke"
SMOKE_SETTINGS = {
    "ROLLOUT_EPISODES": "2",
    "ENN_ROLLOUT_MAX_STEPS": "200",
    "ENN_EPOCHS": "1",
    "ENN_ANNEAL_EPOCHS": "1",
    "ENN_BATCH_SIZE": "64",
    "ENN_HIDDEN_DIM": "16",
    "ENN_DROPOUT": "0",
    "FORECAST_EPISODES": "1",
    "FORECAST_MAX_STEPS": "30",
    "FORECAST_MEAN_TRIALS": "0",
    "FAILURE_EPISODES": "1",
    "FAILURE_MAX_STEPS": "30",
    "FAILURE_SAMPLING_STRIDE": "5",
    "FAILURE_LINES": SMOKE_LINES,
}


def _has_series(chronic_dir: Path, series: str) -> bool:
    return any(
        (chronic_dir / f"{series}{suffix}").is_file()
        for suffix in (".csv", ".csv.bz2")
    )


def _has_usable_chronic(path: Path) -> bool:
    chronics_dir = path / "chronics"
    if not chronics_dir.is_dir():
        return False
    return any(
        chronic.is_dir()
        and all(_has_series(chronic, series) for series in REQUIRED_CHRONIC_SERIES)
        for chronic in chronics_dir.iterdir()
    )


def _scenario_problems(path: Path) -> list[str]:
    problems = [
        f"missing {name}"
        for name in REQUIRED_SCENARIO_FILES
        if not (path / name).is_file()
    ]
    if not _has_usable_chronic(path):
        problems.append("no usable chronic directly under chronics/")
    return problems


def resolve_scenario() -> Path:
    """Find a complete ai4realnet_small scenario without changing .env."""
    candidates = []
    override = os.environ.get("SMOKE_ENV_DIR")
    if override:
        path = Path(override)
        candidates.append(path if path.is_absolute() else ROOT / path)
    candidates.extend([
        ENV_DIR,
        ROOT / "artifacts" / "pipeline_smoke_scenario" / "ai4realnet_small",
    ])

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        problems = _scenario_problems(candidate)
        if not problems:
            return candidate
        checked.append(f"{candidate} ({'; '.join(problems)})")

    raise FileNotFoundError(
        "No complete smoke-test Grid2Op scenario was found. Checked:\n  "
        + "\n  ".join(checked)
        + "\nSet SMOKE_ENV_DIR to a complete ai4realnet_small directory."
    )


def run_pipeline(scenario: Path) -> str:
    env = os.environ.copy()
    env.update(SMOKE_SETTINGS)
    env.update({
        "ENV_NAME": scenario.name,
        "ENV_LOCATION": str(scenario.parent),
        "ARTIFACTS_DIR": str(SMOKE_ARTIFACTS),
    })
    command = [
        sys.executable,
        str(ROOT / "run_pipeline.py"),
        "--agent-name", AGENT_NAME,
        "--force-stage", "all",
        "--smoke-line", PREDICTION_LINE,
    ]

    print("Small pipeline smoke test", flush=True)
    print(f"  scenario  : {scenario}", flush=True)
    print(f"  artifacts : {SMOKE_ARTIFACTS}", flush=True)
    print(f"  lines     : {SMOKE_LINES}", flush=True)

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    output_lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return "".join(output_lines)


def validate_outputs(scenario: Path, pipeline_output: str) -> None:
    agent_root = SMOKE_ARTIFACTS / scenario.name / AGENT_NAME
    rollouts = agent_root / "rollouts"
    models = agent_root / "model"
    failures = agent_root / "failure_forecast"
    required = [
        rollouts / "observations.npy",
        rollouts / "labels.npy",
        rollouts / "actions.npy",
        models / f"enn_{AGENT_NAME}.pth",
        models / "scaler_params.json",
        models / "enn_meta.json",
        models / "enn_pctile_calib.npz",
        failures / "mean_forecaster.pkl",
        failures / "aleatoric_forecaster.pkl",
        failures / "failure_forecast_rows.csv",
        failures / "failure_classifier.pkl",
        failures / "failure_classifier_metadata.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AssertionError("Smoke test outputs are missing:\n  " + "\n  ".join(missing))

    observations = np.load(rollouts / "observations.npy", allow_pickle=False)
    labels = np.load(rollouts / "labels.npy", allow_pickle=False)
    actions = np.load(rollouts / "actions.npy", allow_pickle=False)
    if len(observations) == 0 or len(observations) != len(labels):
        raise AssertionError("Rollout observations and labels are empty or misaligned.")
    if len(actions) < 2:
        raise AssertionError("Smoke rollouts must contain at least two distinct actions.")

    with (failures / "failure_forecast_rows.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    failure_labels = {int(row["failed"]) for row in rows}
    if failure_labels != {0, 1}:
        raise AssertionError(
            f"Smoke failure rows must contain labels 0 and 1; found {failure_labels}."
        )
    if "'failure_prediction':" not in pipeline_output:
        raise AssertionError("The final smoke prediction was not printed.")

    print("\n[ok] small pipeline smoke test passed")
    print(
        f"  rollouts={len(observations)} actions={len(actions)} "
        f"failure_rows={len(rows)} labels={sorted(failure_labels)}"
    )
    print(f"  artifacts={agent_root}")


def main() -> None:
    scenario = resolve_scenario()
    output = run_pipeline(scenario)
    validate_outputs(scenario, output)


if __name__ == "__main__":
    main()
