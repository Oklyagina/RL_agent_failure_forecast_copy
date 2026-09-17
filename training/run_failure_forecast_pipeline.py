"""Run the failure-forecast workflow in one organized command."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_config import AGENT_NAME, ARTIFACTS_DIR, ASSETS_DIR, ENV_DIR, ENV_NAME, SEED  # noqa: E402


def default_agent_artifact_dir(agent_name: str) -> Path:
    """Return the standard artifact root for an agent/environment pair."""
    return ARTIFACTS_DIR / ENV_NAME / agent_name


def default_policy_dir() -> Path:
    """Return the default runnable CurriculumAgent package location."""
    return ASSETS_DIR / ENV_NAME


def split_lines(raw: str | None) -> list[str]:
    """Parse comma-separated line names while preserving CLI-friendly input."""
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def stage(number: int, total: int, name: str, command: list[str]) -> None:
    """Print and run one pipeline stage using the current Python interpreter."""
    print(f"\n=== [{number}/{total}] {name} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-name", default=AGENT_NAME)
    parser.add_argument("--env-dir", type=Path, default=ENV_DIR)
    parser.add_argument("--agent-artifact-dir", type=Path, default=None)
    parser.add_argument("--policy-agent-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--forecast-episodes", type=int, default=None)
    parser.add_argument("--failure-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sampling-stride", type=int, default=20)
    parser.add_argument("--lines", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mean-trials", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--reuse-forecast-data", action="store_true")
    parser.add_argument("--skip-forecasters", action="store_true")
    parser.add_argument("--skip-row-collection", action="store_true")
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--smoke-line", default=None)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    agent_artifacts = args.agent_artifact_dir or default_agent_artifact_dir(args.agent_name)
    policy_dir = args.policy_agent_dir or default_policy_dir()
    out_dir = agent_artifacts / "failure_forecast"
    mean_model = out_dir / "mean_forecaster.pkl"
    aleatoric_model = out_dir / "aleatoric_forecaster.pkl"
    rows_csv = out_dir / "failure_forecast_rows.csv"
    classifier = out_dir / "failure_classifier.pkl"
    lines = split_lines(args.lines)
    smoke_line = args.smoke_line or (lines[0] if lines else None)

    print("Failure forecast pipeline")
    print(f"  env_dir          : {args.env_dir}")
    print(f"  agent_artifacts  : {agent_artifacts}")
    print(f"  policy_agent_dir : {policy_dir}")
    print(f"  output           : {out_dir}")
    print(f"  forecast episodes: {args.forecast_episodes or args.episodes}")
    print(f"  failure episodes : {args.failure_episodes or args.episodes}")
    print(f"  lines            : {args.lines or 'all env lines'}")
    print(f"  smoke line       : {smoke_line or 'skipped'}")

    planned = [
        not args.skip_forecasters,
        not args.skip_row_collection,
        not args.skip_classifier,
        not args.skip_smoke and smoke_line is not None,
    ]
    total = sum(1 for item in planned if item)
    current = 1

    if not args.skip_forecasters:
        cmd = [
            sys.executable,
            "training/train_failure_forecasters.py",
            "--env-dir", str(args.env_dir),
            "--agent-name", args.agent_name,
            "--agent-artifact-dir", str(agent_artifacts),
            "--policy-agent-dir", str(policy_dir),
            "--episodes", str(args.forecast_episodes or args.episodes),
            "--seed", str(args.seed),
            "--mean-trials", str(args.mean_trials),
        ]
        if args.max_steps is not None:
            cmd.extend(["--max-steps", str(args.max_steps)])
        if args.reuse_forecast_data:
            cmd.append("--reuse-data")
        stage(current, total, "Train mean and aleatoric forecasters", cmd)
        current += 1

    if not args.skip_row_collection:
        cmd = [
            sys.executable,
            "training/collect_failure_forecast.py",
            "--agent-name", args.agent_name,
            "--episodes", str(args.failure_episodes or args.episodes),
            "--seed", str(args.seed),
            "--sampling-stride", str(args.sampling_stride),
            "--mean-model", str(mean_model),
            "--aleatoric-model", str(aleatoric_model),
            "--output-csv", str(rows_csv),
        ]
        if args.max_steps is not None:
            cmd.extend(["--max-steps", str(args.max_steps)])
        if args.lines:
            cmd.extend(["--lines", args.lines])
        stage(current, total, "Collect failure-forecast rows", cmd)
        current += 1

    if not args.skip_classifier:
        cmd = [
            sys.executable,
            "training/train_failure_forecast.py",
            "--agent-name", args.agent_name,
            "--input-csv", str(rows_csv),
            "--model-path", str(classifier),
            "--threshold", str(args.threshold),
            "--seed", str(args.seed),
        ]
        stage(current, total, "Train failure classifier", cmd)
        current += 1

    if not args.skip_smoke and smoke_line is not None:
        cmd = [
            sys.executable,
            "training/predict_failure_forecast.py",
            "--agent-name", args.agent_name,
            "--line", smoke_line,
            "--seed", str(args.seed),
            "--mean-model", str(mean_model),
            "--aleatoric-model", str(aleatoric_model),
            "--classifier", str(classifier),
        ]
        stage(current, total, "Smoke-test one prediction", cmd)

    print("\n[ok] failure forecast pipeline finished")


if __name__ == "__main__":
    main()
