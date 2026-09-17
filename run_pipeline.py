"""Orchestrate the active failure-forecast and ENN training pipeline.

This restores the root-level pipeline entrypoint from the archived workflow,
but keeps ENN training on the current rollout-based implementation:

    training/collect_rollouts.py -> training/train_enn.py

The ENN stage is policy-agnostic.  A policy is selected by AGENT_FACTORY when
configured, otherwise the default bundled policy assets are used by the rollout
collector.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_config import (  # noqa: E402
    AGENT_FACTORY,
    AGENT_NAME,
    ARTIFACTS_DIR,
    ASSETS_DIR,
    ENN_ANNEAL_EPOCHS,
    ENN_BATCH_SIZE,
    ENN_DROPOUT,
    ENN_EPOCHS,
    ENN_HIDDEN_DIM,
    ENN_LR,
    ENN_ROLLOUT_MAX_STEPS,
    ENN_VAL_FRAC,
    ENV_DIR,
    ENV_NAME,
    FAILURE_EPISODES,
    FAILURE_LINES,
    FAILURE_MAX_STEPS,
    FAILURE_SAMPLING_STRIDE,
    FAILURE_THRESHOLD,
    FORECAST_EPISODES,
    FORECAST_MAX_STEPS,
    FORECAST_MEAN_TRIALS,
    ROLLOUT_EPISODES,
    SEED,
)
from src.pipeline_artifacts import (  # noqa: E402
    ArtifactStatus,
    StageCheck,
    classify_outputs,
    mark_provenance_stale,
    provenance_path,
    write_provenance,
)


def agent_artifact_dir(agent_name: str) -> Path:
    return ARTIFACTS_DIR / ENV_NAME / agent_name


def rollout_dir(agent_name: str) -> Path:
    return agent_artifact_dir(agent_name) / "rollouts"


def model_dir(agent_name: str) -> Path:
    return agent_artifact_dir(agent_name) / "model"


def failure_dir(agent_name: str) -> Path:
    return agent_artifact_dir(agent_name) / "failure_forecast"


def default_policy_dir() -> Path:
    for path in (ASSETS_DIR / ENV_NAME, ASSETS_DIR / "network36"):
        if (path / "model").is_dir() and (path / "actions").is_dir():
            return path
    return ASSETS_DIR / ENV_NAME


def execute(command: list[str], *, verbose: bool = False) -> None:
    """Run one stage with the current interpreter and repository root."""
    print("\n" + "=" * 72)
    print(" ".join(command))
    print("=" * 72)
    if verbose:
        subprocess.run(command, cwd=ROOT, check=True)
        return

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _check_with_validator(
    outputs: Iterable[Path],
    primary: Path,
    validator: Callable[[], dict[str, int]],
    agent_name: str,
) -> StageCheck:
    check = classify_outputs(outputs, provenance_path(primary), ENV_NAME, agent_name)
    if check.status == ArtifactStatus.MISSING:
        return check
    try:
        check.dimensions = validator()
    except Exception as exc:
        return StageCheck(ArtifactStatus.INCOMPATIBLE, str(exc))
    return check


def validate_rollouts(path: Path) -> dict[str, int]:
    observations = np.load(path / "observations.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(path / "labels.npy", mmap_mode="r", allow_pickle=False)
    actions = np.load(path / "actions.npy", mmap_mode="r", allow_pickle=False)
    if observations.ndim != 2 or labels.ndim != 1 or actions.ndim != 2:
        raise ValueError(
            "rollout bundle must contain 2-D observations/actions and 1-D labels"
        )
    if len(observations) == 0 or len(observations) != len(labels):
        raise ValueError("rollout observations and labels have incompatible row counts")
    if actions.shape[0] < 2:
        raise ValueError("rollouts contain fewer than two distinct actions")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= actions.shape[0]:
        raise ValueError("rollout labels reference actions outside actions.npy")
    return {
        "rows": int(len(observations)),
        "input_dim": int(observations.shape[1]),
        "num_classes": int(actions.shape[0]),
    }


def _resolve_action_set(meta: dict, meta_path: Path) -> Path:
    raw = meta.get("action_set") or meta.get("actions_path")
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            return path
    fallback = meta_path.parents[1] / "rollouts" / "actions.npy"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError("ENN metadata does not reference an existing action set")


def validate_enn_bundle(path: Path, agent_name: str) -> dict[str, int]:
    meta_path = path / "enn_meta.json"
    scaler_path = path / "scaler_params.json"
    calib_path = path / "enn_pctile_calib.npz"
    weights_path = path / f"enn_{agent_name}.pth"
    for required in (meta_path, scaler_path, calib_path, weights_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    input_dim = int(meta["input_dim"])
    num_classes = int(meta["num_classes"])
    if int(scaler.get("n_features_in", -1)) != input_dim:
        raise ValueError("scaler feature count does not match ENN input dimension")
    if meta.get("environment") != ENV_NAME:
        raise ValueError(f"ENN environment is {meta.get('environment')!r}, expected {ENV_NAME!r}")
    action_set = np.load(_resolve_action_set(meta, meta_path), mmap_mode="r", allow_pickle=False)
    if action_set.ndim != 2 or action_set.shape[0] != num_classes:
        raise ValueError("ENN action set row count does not match num_classes")
    with np.load(calib_path, allow_pickle=False) as calibration:
        total_ref = np.asarray(calibration["total_ref"], dtype=float).reshape(-1)
        action_ref = np.asarray(calibration["action_ref"], dtype=float).reshape(-1)
    if len(total_ref) == 0 or len(action_ref) == 0:
        raise ValueError("ENN calibration references must be non-empty")
    return {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "calibration_rows": int(len(total_ref)),
    }


def print_preflight(
    rows: list[tuple[str, StageCheck, str]],
    agent_name: str,
) -> None:
    print("\nPIPELINE PREFLIGHT")
    print(f"  Environment : {ENV_NAME} ({ENV_DIR})")
    print(f"  Agent       : {agent_name}")
    print(f"  Agent source: {'factory ' + AGENT_FACTORY if AGENT_FACTORY else default_policy_dir()}")
    print(f"  Artifacts   : {agent_artifact_dir(agent_name)}")
    print(f"\n  {'Stage':20} {'Status':20} Action")
    print(f"  {'-' * 20} {'-' * 20} {'-' * 24}")
    for stage, check, action in rows:
        print(f"  {stage:20} {check.status.value:20} {action}")


def _write_stage(stage: str, primary: Path, outputs: Iterable[Path],
                 dimensions: dict[str, int], agent_name: str,
                 *, adopted: bool = False) -> None:
    write_provenance(
        provenance_path(primary),
        stage,
        ENV_NAME,
        agent_name,
        outputs,
        dimensions,
        adopted=adopted,
    )


def run_pipeline(args: argparse.Namespace) -> None:
    agent_root = agent_artifact_dir(args.agent_name)
    rollouts = rollout_dir(args.agent_name)
    models = model_dir(args.agent_name)
    failures = failure_dir(args.agent_name)
    policy_dir = default_policy_dir()
    forced = set(args.force_stage)
    force_all = "all" in forced

    rollout_outputs = [
        rollouts / "observations.npy",
        rollouts / "labels.npy",
        rollouts / "actions.npy",
    ]
    enn_outputs = [
        models / f"enn_{args.agent_name}.pth",
        models / "scaler_params.json",
        models / "enn_meta.json",
        models / "enn_pctile_calib.npz",
    ]
    forecast_outputs = [
        failures / "mean_forecaster.pkl",
        failures / "aleatoric_forecaster.pkl",
    ]
    rows_csv = failures / "failure_forecast_rows.csv"
    classifier = failures / "failure_classifier.pkl"
    classifier_meta = failures / "failure_classifier_metadata.json"

    rollout_check = _check_with_validator(
        rollout_outputs,
        rollouts / "observations.npy",
        lambda: validate_rollouts(rollouts),
        args.agent_name,
    )
    enn_check = _check_with_validator(
        enn_outputs,
        models / f"enn_{args.agent_name}.pth",
        lambda: validate_enn_bundle(models, args.agent_name),
        args.agent_name,
    )
    forecast_check = classify_outputs(
        forecast_outputs,
        provenance_path(failures / "mean_forecaster.pkl"),
        ENV_NAME,
        args.agent_name,
    )
    rows_check = classify_outputs([rows_csv], provenance_path(rows_csv), ENV_NAME, args.agent_name)
    classifier_check = classify_outputs(
        [classifier, classifier_meta],
        provenance_path(classifier),
        ENV_NAME,
        args.agent_name,
    )

    collect_rollouts = force_all or "enn-data" in forced or not rollout_check.reusable
    train_enn = force_all or "enn" in forced or collect_rollouts or not enn_check.reusable
    train_forecasters = force_all or "forecast" in forced or not forecast_check.reusable
    collect_rows = (
        force_all
        or "failure-rows" in forced
        or train_forecasters
        or train_enn
        or not rows_check.reusable
    )
    train_classifier = (
        force_all
        or "classifier" in forced
        or collect_rows
        or not classifier_check.reusable
    )

    print_preflight(
        [
            ("ENN rollouts", rollout_check, "collect" if collect_rollouts else "reuse"),
            ("ENN bundle", enn_check, "train" if train_enn else "reuse"),
            ("Forecasters", forecast_check, "train" if train_forecasters else "reuse"),
            ("Failure rows", rows_check, "collect" if collect_rows else "reuse"),
            ("Classifier", classifier_check, "train" if train_classifier else "reuse"),
        ],
        args.agent_name,
    )

    scheduled = [
        ("enn_rollouts", rollouts / "observations.npy", collect_rollouts),
        ("enn", models / f"enn_{args.agent_name}.pth", train_enn),
        ("forecasters", failures / "mean_forecaster.pkl", train_forecasters),
        ("failure_rows", rows_csv, collect_rows),
        ("failure_classifier", classifier, train_classifier),
    ]
    for stage, primary, will_run in scheduled:
        if will_run:
            mark_provenance_stale(
                provenance_path(primary),
                stage,
                ENV_NAME,
                args.agent_name,
                "stage scheduled to run",
            )

    for stage, primary, outputs, check in [
        ("enn_rollouts", rollouts / "observations.npy", rollout_outputs, rollout_check),
        ("enn", models / f"enn_{args.agent_name}.pth", enn_outputs, enn_check),
        ("forecasters", failures / "mean_forecaster.pkl", forecast_outputs, forecast_check),
        ("failure_rows", rows_csv, [rows_csv], rows_check),
        ("failure_classifier", classifier, [classifier, classifier_meta], classifier_check),
    ]:
        if check.status == ArtifactStatus.LEGACY_ADOPTABLE:
            _write_stage(
                stage, primary, outputs, check.dimensions,
                args.agent_name, adopted=True,
            )

    if collect_rollouts:
        command = [
            sys.executable,
            "training/collect_rollouts.py",
            "--agent-name", args.agent_name,
            "--episodes", str(ROLLOUT_EPISODES),
            "--out-dir", str(rollouts),
            "--seed", str(SEED),
        ]
        if ENN_ROLLOUT_MAX_STEPS > 0:
            command.extend(["--max-steps", str(ENN_ROLLOUT_MAX_STEPS)])
        if AGENT_FACTORY:
            command.extend(["--agent-factory", AGENT_FACTORY])
        execute(command, verbose=args.verbose)
        _write_stage(
            "enn_rollouts",
            rollouts / "observations.npy",
            rollout_outputs,
            validate_rollouts(rollouts),
            args.agent_name,
        )

    if train_enn:
        command = [
            sys.executable,
            "training/train_enn.py",
            "--agent-name", args.agent_name,
            "--data-dir", str(rollouts),
            "--out-dir", str(models),
            "--epochs", str(ENN_EPOCHS),
            "--anneal-epochs", str(ENN_ANNEAL_EPOCHS),
            "--batch-size", str(ENN_BATCH_SIZE),
            "--lr", str(ENN_LR),
            "--hidden-dim", str(ENN_HIDDEN_DIM),
            "--dropout", str(ENN_DROPOUT),
            "--val-frac", str(ENN_VAL_FRAC),
            "--seed", str(SEED),
        ]
        execute(command, verbose=args.verbose)
        _write_stage(
            "enn",
            models / f"enn_{args.agent_name}.pth",
            enn_outputs,
            validate_enn_bundle(models, args.agent_name),
            args.agent_name,
        )

    if train_forecasters:
        command = [
            sys.executable,
            "training/train_failure_forecasters.py",
            "--agent-name", args.agent_name,
            "--env-dir", str(ENV_DIR),
            "--agent-artifact-dir", str(agent_root),
            "--policy-agent-dir", str(policy_dir),
            "--out-dir", str(failures),
            "--episodes", str(FORECAST_EPISODES),
            "--seed", str(SEED),
            "--mean-trials", str(FORECAST_MEAN_TRIALS),
        ]
        if FORECAST_MAX_STEPS > 0:
            command.extend(["--max-steps", str(FORECAST_MAX_STEPS)])
        if AGENT_FACTORY:
            command.extend(["--agent-factory", AGENT_FACTORY])
        if args.reuse_forecast_data:
            command.append("--reuse-data")
        execute(command, verbose=args.verbose)
        _write_stage(
            "forecasters",
            failures / "mean_forecaster.pkl",
            forecast_outputs,
            {},
            args.agent_name,
        )

    if collect_rows:
        command = [
            sys.executable,
            "training/collect_failure_forecast.py",
            "--agent-name", args.agent_name,
            "--episodes", str(FAILURE_EPISODES),
            "--seed", str(SEED),
            "--sampling-stride", str(FAILURE_SAMPLING_STRIDE),
            "--mean-model", str(failures / "mean_forecaster.pkl"),
            "--aleatoric-model", str(failures / "aleatoric_forecaster.pkl"),
            "--output-csv", str(rows_csv),
        ]
        if FAILURE_MAX_STEPS > 0:
            command.extend(["--max-steps", str(FAILURE_MAX_STEPS)])
        if FAILURE_LINES:
            command.extend(["--lines", FAILURE_LINES])
        if AGENT_FACTORY:
            command.extend(["--agent-factory", AGENT_FACTORY])
        execute(command, verbose=args.verbose)
        _write_stage("failure_rows", rows_csv, [rows_csv], {}, args.agent_name)

    if train_classifier:
        command = [
            sys.executable,
            "training/train_failure_forecast.py",
            "--agent-name", args.agent_name,
            "--input-csv", str(rows_csv),
            "--model-path", str(classifier),
            "--threshold", str(FAILURE_THRESHOLD),
            "--seed", str(SEED),
        ]
        execute(command, verbose=args.verbose)
        _write_stage(
            "failure_classifier",
            classifier,
            [classifier, classifier_meta],
            {},
            args.agent_name,
        )

    if not args.skip_smoke and (args.smoke_line or FAILURE_LINES):
        smoke_line = args.smoke_line or FAILURE_LINES.split(",")[0].strip()
        command = [
            sys.executable,
            "training/predict_failure_forecast.py",
            "--agent-name", args.agent_name,
            "--line", smoke_line,
            "--seed", str(SEED),
            "--mean-model", str(failures / "mean_forecaster.pkl"),
            "--aleatoric-model", str(failures / "aleatoric_forecaster.pkl"),
            "--classifier", str(classifier),
        ]
        execute(command, verbose=args.verbose)

    print("\n[ok] pipeline finished")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the configured ENN and failure-forecast pipeline.",
    )
    parser.add_argument(
        "--agent-name",
        default=AGENT_NAME,
        help="artifact namespace; policy selection comes from AGENT_FACTORY",
    )
    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        choices=("enn-data", "enn", "forecast", "failure-rows", "classifier", "all"),
        help="rerun a stage; may be supplied more than once",
    )
    parser.add_argument(
        "--reuse-forecast-data",
        action="store_true",
        help="reuse cached forecaster training rows when that stage runs",
    )
    parser.add_argument("--smoke-line", help="line name for a final smoke prediction")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    run_pipeline(build_parser().parse_args())


if __name__ == "__main__":
    main()
