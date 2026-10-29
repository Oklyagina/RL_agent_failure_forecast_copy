"""Train the production failure-forecast bundle from configured full-case data.

Edit ``.env`` to choose the environment, agent, training duration, and lines,
then run this file directly.  The public
``run_full_forecasting`` function accepts the same values as keyword overrides
for programmatic use; no command-line parser or subprocess is involved.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tqdm import tqdm

from project_config import (
    AGENT_FACTORY,
    AGENT_NAME,
    ARTIFACTS_DIR,
    ASSETS_DIR,
    ENV_DIR,
    ENV_NAME,
    FAILURE_CLASSIFIER_TRIALS,
    FAILURE_EPISODES,
    FAILURE_LINES,
    FAILURE_MAX_STEPS,
    FAILURE_SEED_START,
    FORECAST_EPISODES,
    FORECAST_HGB_THREADS,
    FORECAST_MAX_STEPS,
    FORECAST_MEAN_TRIALS,
    SEED,
    configure_grid2op_warnings,
)
from src.agent_runtime import build_agent
from src.failure_forecast import (
    HORIZON_STEPS,
    LABEL_DISCONNECTION_TEST,
    LABEL_EPISODE_ENDED,
    LEGACY_PHASES,
    MEAN_BEST_PARAMS_FILENAME,
    STEP_MINUTES,
    FailureForecastConfig,
    FailureForecastPredictor,
    collect_failure_forecast_phases,
    collect_forecaster_training_data,
    load_or_train_forecasters,
    normalize_line_name,
    prepare_classifier_dataframe,
    save_forecaster_training_data,
    train_failure_classifier,
    write_rows_csv,
)
from src.pipeline_artifacts import mark_provenance_stale, provenance_path, write_provenance


def _make_environment(env_dir: Path, **make_kwargs: Any) -> Any:
    """Create the configured Grid2Op environment with the production backend."""
    import grid2op
    from lightsim2grid import LightSimBackend

    return grid2op.make(str(env_dir), backend=LightSimBackend(), **make_kwargs)


def _valid_policy_dir(path: Path) -> bool:
    """Return whether a directory contains a runnable CurriculumAgent bundle."""
    variables = path / "model" / "variables"
    required = (
        path / "actions" / "actions.npy",
        path / "model" / "saved_model.pb",
        variables / "variables.index",
        variables / "variables.data-00000-of-00001",
    )
    return all(item.is_file() and item.stat().st_size > 0 for item in required)


def _default_policy_dir() -> Path:
    """Find the configured runnable CurriculumAgent package."""
    for path in (ASSETS_DIR / ENV_NAME, ASSETS_DIR / "network36"):
        if _valid_policy_dir(path):
            return path
    return ASSETS_DIR / ENV_NAME


def _make_agent(env: Any, agent_factory: str | None,
                policy_agent_dir: Path | None) -> Any:
    """Build the configured policy through the existing agent interface."""
    if agent_factory:
        return build_agent(env, factory_spec=agent_factory, agent_path=None)
    policy_dir = policy_agent_dir or _default_policy_dir()
    if not _valid_policy_dir(policy_dir):
        raise FileNotFoundError(
            f"Policy agent directory is incomplete: {policy_dir}. Expected "
            "model/saved_model.pb, model/variables, and actions/actions.npy, "
            "or configure AGENT_FACTORY."
        )
    return build_agent(env, agent_path=policy_dir)


def _configured_lines(lines: Sequence[str] | None, env: Any) -> list[str]:
    """Resolve explicit lines, FAILURE_LINES, or all lines in that order."""
    if lines is None:
        selected = [part.strip() for part in FAILURE_LINES.split(",") if part.strip()]
    elif isinstance(lines, str):
        selected = [part.strip() for part in lines.split(",") if part.strip()]
    else:
        selected = [str(part).strip() for part in lines if str(part).strip()]
    if selected:
        return selected
    names = getattr(env, "name_line", None)
    if names is None:
        raise ValueError("No lines were configured and env.name_line is unavailable.")
    return [str(name) for name in names]


def _step_limit(value: int | None, name: str) -> int | None:
    """Translate the configured zero sentinel into an uncapped episode."""
    if value is None or int(value) == 0:
        return None
    if int(value) < 0:
        raise ValueError(f"{name} must be zero or a positive integer.")
    if int(value) <= HORIZON_STEPS:
        raise ValueError(
            f"{name} must exceed {HORIZON_STEPS} to produce t+12 samples."
        )
    return int(value)


def _configure_logging() -> None:
    """Keep long production runs readable without hiding errors."""
    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("curriculumagent").setLevel(logging.WARNING)


def _validate_forecast_arrays(
    X: np.ndarray,
    y: np.ndarray,
    episode_ids: np.ndarray,
) -> dict[str, int]:
    """Validate the persisted feature/target contract for forecast training."""
    if X.ndim != 2 or y.ndim != 2:
        raise ValueError(f"forecast arrays must be two-dimensional: X={X.shape}, y={y.shape}")
    if len(X) == 0 or len(X) != len(y):
        raise ValueError(f"forecast arrays need equal nonzero row counts: X={X.shape}, y={y.shape}")
    if X.dtype != np.dtype(np.float32) or y.dtype != np.dtype(np.float32):
        raise ValueError(
            "forecast arrays must use float32: "
            f"X={X.dtype}, y={y.dtype}"
        )
    if y.shape[1] <= 0:
        raise ValueError("forecast targets must contain at least one injection value")
    if episode_ids.ndim != 1 or len(episode_ids) != len(X):
        raise ValueError(
            "forecast episode IDs must be one-dimensional and match X/y rows"
        )
    if not np.issubdtype(episode_ids.dtype, np.integer):
        raise ValueError(f"forecast episode IDs must be integers: {episode_ids.dtype}")

    # get_features_with_history stores four injection snapshots (current, 1h,
    # 1d, 1w) followed by seven calendar/time features.
    expected_input_dim = 4 * int(y.shape[1]) + 7
    if int(X.shape[1]) != expected_input_dim:
        raise ValueError(
            f"forecast input width is {X.shape[1]}, expected {expected_input_dim} "
            f"for output width {y.shape[1]}"
        )
    return {
        "rows": int(len(X)),
        "input_dim": int(X.shape[1]),
        "output_dim": int(y.shape[1]),
        "episodes_with_rows": int(len(np.unique(episode_ids))),
    }


def _forecast_data_dimensions(
    X: np.ndarray,
    y: np.ndarray,
    episode_ids: np.ndarray,
    *,
    forecast_episodes: int,
    forecast_limit: int | None,
    seed: int,
) -> dict[str, int]:
    """Describe the dataset and the collection settings needed for reuse."""
    dimensions = _validate_forecast_arrays(X, y, episode_ids)
    dimensions.update({
        "forecast_episodes": int(forecast_episodes),
        "forecast_max_steps": 0 if forecast_limit is None else int(forecast_limit),
        "seed": int(seed),
        "horizon_steps": int(HORIZON_STEPS),
        "horizon_minutes": int(HORIZON_STEPS * STEP_MINUTES),
        "forecast_sampling_stride": 1,
    })
    return dimensions


def _load_forecast_data_cache(
    paths: dict[str, Path],
    *,
    agent_name: str,
    forecast_episodes: int,
    forecast_limit: int | None,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, str]:
    """Load a structurally valid, provenance-confirmed t+12 dataset cache."""
    outputs = [
        paths["forecast_X"],
        paths["forecast_y"],
        paths["forecast_episode_ids"],
    ]
    missing = [path.name for path in outputs if not path.is_file()]
    if missing:
        return None, "missing " + ", ".join(missing)

    sidecar = provenance_path(paths["forecast_X"])
    if not sidecar.is_file():
        return None, "forecast data provenance is missing"
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"forecast data provenance is invalid: {exc}"

    expected_settings = {
        "forecast_episodes": int(forecast_episodes),
        "forecast_max_steps": 0 if forecast_limit is None else int(forecast_limit),
        "seed": int(seed),
        "horizon_steps": int(HORIZON_STEPS),
        "horizon_minutes": int(HORIZON_STEPS * STEP_MINUTES),
        "forecast_sampling_stride": 1,
    }
    if metadata.get("stage") != "forecast_data":
        return None, "forecast data provenance has the wrong stage"
    if metadata.get("status") != "reusable":
        return None, "forecast data provenance is not reusable"
    if metadata.get("environment") != ENV_NAME:
        return None, "forecast data provenance has a different environment"
    if metadata.get("agent") != agent_name:
        return None, "forecast data provenance has a different agent"
    metadata_outputs = metadata.get("outputs")
    if not isinstance(metadata_outputs, list):
        return None, "forecast data provenance has invalid outputs"
    recorded_outputs = sorted(str(path) for path in metadata_outputs)
    if recorded_outputs != sorted(str(path) for path in outputs):
        return None, "forecast data provenance references different files"
    recorded_dimensions = metadata.get("dimensions") or {}
    for key, expected in expected_settings.items():
        if recorded_dimensions.get(key) != expected:
            return None, f"forecast data provenance has incompatible {key}"

    try:
        X = np.load(paths["forecast_X"], mmap_mode="r", allow_pickle=False)
        y = np.load(paths["forecast_y"], mmap_mode="r", allow_pickle=False)
        episode_ids = np.load(
            paths["forecast_episode_ids"], mmap_mode="r", allow_pickle=False
        )
        actual_dimensions = _validate_forecast_arrays(X, y, episode_ids)
    except (OSError, ValueError) as exc:
        return None, f"forecast arrays are invalid: {exc}"
    for key, actual in actual_dimensions.items():
        if recorded_dimensions.get(key) != actual:
            return None, f"forecast data provenance has incompatible {key}"
    return (X, y, episode_ids), "compatible every-step t+12 forecast data"


def _validate_rows(
    rows: list[dict[str, Any]],
    expected_lines: Sequence[str] | None = None,
) -> dict[str, int]:
    """Validate horizon and label invariants before classifier training."""
    if not rows:
        raise RuntimeError("Failure-row collection produced no rows.")
    if {int(row["horizon_steps"]) for row in rows} != {HORIZON_STEPS}:
        raise RuntimeError("Failure rows do not consistently use a t+12 horizon.")
    expected_minutes = HORIZON_STEPS * STEP_MINUTES
    if {int(row["horizon_minutes"]) for row in rows} != {expected_minutes}:
        raise RuntimeError("Failure rows do not consistently use a 60-minute horizon.")
    labels = {int(row["failed"]) for row in rows}
    if labels != {0, 1}:
        raise RuntimeError(
            "Failure classifier training requires both labels 0 and 1; "
            f"collection produced {sorted(labels)}."
        )
    row_lines = list(dict.fromkeys(str(row["line_disconnected"]) for row in rows))
    # Only the configured lines are disconnected in the forced-disconnection test.
    if expected_lines is not None and not set(row_lines) <= {str(line) for line in expected_lines}:
        raise RuntimeError(f"Failure rows contain unconfigured lines: {row_lines}.")
    allowed_reasons = {LABEL_DISCONNECTION_TEST, LABEL_EPISODE_ENDED}
    label_reasons = {str(row.get("label_reason", "")) for row in rows}
    if not label_reasons <= allowed_reasons:
        raise RuntimeError(f"Unexpected label reasons: {label_reasons}")
    return {
        "rows": len(rows),
        "lines": len(row_lines),
        "failures": sum(int(row["failed"]) for row in rows),
    }


def _validate_artifacts(
    paths: dict[str, Path],
    agent_name: str,
    expected_lines: Sequence[str],
) -> tuple[Any, Any]:
    """Reopen the complete saved bundle and validate its metadata contract."""
    import joblib

    required = [
        paths["forecast_X"], paths["forecast_y"],
        paths["forecast_episode_ids"], paths["mean_forecaster"],
        paths["aleatoric_forecaster"], paths["failure_rows"],
        paths["failure_classifier"], paths["classifier_metadata"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing forecast artifacts: " + ", ".join(missing))

    X = np.load(paths["forecast_X"], mmap_mode="r", allow_pickle=False)
    y = np.load(paths["forecast_y"], mmap_mode="r", allow_pickle=False)
    episode_ids = np.load(
        paths["forecast_episode_ids"], mmap_mode="r", allow_pickle=False
    )
    try:
        _validate_forecast_arrays(X, y, episode_ids)
    except ValueError as exc:
        raise RuntimeError(f"Invalid saved forecast arrays: {exc}") from exc

    mean_model = joblib.load(paths["mean_forecaster"])
    aleatoric_model = joblib.load(paths["aleatoric_forecaster"])
    joblib.load(paths["failure_classifier"])
    metadata = json.loads(paths["classifier_metadata"].read_text(encoding="utf-8"))
    if metadata.get("environment") != ENV_NAME:
        raise RuntimeError(
            f"Classifier environment is {metadata.get('environment')!r}, "
            f"expected {ENV_NAME!r}."
        )
    if metadata.get("agent") != agent_name:
        raise RuntimeError(
            f"Classifier agent is {metadata.get('agent')!r}, expected {agent_name!r}."
        )
    if not metadata.get("features") or not metadata.get("line_map"):
        raise RuntimeError("Classifier metadata is missing features or the line map.")
    if not set(metadata["line_map"]) <= {normalize_line_name(line) for line in expected_lines}:
        raise RuntimeError("Classifier metadata contains lines that were not configured.")
    decision_threshold = float(metadata.get("decision_threshold", np.nan))
    if not np.isfinite(decision_threshold) or not 0.0 <= decision_threshold <= 1.0:
        raise RuntimeError("Classifier metadata contains an invalid decision threshold.")
    print(f"[artifacts] reopened X={X.shape} y={y.shape} and all model files")
    return mean_model, aleatoric_model


def _validate_inference(
    env_dir: Path,
    output_dir: Path,
    agent_name: str,
    lines: Sequence[str],
    mean_model: Any,
    aleatoric_model: Any,
    seed: int,
) -> list[dict[str, Any]]:
    """Run one inference pass; it needs no agent and no simulation."""
    env = _make_environment(env_dir)
    try:
        try:
            obs = env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
            obs = env.reset()
        cfg = FailureForecastConfig.from_env(
            env,
            env_name=ENV_NAME,
            agent_name=agent_name,
            artifact_dir=output_dir,
            lines_to_test=lines,
        )
        predictor = FailureForecastPredictor.load(
            output_dir / "failure_classifier.pkl"
        )
        predictions = predictor.predict_many(
            obs,
            [obs],
            lines,
            mean_model,
            aleatoric_model,
            cfg,
            env=env,
        )
    finally:
        env.close()

    if len(predictions) != len(lines):
        raise RuntimeError(
            f"Expected {len(lines)} predictions, received {len(predictions)}."
        )
    for prediction in predictions:
        if "failure_probability" not in prediction or "failure_prediction" not in prediction:
            raise RuntimeError(f"Incomplete inference result: {prediction}")
    return predictions


def run_full_forecasting(
    *,
    forecast_episodes: int = FORECAST_EPISODES,
    failure_episodes: int = FAILURE_EPISODES,
    forecast_max_steps: int = FORECAST_MAX_STEPS,
    failure_max_steps: int = FAILURE_MAX_STEPS,
    mean_trials: int = FORECAST_MEAN_TRIALS,
    hgb_threads: int = FORECAST_HGB_THREADS,
    classifier_trials: int = FAILURE_CLASSIFIER_TRIALS,
    failure_seed_start: int = FAILURE_SEED_START,
    lines: Sequence[str] | None = None,
    seed: int = SEED,
    env_dir: Path = ENV_DIR,
    agent_name: str = AGENT_NAME,
    agent_factory: str | None = AGENT_FACTORY or None,
    policy_agent_dir: Path | None = None,
    artifact_dir: Path | None = None,
    reuse_forecast_data: bool = True,
    reuse_forecasters: bool = True,
    force_mean_retune: bool = False,
) -> dict[str, Path]:
    """Train and validate the configured production failure-forecast bundle.

    ``artifact_dir`` is the final ``failure_forecast`` output directory.  When
    omitted, the canonical configured location is used.
    """
    if int(forecast_episodes) <= 0 or int(failure_episodes) <= 0:
        raise ValueError("forecast_episodes and failure_episodes must be positive.")
    if int(mean_trials) < 0:
        raise ValueError("mean_trials must be zero or positive.")
    if int(classifier_trials) < 0:
        raise ValueError("classifier_trials must be zero or positive.")
    if int(hgb_threads) <= 0:
        raise ValueError("hgb_threads must be positive.")
    forecast_limit = _step_limit(forecast_max_steps, "forecast_max_steps")
    failure_limit = _step_limit(failure_max_steps, "failure_max_steps")
    env_dir = Path(env_dir)
    if not env_dir.is_dir():
        raise FileNotFoundError(f"Grid2Op environment not found: {env_dir}")
    output_dir = Path(artifact_dir) if artifact_dir else (
        ARTIFACTS_DIR / ENV_NAME / agent_name / "failure_forecast"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "forecast_X": output_dir / "forecast_X.npy",
        "forecast_y": output_dir / "forecast_y.npy",
        "forecast_episode_ids": output_dir / "forecast_episode_ids.npy",
        "mean_forecaster": output_dir / "mean_forecaster.pkl",
        "mean_best_params": output_dir / MEAN_BEST_PARAMS_FILENAME,
        "aleatoric_forecaster": output_dir / "aleatoric_forecaster.pkl",
        "failure_rows": output_dir / "failure_forecast_rows.csv",
        "failure_classifier": output_dir / "failure_classifier.pkl",
        "classifier_metadata": output_dir / "failure_classifier_metadata.json",
    }
    forecast_outputs = [paths["mean_forecaster"], paths["aleatoric_forecaster"]]
    classifier_outputs = [paths["failure_classifier"], paths["classifier_metadata"]]

    configure_grid2op_warnings()
    _configure_logging()
    print("Full-case failure-forecast training")
    print(f"  environment       : {ENV_NAME} ({env_dir})")
    print(f"  agent             : {agent_name}")
    print(f"  forecast episodes : {forecast_episodes}")
    print("  forecast cadence  : every step (legacy), target t+12")
    print(f"  mean trials       : {mean_trials}")
    print("  mean output jobs  : 1 (sequential)")
    print(f"  HGB threads       : {hgb_threads}")
    print(f"  classifier trials : {classifier_trials}")
    print(f"  failure episodes  : {failure_episodes} x {len(LEGACY_PHASES)} phases")
    print(f"  forecast max steps: {forecast_limit or 'full episode'}")
    print(f"  failure max steps : {failure_limit or 'full episode'}")
    print(f"  reuse forecast data: {bool(reuse_forecast_data)}")
    print(f"  reuse forecasters : {bool(reuse_forecasters)}")
    print(f"  output            : {output_dir}")

    cached_data = None
    cache_reason = "reuse disabled"
    if reuse_forecast_data:
        cached_data, cache_reason = _load_forecast_data_cache(
            paths,
            agent_name=agent_name,
            forecast_episodes=int(forecast_episodes),
            forecast_limit=forecast_limit,
            seed=int(seed),
        )

    if cached_data is not None:
        print("\n[1/6] Reuse genuine t -> t+12 forecast pairs")
        X, y, episode_ids = cached_data
        print(f"[cache] {cache_reason}: X={X.shape} y={y.shape}")
    else:
        print("\n[1/6] Collect genuine t -> t+12 forecast pairs")
        print(f"[cache] collecting fresh data: {cache_reason}")
        mark_provenance_stale(
            provenance_path(paths["forecast_X"]),
            "forecast_data",
            ENV_NAME,
            agent_name,
            cache_reason,
        )
        env = _make_environment(env_dir)
        progress = tqdm(
            total=int(forecast_episodes), desc="Forecast data", unit="episode"
        )
        try:
            agent = _make_agent(env, agent_factory, policy_agent_dir)

            def forecast_progress(_episode: int, steps: int, total_rows: int) -> None:
                progress.update(1)
                progress.set_postfix(last_steps=steps, rows=total_rows, refresh=False)

            X, y, episode_ids = collect_forecaster_training_data(
                env,
                agent,
                episodes=int(forecast_episodes),
                seed=int(seed),
                max_steps=forecast_limit,
                sampling_stride=1,
                return_episode_ids=True,
                progress_callback=forecast_progress,
            )
        finally:
            progress.close()
            env.close()
        save_forecaster_training_data(output_dir, X, y, episode_ids)
        data_dimensions = _forecast_data_dimensions(
            X,
            y,
            episode_ids,
            forecast_episodes=int(forecast_episodes),
            forecast_limit=forecast_limit,
            seed=int(seed),
        )
        write_provenance(
            provenance_path(paths["forecast_X"]),
            "forecast_data",
            ENV_NAME,
            agent_name,
            [
                paths["forecast_X"],
                paths["forecast_y"],
                paths["forecast_episode_ids"],
            ],
            data_dimensions,
        )
        print(f"[data] X={X.shape} y={y.shape}")

    expected_bundle_dimensions = {
        "rows": int(len(X)),
        "input_dim": int(X.shape[1]),
        "output_dim": int(y.shape[1]),
    }

    def mark_forecasters_stale() -> None:
        mark_provenance_stale(
            provenance_path(paths["mean_forecaster"]),
            "forecasters",
            ENV_NAME,
            agent_name,
            "forecaster stage is being resumed or replaced",
        )

    print("\n[2-3/6] Load or train injection forecasters")
    mean_model, aleatoric_model, _ = load_or_train_forecasters(
        X,
        y,
        episode_ids,
        mean_path=paths["mean_forecaster"],
        aleatoric_path=paths["aleatoric_forecaster"],
        best_params_path=paths["mean_best_params"],
        n_trials=int(mean_trials),
        seed=int(seed),
        native_threads=int(hgb_threads),
        reuse=bool(reuse_forecasters),
        force_retune=bool(force_mean_retune),
        on_train_start=mark_forecasters_stale,
    )
    write_provenance(
        provenance_path(paths["mean_forecaster"]),
        "forecasters",
        ENV_NAME,
        agent_name,
        forecast_outputs,
        expected_bundle_dimensions,
    )

    print("\n[4/6] Collect forced-disconnection failure rows (legacy phases)")
    mark_provenance_stale(
        provenance_path(paths["failure_rows"]),
        "failure_rows",
        ENV_NAME,
        agent_name,
        "failure-row collection is starting",
    )
    env = _make_environment(env_dir)
    progress = tqdm(
        total=int(failure_episodes) * len(LEGACY_PHASES),
        desc="Failure rows", unit="episode",
    )
    try:
        agent = _make_agent(env, agent_factory, policy_agent_dir)
        selected_lines = _configured_lines(lines, env)
        cfg = FailureForecastConfig.from_env(
            env,
            env_name=ENV_NAME,
            agent_name=agent_name,
            artifact_dir=output_dir,
            lines_to_test=selected_lines,
        )

        def failure_progress(_episode: int, steps: int, stats: Any) -> None:
            progress.n = stats["episodes"]
            progress.set_postfix(
                last_steps=steps, samples=stats["rows"], pos=stats["positive"],
                neg=stats["negative"], points=stats["sample_points"], refresh=False,
            )
            progress.refresh()

        def make_phase_env(**kwargs: Any) -> Any:
            return _make_environment(env_dir, **kwargs)

        rows, collection_stats = collect_failure_forecast_phases(
            agent,
            make_phase_env,
            mean_model,
            aleatoric_model,
            cfg,
            episodes=int(failure_episodes),
            seed=int(failure_seed_start),
            max_steps=failure_limit,
            progress_callback=failure_progress,
        )
    finally:
        progress.close()
        env.close()
    print("[labels] collection stats: " + ", ".join(
        f"{key}={value}" for key, value in sorted(collection_stats.items())))
    row_dimensions = _validate_rows(rows, selected_lines)
    written = write_rows_csv(rows, paths["failure_rows"])
    if written != len(rows):
        raise RuntimeError(f"Wrote {written} rows, expected {len(rows)}.")
    write_provenance(
        provenance_path(paths["failure_rows"]),
        "failure_rows",
        ENV_NAME,
        agent_name,
        [paths["failure_rows"]],
        row_dimensions,
    )
    print(
        f"[data] rows={row_dimensions['rows']} lines={row_dimensions['lines']} "
        f"failures={row_dimensions['failures']} "
        f"successes={row_dimensions['rows'] - row_dimensions['failures']}"
    )

    print("\n[5/6] Train failure classifier")
    mark_provenance_stale(
        provenance_path(paths["failure_classifier"]),
        "failure_classifier",
        ENV_NAME,
        agent_name,
        "failure-classifier training is starting",
    )
    dataframe = prepare_classifier_dataframe(paths["failure_rows"])
    _, classifier_info = train_failure_classifier(
        dataframe,
        model_path=paths["failure_classifier"],
        seed=int(seed),
        env_name=ENV_NAME,
        agent_name=agent_name,
        n_trials=int(classifier_trials),
    )
    write_provenance(
        provenance_path(paths["failure_classifier"]),
        "failure_classifier",
        ENV_NAME,
        agent_name,
        classifier_outputs,
        {
            "rows": int(len(dataframe)),
            "features": int(len(classifier_info["features"])),
            "lines": int(len(classifier_info["line_map"])),
        },
    )
    metrics = classifier_info["metrics"]
    print(
        f"[test] threshold={classifier_info['decision_threshold']:.6f} "
        f"F2={metrics['f2']:.3f} precision={metrics['precision']:.3f} "
        f"recall={metrics['recall']:.3f} PR-AUC={metrics['pr_auc']:.3f} "
        f"FA={metrics['false_alarms']} ({metrics['false_alarm_pct']:.2f}%) "
        f"missed={metrics['missed_failures']} "
        f"({metrics['missed_failure_pct']:.2f}%)"
    )
    baseline = classifier_info["baseline_threshold_0_5_metrics"]
    print(
        f"[baseline@0.5] F2={baseline['f2']:.3f} "
        f"precision={baseline['precision']:.3f} recall={baseline['recall']:.3f} "
        f"FA={baseline['false_alarms']} missed={baseline['missed_failures']}"
    )
    split = classifier_info["split"]
    print(
        f"[split] {split['strategy']} "
        f"train_episodes={split['train_episodes']} "
        f"val_episodes={split['val_episodes']} "
        f"test_episodes={split['test_episodes']}"
    )
    label_reason_counts = classifier_info["label_reason_counts"]
    print(f"[labels] {label_reason_counts}")
    observed_failures = int(classifier_info["label_counts"].get("1", 0))
    if observed_failures < 20:
        print(
            f"[limitation] only {observed_failures} failure rows were "
            "collected; collect more episodes before relying on the classifier."
        )

    print("\n[6/6] Validate per-line inference")
    mean_model, aleatoric_model = _validate_artifacts(
        paths, agent_name, selected_lines
    )
    predictions = _validate_inference(
        env_dir,
        output_dir,
        agent_name,
        list(json.loads(paths["classifier_metadata"].read_text(encoding="utf-8"))["line_map"]),
        mean_model,
        aleatoric_model,
        int(seed),
    )
    print(json.dumps(predictions, indent=2, default=float))
    print("\n[ok] full-case failure-forecast training completed")
    return paths


def main() -> None:
    """Run with values loaded from .env and project_config.py."""
    run_full_forecasting()


if __name__ == "__main__":
    main()
