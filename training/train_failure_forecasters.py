"""Train legacy-style mean and aleatoric forecasters for failure forecasting."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import grid2op
from lightsim2grid import LightSimBackend
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_config import (  # noqa: E402
    AGENT_FACTORY,
    AGENT_NAME,
    ARTIFACTS_DIR,
    ASSETS_DIR,
    ENV_DIR,
    ENV_NAME,
    SEED,
    configure_grid2op_warnings,
)
from src.agent_runtime import build_agent  # noqa: E402
from src.failure_forecast import (  # noqa: E402
    collect_forecaster_training_data,
    save_forecaster_training_data,
    train_aleatoric_forecaster,
    train_mean_forecaster,
)


def print_stage(number: int, total: int, title: str) -> None:
    """Print a compact stage marker for long-running forecast training."""
    print(f"\n[{number}/{total}] {title}")


def default_agent_artifact_dir(agent_name: str) -> Path:
    """Return the configured artifact root for one agent/environment pair."""
    return ARTIFACTS_DIR / ENV_NAME / agent_name


def default_forecast_dir(agent_name: str) -> Path:
    """Return where forecast data and trained forecasters are stored."""
    return default_agent_artifact_dir(agent_name) / "failure_forecast"


def valid_policy_dir(path: Path) -> bool:
    """Check whether a directory can be loaded as a CurriculumAgent package."""
    return (
        (path / "model" / "saved_model.pb").is_file()
        and (path / "actions" / "actions.npy").is_file()
    )


def default_policy_dir(agent_artifact_dir: Path) -> Path:
    """Prefer an explicit agent package, then fall back to active asset paths."""
    candidates = [
        agent_artifact_dir,
        ASSETS_DIR / ENV_NAME,
        ASSETS_DIR / "network36",
    ]
    for path in candidates:
        if valid_policy_dir(path):
            return path
    return ASSETS_DIR / ENV_NAME


def configure_quiet_logging() -> None:
    """Suppress verbose policy INFO logs while preserving warnings/errors."""
    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("curriculumagent").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-name", default=AGENT_NAME)
    parser.add_argument("--agent-factory", default=AGENT_FACTORY or None)
    parser.add_argument("--env-dir", type=Path, default=ENV_DIR)
    parser.add_argument(
        "--agent-artifact-dir",
        type=Path,
        default=None,
        help="artifact/output root; default artifacts/<ENV_NAME>/<agent-name>",
    )
    parser.add_argument(
        "--policy-agent-dir",
        type=Path,
        default=None,
        help="runnable CurriculumAgent package with model/ and actions/; default auto-discovers assets/<ENV_NAME>",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mean-trials", type=int, default=20)
    parser.add_argument("--max-subsample", type=int, default=5000)
    parser.add_argument("--reuse-data", action="store_true",
                        help="reuse forecast_X.npy and forecast_y.npy from out-dir when present")
    args = parser.parse_args()

    configure_grid2op_warnings()
    configure_quiet_logging()
    agent_artifact_dir = args.agent_artifact_dir or default_agent_artifact_dir(args.agent_name)
    out_dir = args.out_dir or (agent_artifact_dir / "failure_forecast")
    mean_path = out_dir / "mean_forecaster.pkl"
    aleatoric_path = out_dir / "aleatoric_forecaster.pkl"
    x_path = out_dir / "forecast_X.npy"
    y_path = out_dir / "forecast_y.npy"

    print("Failure forecaster training")
    print(f"  env_dir       : {args.env_dir}")
    print(f"  agent_name    : {args.agent_name}")
    print(f"  artifacts     : {agent_artifact_dir}")
    print(f"  output        : {out_dir}")
    print(f"  episodes      : {args.episodes}")
    print(f"  mean_trials   : {args.mean_trials}")

    if args.reuse_data and x_path.is_file() and y_path.is_file():
        import numpy as np

        print_stage(1, 3, "Reusing forecast training arrays")
        X = np.load(x_path)
        y = np.load(y_path)
        print(f"[data] reused {X.shape[0]} rows from {out_dir}")
    else:
        print_stage(1, 3, "Collecting forecast training arrays")
        env = grid2op.make(str(args.env_dir), backend=LightSimBackend())
        progress = tqdm(total=args.episodes, desc="Collecting forecast data", unit="episode")
        try:
            spec = args.agent_factory or None
            policy_dir = args.policy_agent_dir or default_policy_dir(agent_artifact_dir)
            if spec:
                agent = build_agent(env, factory_spec=spec, agent_path=None)
            else:
                if not valid_policy_dir(policy_dir):
                    raise FileNotFoundError(
                        f"Policy agent directory is incomplete: {policy_dir}. "
                        "Expected model/saved_model.pb and actions/actions.npy. "
                        "Pass --policy-agent-dir pointing to the runnable CurriculumAgent package."
                    )
                agent = build_agent(env, agent_path=policy_dir)

            def update(_episode: int, steps: int, total_rows: int) -> None:
                progress.update(1)
                progress.set_postfix(last_steps=steps, rows=total_rows, refresh=False)

            X, y = collect_forecaster_training_data(
                env,
                agent,
                episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps,
                progress_callback=update,
            )
        finally:
            progress.close()
            env.close()
        save_forecaster_training_data(out_dir, X, y)
        print(f"[data] wrote X={X.shape} y={y.shape} -> {out_dir}")

    print_stage(2, 3, "Training mean forecaster")

    def report_trial(done: int, total: int, best_rmse: float) -> None:
        """Print occasional Optuna progress without flooding the console."""
        if done == 1 or done == total or done % max(1, total // 5) == 0:
            print(f"[mean] trial {done}/{total} best_rmse={best_rmse:.4f}")

    mean_model = train_mean_forecaster(
        X,
        y,
        model_path=mean_path,
        n_trials=args.mean_trials,
        max_subsample=args.max_subsample,
        seed=args.seed,
        progress_callback=report_trial if args.mean_trials > 0 else None,
    )
    print_stage(3, 3, "Training aleatoric forecaster")
    train_aleatoric_forecaster(X, y, mean_model, model_path=aleatoric_path)
    print(f"[ok] wrote mean forecaster -> {mean_path}")
    print(f"[ok] wrote aleatoric forecaster -> {aleatoric_path}")


if __name__ == "__main__":
    main()
