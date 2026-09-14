"""Generate a CSV dataset for future RL-agent failure classification.

Most users can run this with no arguments:

    python training/collect_failure_dataset.py

The script will run the configured Grid2Op agent, write one row per timestep,
and store the CSV under artifacts/<ENV_NAME>/<AGENT_NAME>/failure_dataset/.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    import loguru
except ImportError:  # pragma: no cover - project requirements include loguru
    class _FallbackLogger:
        def _format(self, message: str, **kwargs) -> str:
            try:
                return message.format(**kwargs)
            except Exception:
                return message

        def info(self, message: str, **kwargs) -> None:
            print("[info] " + self._format(message, **kwargs))

        def warning(self, message: str, **kwargs) -> None:
            print("[warn] " + self._format(message, **kwargs))

        def success(self, message: str, **kwargs) -> None:
            print("[ok] " + self._format(message, **kwargs))

    class _FallbackLoguru:
        logger = _FallbackLogger()

    loguru = _FallbackLoguru()

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
    ENN_ROLLOUT_MAX_STEPS,
    ROLLOUT_EPISODES,
    SEED,
    configure_grid2op_warnings,
)
from src.agent_runtime import build_agent  # noqa: E402
from src.failure_dataset import (  # noqa: E402
    collect_fresh_failure_rows,
    collect_replayed_failure_rows,
    iter_artifact_feature_rows,
    rollout_artifacts_available,
    write_failure_dataset_csv,
)


def default_rollout_dir(agent_name: str) -> Path:
    return ARTIFACTS_DIR / ENV_NAME / agent_name / "rollouts"


def default_output_path(agent_name: str) -> Path:
    return ARTIFACTS_DIR / ENV_NAME / agent_name / "failure_dataset" / "failure_dataset.csv"


def _candidate_agent_dirs() -> list[Path]:
    return [
        ASSETS_DIR / ENV_NAME,
        ASSETS_DIR / "network36",
    ]


def _default_agent_path() -> Path:
    for path in _candidate_agent_dirs():
        if (path / "model").is_dir() and (path / "actions").is_dir():
            return path
    return ASSETS_DIR / ENV_NAME


def configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger().setLevel(logging.WARNING)


def make_env():
    import grid2op
    from lightsim2grid import LightSimBackend

    return grid2op.make(str(ENV_DIR), backend=LightSimBackend())


def log_progress(event: str, payload: dict) -> None:
    if event == "episode_start":
        loguru.logger.info(
            "Starting episode {episode}/{last_episode}",
            episode=payload["episode"] + 1,
            last_episode=payload["episodes"],
        )
    elif event == "episode_ready":
        loguru.logger.info(
            "Episode {episode} ready | chronic={chronic} | grid step={step}/{max_step}",
            episode=payload["episode"] + 1,
            chronic=payload.get("chronic_name"),
            step=payload.get("grid2op_current_step"),
            max_step=payload.get("grid2op_max_step"),
        )
    elif event == "step_start":
        loguru.logger.info(
            "Collecting row {sample_index} | episode={episode} step={step} "
            "| grid step={grid_step} | max_rho={max_rho}",
            sample_index=payload["sample_index"],
            episode=payload["episode"] + 1,
            step=payload["step"],
            grid_step=payload.get("grid2op_current_step"),
            max_rho=payload.get("max_rho"),
        )
    elif event == "failure":
        loguru.logger.info(
            "Failure label=1 at row {sample_index} | episode={episode} step={step} "
            "| grid step={grid_step}/{max_step}",
            sample_index=payload["sample_index"],
            episode=payload["episode"] + 1,
            step=payload["step"],
            grid_step=payload.get("grid2op_current_step"),
            max_step=payload.get("grid2op_max_step"),
        )
    elif event == "episode_end":
        loguru.logger.info(
            "Finished episode {episode} | steps={steps} | total rows={total_rows} | done={done}",
            episode=payload["episode"] + 1,
            steps=payload["steps"],
            total_rows=payload["total_rows"],
            done=payload["done"],
        )


def collect(
    *,
    agent_name: str,
    episodes: int,
    seed: int,
    max_steps: int | None,
    rollout_dir: Path,
    output_path: Path,
    mode: str,
    agent_factory_spec: str | None,
    progress_every: int,
) -> None:
    configure_logging()
    configure_grid2op_warnings()

    if max_steps is not None and max_steps <= 0:
        max_steps = None
    if progress_every <= 0:
        progress_every = 25

    artifacts_ready = rollout_artifacts_available(rollout_dir)
    loguru.logger.info("Starting failure dataset collection")
    loguru.logger.info("Environment: {env_name} at {env_dir}", env_name=ENV_NAME, env_dir=ENV_DIR)
    loguru.logger.info("Agent: {agent_name}", agent_name=agent_name)
    loguru.logger.info("Mode: {mode}", mode=mode)
    loguru.logger.info("Episodes: {episodes} | max_steps: {max_steps}", episodes=episodes, max_steps=max_steps)
    loguru.logger.info("Rollout artifacts: {status} at {path}", status="found" if artifacts_ready else "not found", path=rollout_dir)
    loguru.logger.info("CSV output: {path}", path=output_path)
    if mode in {"replay", "features-only"} and not artifacts_ready:
        raise FileNotFoundError(
            f"Existing rollout artifacts were requested but not found under {rollout_dir}."
        )

    if mode == "features-only":
        loguru.logger.info("Exporting saved rollout observations only; no failure labels will be created")
        env = make_env()
        try:
            loguru.logger.info("Grid2Op environment created")
            example_obs = env.reset()
            loguru.logger.info("Example observation loaded for semantic feature names")
            rows = iter_artifact_feature_rows(rollout_dir, example_obs)
            loguru.logger.info("Writing CSV...")
            written = write_failure_dataset_csv(rows, output_path)
        finally:
            env.close()
        loguru.logger.success(
            "Wrote {written} feature-only rows to {path}. No failure label was invented.",
            written=written,
            path=output_path,
        )
        return

    loguru.logger.info("Creating Grid2Op environment...")
    env = make_env()
    loguru.logger.info("Grid2Op environment created")
    try:
        if mode in {"auto", "replay"} and artifacts_ready:
            try:
                loguru.logger.info("Trying replay path with saved rollout actions")
                rows = collect_replayed_failure_rows(
                    env,
                    rollout_dir,
                    episodes=episodes,
                    seed=seed,
                    max_steps=max_steps,
                    progress_callback=log_progress,
                    progress_every=progress_every,
                )
                loguru.logger.info("Replay collection finished; writing CSV...")
                written = write_failure_dataset_csv(rows, output_path)
                loguru.logger.success("Wrote {written} replay-labeled rows to {path}", written=written, path=output_path)
                return
            except ValueError as exc:
                if mode == "replay":
                    raise
                loguru.logger.warning(
                    "Artifact replay was not aligned; falling back to fresh collection: {error}",
                    error=exc,
                )
                env.close()
                loguru.logger.info("Recreating Grid2Op environment for fresh collection...")
                env = make_env()
                loguru.logger.info("Grid2Op environment recreated")

        spec = agent_factory_spec or AGENT_FACTORY or None
        if spec:
            loguru.logger.info("Building agent from factory: {spec}", spec=spec)
        else:
            loguru.logger.info("Building bundled CurriculumAgent from assets: {path}", path=_default_agent_path())
        agent = build_agent(
            env,
            factory_spec=spec,
            agent_path=None if spec else _default_agent_path(),
        )
        loguru.logger.info("Agent ready: {agent_type}", agent_type=type(agent).__name__)
        loguru.logger.info("Starting fresh environment rollout collection")
        rows = collect_fresh_failure_rows(
            env,
            agent,
            episodes=episodes,
            seed=seed,
            max_steps=max_steps,
            progress_callback=log_progress,
            progress_every=progress_every,
        )
        loguru.logger.info("Fresh collection finished; writing CSV...")
        written = write_failure_dataset_csv(rows, output_path)
        loguru.logger.success("Wrote {written} fresh-labeled rows to {path}", written=written, path=output_path)
    finally:
        loguru.logger.info("Closing Grid2Op environment")
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a simple CSV dataset for future failure prediction. "
            "By default this runs the configured agent and labels each timestep "
            "from the next env.step(...) result."
        ),
        epilog=(
            "Typical use: python training/collect_failure_dataset.py\n"
            "Output: artifacts/<ENV_NAME>/<AGENT_NAME>/failure_dataset/failure_dataset.csv\n"
            "Note: existing rollout labels are action classes, not failure labels."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    basic = parser.add_argument_group("common options")
    basic.add_argument("--episodes", type=int, default=ROLLOUT_EPISODES)
    basic.add_argument(
        "--max-steps",
        type=int,
        default=(ENN_ROLLOUT_MAX_STEPS or None),
        help="Maximum steps per episode. Use 0 to run until Grid2Op stops.",
    )
    basic.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path. By default it is written under artifacts/<env>/<agent>/failure_dataset/.",
    )
    basic.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print step progress every N steps inside each episode.",
    )

    advanced = parser.add_argument_group("advanced options")
    advanced.add_argument("--agent-name", default=AGENT_NAME)
    advanced.add_argument(
        "--agent-factory",
        default=AGENT_FACTORY or None,
        help="module:function; receives env and returns an agent",
    )
    advanced.add_argument("--seed", type=int, default=SEED)
    advanced.add_argument("--rollout-dir", type=Path, default=None)
    advanced.add_argument(
        "--mode",
        choices=("fresh", "auto", "replay", "features-only"),
        default="auto",
        help=(
            "fresh: normal safe run; auto: try saved rollout actions, then fall back to fresh; "
            "replay: require exact saved-rollout alignment; features-only: export saved "
            "observations without a failure label."
        ),
    )
    args = parser.parse_args()

    rollout_dir = args.rollout_dir or default_rollout_dir(args.agent_name)
    output_path = args.output or default_output_path(args.agent_name)
    collect(
        agent_name=args.agent_name,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        rollout_dir=rollout_dir,
        output_path=output_path,
        mode=args.mode,
        agent_factory_spec=args.agent_factory,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
