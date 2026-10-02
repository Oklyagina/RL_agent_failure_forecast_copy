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
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - project requirements include tqdm
    class tqdm:
        def __init__(self, total=None, desc=None, unit=None, dynamic_ncols=True):
            del unit, dynamic_ncols
            self.total = total
            self.count = 0
            self.desc = desc or "progress"
            print(f"{self.desc}: 0/{self.total or '?'}")

        def update(self, n=1):
            self.count += n
            print(f"\r{self.desc}: {self.count}/{self.total or '?'}", end="")

        def set_postfix(self, **kwargs):
            del kwargs

        def write(self, message):
            print(f"\n{message}")

        def close(self):
            print()

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


class RolloutProgress:
    """Show compact per-episode step progress with tqdm."""

    def __init__(self, *, max_steps: int | None):
        self.max_steps = max_steps
        self.bar = None
        self.episodes = None

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def __call__(self, event: str, payload: dict) -> None:
        if event == "episode_start":
            self.episodes = payload.get("episodes")
        elif event == "episode_ready":
            self.close()
            total = self.max_steps or payload.get("grid2op_max_step")
            episode = payload["episode"] + 1
            if self.episodes:
                desc = f"episode {episode}/{self.episodes}"
            else:
                desc = f"episode {episode}"
            self.bar = tqdm(total=total, desc=desc, unit="step", dynamic_ncols=True)
        elif event == "step_done":
            if self.bar is not None:
                self.bar.update(1)
                self.bar.set_postfix(step=payload.get("step"), rows=payload.get("total_rows"), refresh=False)
        elif event == "failure":
            if self.bar is not None:
                self.bar.write(
                    f"failure=1 at episode {payload['episode'] + 1}, "
                    f"step {payload['step']}"
                )
        elif event == "episode_end":
            self.close()


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
) -> None:
    configure_logging()
    configure_grid2op_warnings()

    if max_steps is not None and max_steps <= 0:
        max_steps = None
    artifacts_ready = rollout_artifacts_available(rollout_dir)
    loguru.logger.info(
        "Starting failure dataset collection | env={env_name} | agent={agent_name} | mode={mode} | episodes={episodes}",
        env_name=ENV_NAME,
        agent_name=agent_name,
        mode=mode,
        episodes=episodes,
    )
    loguru.logger.info("CSV output: {path}", path=output_path)
    if mode in {"replay", "features-only"} and not artifacts_ready:
        raise FileNotFoundError(
            f"Existing rollout artifacts were requested but not found under {rollout_dir}."
        )

    if mode == "features-only":
        loguru.logger.info("Exporting saved rollout observations only; no failure labels will be created")
        env = make_env()
        try:
            example_obs = env.reset()
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

    env = make_env()
    progress = RolloutProgress(max_steps=max_steps)
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
                    progress_callback=progress,
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
                progress.close()
                env.close()
                env = make_env()
                progress = RolloutProgress(max_steps=max_steps)

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
            progress_callback=progress,
        )
        loguru.logger.info("Fresh collection finished; writing CSV...")
        written = write_failure_dataset_csv(rows, output_path)
        loguru.logger.success("Wrote {written} fresh-labeled rows to {path}", written=written, path=output_path)
    finally:
        progress.close()
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
    )


if __name__ == "__main__":
    main()
