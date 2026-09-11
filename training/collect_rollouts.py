"""Collect behavior-cloning rollouts for ENN training.

The collector is policy-agnostic.  By default it loads the bundled agent assets;
for any other policy set ``AGENT_FACTORY=package.module:function`` or pass
``--agent-factory``.  The factory receives the Grid2Op environment and returns
an object exposing ``act``.
"""
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

from project_config import (
    AGENT_FACTORY, AGENT_NAME, ARTIFACTS_DIR, ASSETS_DIR, ENV_DIR, ENV_NAME,
    ENN_ROLLOUT_MAX_STEPS, ROLLOUT_EPISODES, SEED,
    configure_grid2op_warnings,
)
from src.agent_runtime import build_agent
from src.enn_data import collect_policy_rollouts, save_rollout_bundle


def default_rollout_dir(agent_name: str) -> Path:
    return ARTIFACTS_DIR / ENV_NAME / agent_name / "rollouts"


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
    """Keep rollout collection output focused on warnings and progress."""
    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger().setLevel(logging.WARNING)


def collect(
    agent_name: str,
    episodes: int,
    out_dir: Path,
    seed: int = 0,
    max_steps: int | None = None,
    agent_factory_spec: str | None = None,
) -> None:
    configure_logging()
    configure_grid2op_warnings()
    env = grid2op.make(str(ENV_DIR), backend=LightSimBackend())
    progress = tqdm(
        total=episodes,
        desc="Collecting rollouts",
        unit="episode",
        dynamic_ncols=True,
    )

    def update_progress(_episode: int, steps: int, total_pairs: int) -> None:
        progress.update(1)
        progress.set_postfix(
            last_steps=steps,
            total_pairs=total_pairs,
            refresh=False,
        )

    try:
        spec = agent_factory_spec or AGENT_FACTORY or None
        agent = build_agent(
            env,
            factory_spec=spec,
            agent_path=None if spec else _default_agent_path(),
        )
        observations, labels, action_set = collect_policy_rollouts(
            env,
            agent,
            episodes=episodes,
            seed=seed,
            max_steps=max_steps,
            progress_callback=update_progress,
        )
    finally:
        progress.close()
        env.close()

    save_rollout_bundle(out_dir, observations, labels, action_set, seed=seed)
    print(
        f"[ok] {len(observations)} observation/action pairs | "
        f"{len(action_set)} distinct actions -> {out_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-name", default=AGENT_NAME)
    parser.add_argument("--agent-factory", default=AGENT_FACTORY or None,
                        help="module:function; receives env and returns an agent")
    parser.add_argument("--episodes", type=int, default=ROLLOUT_EPISODES)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-steps", type=int,
                        default=(ENN_ROLLOUT_MAX_STEPS or None))
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = default_rollout_dir(args.agent_name)
    collect(
        args.agent_name, args.episodes, args.out_dir, args.seed,
        args.max_steps, args.agent_factory,
    )
