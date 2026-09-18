"""Smoke-run one failure prediction from configured Grid2Op artifacts."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import grid2op
import joblib
from lightsim2grid import LightSimBackend

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
from src.agent_runtime import build_agent, call_agent  # noqa: E402
from src.failure_forecast import FailureForecastConfig, FailureForecastPredictor  # noqa: E402


def default_dir(agent_name: str) -> Path:
    """Return the default failure-forecast artifact directory."""
    return ARTIFACTS_DIR / ENV_NAME / agent_name / "failure_forecast"


def _default_agent_path() -> Path:
    """Find the runnable CurriculumAgent package for smoke predictions."""
    for path in (ASSETS_DIR / ENV_NAME, ASSETS_DIR / "network36"):
        if (path / "model").is_dir() and (path / "actions").is_dir():
            return path
    return ASSETS_DIR / ENV_NAME


def configure_quiet_logging() -> None:
    """Suppress verbose policy INFO logs during the smoke rollout."""
    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("curriculumagent").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-name", default=AGENT_NAME)
    parser.add_argument("--agent-factory", default=AGENT_FACTORY or None)
    parser.add_argument("--line", required=True)
    parser.add_argument("--step", type=int, default=12)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mean-model", type=Path, required=True)
    parser.add_argument("--aleatoric-model", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, default=None)
    args = parser.parse_args()

    configure_grid2op_warnings()
    configure_quiet_logging()
    out_dir = default_dir(args.agent_name)
    predictor = FailureForecastPredictor.load(
        args.classifier or (out_dir / "failure_classifier.pkl")
    )
    mean_model = joblib.load(args.mean_model)
    aleatoric_model = joblib.load(args.aleatoric_model)

    env = grid2op.make(str(ENV_DIR), backend=LightSimBackend())
    try:
        spec = args.agent_factory or None
        agent = build_agent(
            env,
            factory_spec=spec,
            agent_path=None if spec else _default_agent_path(),
        )
        obs = env.reset(seed=args.seed)
        observations = []
        reward = getattr(env, "reward_range", (0.0, 0.0))[0]
        done = False
        for _ in range(max(0, args.step)):
            observations.append(obs)
            action = call_agent(agent, obs, reward=float(reward), done=done)
            obs, reward, done, _ = env.step(action)
            if done:
                raise RuntimeError("Episode ended before the requested smoke step.")
        observations.append(obs)
        cfg = FailureForecastConfig.from_env(
            env,
            env_name=ENV_NAME,
            agent_name=args.agent_name,
            artifact_dir=out_dir,
            lines_to_test=[args.line],
        )
        print(predictor.predict(
            env, agent, obs, observations, args.line,
            mean_model, aleatoric_model, cfg,
        ))
    finally:
        env.close()


if __name__ == "__main__":
    main()
