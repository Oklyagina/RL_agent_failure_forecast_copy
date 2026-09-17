"""Collect one-hour-ahead line-contingency failure-forecast rows."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import grid2op
import joblib
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
    FailureForecastConfig,
    collect_failure_forecast_rows,
    write_rows_csv,
)


def default_output_dir(agent_name: str) -> Path:
    """Return the standard artifact folder for collected failure rows."""
    return ARTIFACTS_DIR / ENV_NAME / agent_name / "failure_forecast"


def _default_agent_path() -> Path:
    """Find the runnable CurriculumAgent package used when no factory is set."""
    for path in (ASSETS_DIR / ENV_NAME, ASSETS_DIR / "network36"):
        if (path / "model").is_dir() and (path / "actions").is_dir():
            return path
    return ASSETS_DIR / ENV_NAME


def _parse_lines(raw: str | None, env) -> list:
    """Parse CLI line ids/names, defaulting to every line known by Grid2Op."""
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    names = getattr(env, "name_line", None)
    if names is None:
        raise ValueError("--lines is required when env.name_line is unavailable.")
    return list(names)


def configure_quiet_logging() -> None:
    """Suppress verbose policy INFO logs while keeping warnings visible."""
    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("curriculumagent").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-name", default=AGENT_NAME)
    parser.add_argument("--agent-factory", default=AGENT_FACTORY or None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sampling-stride", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lines", default=None,
                        help="comma-separated line names/ids; default: all env lines")
    parser.add_argument("--mean-model", type=Path, required=True)
    parser.add_argument("--aleatoric-model", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    configure_grid2op_warnings()
    configure_quiet_logging()
    env = grid2op.make(str(ENV_DIR), backend=LightSimBackend())
    progress = tqdm(total=args.episodes, desc="Collecting failure rows", unit="episode")
    try:
        spec = args.agent_factory or None
        agent = build_agent(
            env,
            factory_spec=spec,
            agent_path=None if spec else _default_agent_path(),
        )
        lines = _parse_lines(args.lines, env)
        out_dir = default_output_dir(args.agent_name)
        cfg = FailureForecastConfig.from_env(
            env,
            env_name=ENV_NAME,
            agent_name=args.agent_name,
            artifact_dir=out_dir,
            lines_to_test=lines,
        )
        mean_model = joblib.load(args.mean_model)
        aleatoric_model = joblib.load(args.aleatoric_model)

        def update(_episode: int, steps: int, total_rows: int) -> None:
            progress.update(1)
            progress.set_postfix(last_steps=steps, total_rows=total_rows, refresh=False)

        rows = collect_failure_forecast_rows(
            env,
            agent,
            mean_model,
            aleatoric_model,
            cfg,
            episodes=args.episodes,
            seed=args.seed,
            sampling_stride=args.sampling_stride,
            max_steps=args.max_steps,
            progress_callback=update,
        )
    finally:
        progress.close()
        env.close()

    output_csv = args.output_csv or (default_output_dir(args.agent_name) / "failure_forecast_rows.csv")
    count = write_rows_csv(rows, output_csv)
    print(f"[ok] wrote {count} rows -> {output_csv}")


if __name__ == "__main__":
    main()
