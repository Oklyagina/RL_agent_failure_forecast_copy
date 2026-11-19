"""Collect labeled forced-disconnection failure rows (legacy procedure, t+12)."""
from __future__ import annotations

import sys
from pathlib import Path

import grid2op
import joblib
from lightsim2grid import LightSimBackend
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import project_config as config  # noqa: E402
from src.agent_runtime import build_agent  # noqa: E402
from src.failure_forecast import (  # noqa: E402
    LEGACY_PHASES,
    FailureForecastConfig,
    collect_failure_forecast_phases,
    write_rows_csv,
)
from src.pipeline_support import (  # noqa: E402
    configure_quiet_logging,
    failure_dir,
    parse_lines,
    resolve_policy_dir,
)
from src.runtime_artifacts import load_enn_uncertainty  # noqa: E402


def default_output_dir(agent_name: str) -> Path:
    """Return the standard artifact folder for collected failure rows."""
    return failure_dir(config.ARTIFACTS_DIR, config.ENV_NAME, agent_name)


def configured_output_dir() -> Path:
    """Return the configured failure-forecast artifact directory."""
    return config.FAILURE_FORECAST_DIR or default_output_dir(config.AGENT_NAME)


def _default_agent_path() -> Path:
    """Find the runnable CurriculumAgent package used when no factory is set."""
    return resolve_policy_dir(config.ASSETS_DIR, config.ENV_NAME)


def main() -> None:
    config.configure_grid2op_warnings()
    configure_quiet_logging()
    env = grid2op.make(str(config.ENV_DIR), backend=LightSimBackend())
    progress = tqdm(
        total=config.FAILURE_EPISODES * len(LEGACY_PHASES),
        desc="Collecting failure rows",
        unit="episode",
    )
    try:
        spec = config.AGENT_FACTORY or None
        agent = build_agent(
            env,
            factory_spec=spec,
            agent_path=None if spec else _default_agent_path(),
        )
        lines = parse_lines(config.FAILURE_LINES or None, env)
        out_dir = configured_output_dir()
        cfg = FailureForecastConfig.from_env(
            env,
            env_name=config.ENV_NAME,
            agent_name=config.AGENT_NAME,
            artifact_dir=out_dir,
            lines_to_test=lines,
        )
        mean_model = joblib.load(out_dir / "mean_forecaster.pkl")
        aleatoric_model = joblib.load(out_dir / "aleatoric_forecaster.pkl")
        enn, uncertainty_fn = load_enn_uncertainty()

        def update(_episode: int, steps: int, stats) -> None:
            progress.n = stats["episodes"]
            progress.set_postfix(
                last_steps=steps,
                samples=stats["rows"],
                pos=stats["positive"],
                neg=stats["negative"],
                points=stats["sample_points"],
                refresh=False,
            )
            progress.refresh()

        def make_env(**kwargs):
            return grid2op.make(str(config.ENV_DIR), backend=LightSimBackend(), **kwargs)

        rows, stats = collect_failure_forecast_phases(
            agent,
            make_env,
            mean_model,
            aleatoric_model,
            cfg,
            episodes=config.FAILURE_EPISODES,
            seed=config.FAILURE_SEED_START,
            max_steps=config.FAILURE_MAX_STEPS or None,
            model_enn=enn,
            get_uncertainty_fn=uncertainty_fn,
            progress_callback=update,
        )
    finally:
        progress.close()
        env.close()

    print("[labels] " + ", ".join(f"{key}={value}" for key, value in sorted(stats.items())))
    if stats["positive"] == 0 or stats["negative"] == 0:
        raise SystemExit(
            "Need both failures (1) and survivals (0) after the forced line "
            "disconnection; collect more episodes or steps."
        )
    output_csv = configured_output_dir() / "failure_forecast_rows.csv"
    count = write_rows_csv(rows, output_csv)
    print(f"[ok] wrote {count} rows -> {output_csv}")


if __name__ == "__main__":
    main()
