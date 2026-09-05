"""
Creation of the model whose failure will be predicted on the next steps.
The environment is either default from grid2op, or one of those available here: https://github.com/ainetus/grid2op-scenario
"""
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grid2op
import loguru
import ray
from lightsim2grid import LightSimBackend
from curriculumagent.baseline import CurriculumAgent

from project_config import (ASSETS_DIR, CURRICULUM_ITERATIONS,
                            CURRICULUM_JOBS, ENV_DIR, ENV_NAME)

VERBOSE = False
SHOW_PROGRESS = True
LOG_LEVEL = logging.INFO if VERBOSE else logging.WARNING

def configure_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"You are using a legacy grid2op version, please upgrade grid2op\.",
        category=UserWarning,
        module=r"lightsim2grid\.lightSimBackend",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"There were some Nan in the pp_net\.trafo\[\"tap_step_degree\"\], they have been replaced by 0",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"We found either some slack coefficient to be < 0\. or they were all 0\.We set them all to 1\.0 to avoid such issues",
        category=UserWarning,
        module=r"lightsim2grid\.gridmodel\.from_pandapower\._aux_add_slack",
    )
    for message in (
        r"`UnifiedLogger` will be removed in Ray 2\.7\.",
        r"The `JsonLogger interface is deprecated in favor of the `ray\.tune\.json\.JsonLoggerCallback` interface and will be removed in Ray 2\.7\.",
        r"The `CSVLogger interface is deprecated in favor of the `ray\.tune\.csv\.CSVLoggerCallback` interface and will be removed in Ray 2\.7\.",
        r"The `TBXLogger interface is deprecated in favor of the `ray\.tune\.tensorboardx\.TBXLoggerCallback` interface and will be removed in Ray 2\.7\.",
    ):
        warnings.filterwarnings(
            "ignore",
            message=message,
            category=DeprecationWarning,
            module=r"ray\..*",
        )

def configure_logging() -> None:
    logging.basicConfig(level=LOG_LEVEL, force=True)
    logging.getLogger().setLevel(LOG_LEVEL)
    logging.disable(logging.INFO if not VERBOSE else logging.NOTSET)

def shutdown_ray() -> None:
    if ray.is_initialized():
        ray.shutdown()

configure_logging()
configure_warnings()


def main() -> None:
    configure_logging()
    configure_warnings()

    loguru.logger.info("Making environment....")
    env = grid2op.make(str(ENV_DIR), backend=LightSimBackend())
    loguru.logger.success("Environment created!")

    loguru.logger.info("Initialising the agent....")
    agent = CurriculumAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        name=ENV_NAME,
    )
    loguru.logger.success("Initialised!")

    try:
        loguru.logger.info("Started training....")
        agent.train_full_pipeline(
            env=env,
            name=ENV_NAME,
            iterations=CURRICULUM_ITERATIONS,
            save_path=ASSETS_DIR / ENV_NAME,
            jobs=CURRICULUM_JOBS,
            log_level=LOG_LEVEL,
            show_progress=SHOW_PROGRESS,
        )
    except KeyboardInterrupt:
        loguru.logger.warning("Interrupted by user; shutting down Ray...")
        raise SystemExit(130)
    finally:
        shutdown_ray()

    loguru.logger.success("Success!")


if __name__ == "__main__":
    main()
