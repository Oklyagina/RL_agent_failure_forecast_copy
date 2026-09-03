"""
Creation of the model whose failure will be predicted on the next steps.
The environment is either default from grid2op, or one of those available here: https://github.com/ainetus/grid2op-scenario
"""
import logging
import sys
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

def configure_logging() -> None:
    logging.basicConfig(level=LOG_LEVEL, force=True)
    logging.getLogger().setLevel(LOG_LEVEL)
    logging.disable(logging.INFO if not VERBOSE else logging.NOTSET)

def shutdown_ray() -> None:
    if ray.is_initialized():
        ray.shutdown()

configure_logging()


def main() -> None:
    configure_logging()

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
