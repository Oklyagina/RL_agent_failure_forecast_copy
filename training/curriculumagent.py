"""
Creation of thee model whose  failure will be predcted on the next steps.
The environment is either default from grid2op, or one of available here: https://github.com/ainetus/grid2op-scenario
"""
import logging

from project_config import (ASSETS_DIR, CURRICULUM_ITERATIONS,
                            CURRICULUM_JOBS, ENV_DIR, ENV_LOCATION, ENV_NAME)

ENV_LOC = ENV_LOCATION
ENV_PATH = ENV_DIR
ITERATIONS = CURRICULUM_ITERATIONS
JOBS = CURRICULUM_JOBS
VERBOSE = False
SHOW_PROGRESS = True
LOG_LEVEL = logging.INFO if VERBOSE else logging.WARNING


def configure_logging() -> None:
    logging.basicConfig(level=LOG_LEVEL, force=True)
    logging.getLogger().setLevel(LOG_LEVEL)
    logging.disable(logging.INFO if not VERBOSE else logging.NOTSET)


configure_logging()

import grid2op
import loguru
import ray
from lightsim2grid import LightSimBackend

from curriculumagent.baseline import CurriculumAgent


def shutdown_ray() -> None:
    if ray.is_initialized():
        ray.shutdown()


def main() -> None:
    configure_logging()

    loguru.logger.info("Making environment....")
    env = grid2op.make(str(ENV_PATH), backend=LightSimBackend())
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
            iterations=ITERATIONS,
            save_path=ASSETS_DIR / ENV_NAME,
            jobs=JOBS,
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
