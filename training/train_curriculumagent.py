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
                            CURRICULUM_JOBS,
                            CURRICULUM_TUTOR_BEST_ACTION_THRESHOLD,
                            CURRICULUM_TUTOR_DO_NOTHING_THRESHOLD,
                            CURRICULUM_TUTOR_MIN_UNIQUE_ROWS,
                            configure_grid2op_warnings, ENV_DIR,
                            ENV_NAME)

VERBOSE = False
SHOW_PROGRESS = True
LOG_LEVEL = logging.INFO if VERBOSE else logging.WARNING
warnings.filterwarnings("ignore", category=DeprecationWarning)
SUPPRESSED_LOG_MESSAGES = (
    "Your env doesn't have a .spec.max_episode_steps attribute.",
    "You have specified 1 evaluation workers, but your `evaluation_interval` is None!",
)


class _TrainingNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(suppressed in message for suppressed in SUPPRESSED_LOG_MESSAGES)


TRAINING_NOISE_FILTER = _TrainingNoiseFilter()


def configure_warnings() -> None:
    configure_grid2op_warnings()
    warning_categories = [DeprecationWarning]
    try:
        from ray.rllib.utils.deprecation import RayDeprecationWarning

        warning_categories.append(RayDeprecationWarning)
    except Exception:
        pass

    warnings.filterwarnings(
        "ignore",
        message=r"The custom dictionary did not have the correct keys\. Using default model\.",
        category=UserWarning,
        module=r"curriculumagent\.senior\.rllib_execution\.senior_model_rllib",
    )
    for message in (
        r".*UnifiedLogger.*will be removed in Ray 2\.7\.",
        r".*JsonLogger interface is deprecated.*will be removed in Ray 2\.7\.",
        r".*CSVLogger interface is deprecated.*will be removed in Ray 2\.7\.",
        r".*TBXLogger interface is deprecated.*will be removed in Ray 2\.7\.",
    ):
        for category in warning_categories:
            warnings.filterwarnings(
                "ignore",
                message=message,
                category=category,
                module=r"ray\..*",
            )

def configure_logging() -> None:
    logging.basicConfig(level=LOG_LEVEL, force=True)
    logging.getLogger().setLevel(LOG_LEVEL)
    for handler in logging.getLogger().handlers:
        handler.addFilter(TRAINING_NOISE_FILTER)
    for logger_name in (
        "ray.rllib.env.env_context",
        "ray.rllib.algorithms.algorithm_config",
    ):
        logging.getLogger(logger_name).addFilter(TRAINING_NOISE_FILTER)
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
            tutor_do_nothing_threshold=CURRICULUM_TUTOR_DO_NOTHING_THRESHOLD,
            tutor_best_action_threshold=CURRICULUM_TUTOR_BEST_ACTION_THRESHOLD,
            min_unique_tutor_rows=CURRICULUM_TUTOR_MIN_UNIQUE_ROWS,
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
