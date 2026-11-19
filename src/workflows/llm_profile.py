"""Load a combined LLM + pipeline JSON profile (``python main.py train --profile``).

The profile combines two kinds of settings:

* Top-level fields configure the OpenAI-compatible HTTP client.
* The nested ``pipeline`` object configures the ``train`` workflow.

Keeping the connection fields at the top level preserves compatibility with
``LLMConfig.from_json`` and the legacy ``dual_llm.py --llm-config`` command.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from project_config import get_optional_config
from src.explainability.llm_client import LLMConfig
from src.workflows.train import FORCE_STAGES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "src" / "llm_config.json"
VALID_FORCE_STAGES = frozenset(FORCE_STAGES)


class ProfileError(ValueError):
    """Report an invalid profile without including its API key."""


@dataclass(frozen=True)
class PipelineProfile:
    """Validated settings for the ``train`` workflow."""

    config_path: Path
    agent_name: str = "curriculum"
    llm_iterations: int = 5
    force_stages: tuple[str, ...] = ()
    reuse_forecast_data: bool = False
    force_shap_recalculation: bool = False
    smoke_line: str | None = None
    rules_dir: Path | None = None
    skip_smoke: bool = False
    verbose: bool = True


def _non_empty_string(
    values: Mapping[str, Any],
    name: str,
    label: str,
    default: str | None = None,
) -> str:
    """Return a required non-empty string or raise a clear profile error."""
    value = values.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{label} '{name}' must be a non-empty string")
    return value


def _boolean(
    values: Mapping[str, Any], name: str, default: bool, label: str = "Pipeline option"
) -> bool:
    """Read a strict JSON boolean; integers such as 0 and 1 are rejected."""
    value = values.get(name, default)
    if not isinstance(value, bool):
        raise ProfileError(f"{label} '{name}' must be a bool")
    return value


def _integer(
    values: Mapping[str, Any],
    name: str,
    default: int,
    minimum: int,
    label: str,
) -> int:
    """Read an integer with a minimum accepted value."""
    value = values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileError(f"{label} '{name}' must be an integer")
    if value < minimum:
        raise ProfileError(f"{label} '{name}' must be at least {minimum}")
    return value


def _number(
    values: Mapping[str, Any],
    name: str,
    default: float,
    minimum: float,
) -> float:
    """Read an integer or float connection setting with a lower bound."""
    value = values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"LLM configuration field '{name}' must be a number")
    if value < minimum:
        raise ProfileError(
            f"LLM configuration field '{name}' must be at least {minimum}"
        )
    return float(value)


def _validate_connection(values: Mapping[str, Any]) -> None:
    """Validate connection fields without exposing credential values."""
    _non_empty_string(values, "api_url", "LLM configuration field")
    _non_empty_string(values, "model", "LLM configuration field")

    api_key = values.get("api_key")
    api_key_env = values.get("api_key_env")
    if api_key is not None and (not isinstance(api_key, str) or not api_key):
        raise ProfileError(
            "LLM configuration field 'api_key' must be a non-empty string"
        )
    if api_key_env is not None and (
        not isinstance(api_key_env, str) or not api_key_env.strip()
    ):
        raise ProfileError(
            "LLM configuration field 'api_key_env' must be a non-empty string"
        )
    if not api_key and not api_key_env:
        raise ProfileError("LLM configuration requires 'api_key' or 'api_key_env'")
    if api_key_env and not api_key and not get_optional_config(api_key_env):
        raise ProfileError(
            "The configured LLM API key environment variable is not set"
        )

    _integer(
        values,
        "max_tokens",
        default=4000,
        minimum=1,
        label="LLM configuration field",
    )
    _integer(
        values,
        "retries",
        default=3,
        minimum=1,
        label="LLM configuration field",
    )
    _number(values, "timeout", default=180.0, minimum=0.001)
    _number(values, "retry_delay", default=3.0, minimum=0.0)
    _boolean(
        values,
        "verify_ssl",
        default=True,
        label="LLM configuration field",
    )
    if "enable_thinking" in values:
        _boolean(
            values,
            "enable_thinking",
            default=False,
            label="LLM configuration field",
        )

    # Use the same final parser as dual_llm.py so both entrypoints agree.
    try:
        LLMConfig.from_mapping(values)
    except (TypeError, ValueError) as exc:
        raise ProfileError("Invalid LLM connection configuration") from exc


def _optional_string(values: Mapping[str, Any], name: str) -> str | None:
    """Read a pipeline value that may be either null or a non-empty string."""
    value = values.get(name)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ProfileError(f"Pipeline option '{name}' must be null or a string")
    return value


def _resolve_rules_dir(value: str | None) -> Path | None:
    """Resolve a configured rules directory relative to the repository root."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_profile(path: str | Path = DEFAULT_CONFIG_PATH) -> PipelineProfile:
    """Load and validate a combined LLM connection and pipeline profile."""
    config_path = Path(path).expanduser().resolve()
    try:
        values = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileError(
            f"Malformed JSON in LLM pipeline profile at line {exc.lineno}, "
            f"column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise ProfileError(f"Cannot read LLM pipeline profile: {config_path}") from exc

    if not isinstance(values, dict):
        raise ProfileError("LLM pipeline profile must be a JSON object")
    _validate_connection(values)

    pipeline = values.get("pipeline", {})
    if not isinstance(pipeline, dict):
        raise ProfileError("Pipeline option 'pipeline' must be an object")

    agent_name = _non_empty_string(
        pipeline, "agent_name", "Pipeline option", default="curriculum"
    )
    llm_iterations = _integer(
        pipeline,
        "llm_iterations",
        default=5,
        minimum=1,
        label="Pipeline option",
    )

    force_stages = pipeline.get("force_stages", [])
    if not isinstance(force_stages, list) or any(
        not isinstance(stage, str) for stage in force_stages
    ):
        raise ProfileError("Pipeline option 'force_stages' must be a list of strings")
    unknown_stages = sorted(set(force_stages) - VALID_FORCE_STAGES)
    if unknown_stages:
        raise ProfileError(
            "Pipeline option 'force_stages' contains unknown stages: "
            + ", ".join(unknown_stages)
        )

    smoke_line = _optional_string(pipeline, "smoke_line")
    rules_dir = _resolve_rules_dir(_optional_string(pipeline, "rules_dir"))

    return PipelineProfile(
        config_path=config_path,
        agent_name=agent_name,
        llm_iterations=llm_iterations,
        force_stages=tuple(force_stages),
        reuse_forecast_data=_boolean(pipeline, "reuse_forecast_data", False),
        force_shap_recalculation=_boolean(
            pipeline, "force_shap_recalculation", False
        ),
        smoke_line=smoke_line,
        rules_dir=rules_dir,
        skip_smoke=_boolean(pipeline, "skip_smoke", False),
        verbose=_boolean(pipeline, "verbose", True),
    )


def profile_to_pipeline_args(profile: PipelineProfile) -> argparse.Namespace:
    """Translate profile names to the ``train`` argument names."""
    return argparse.Namespace(
        agent_name=profile.agent_name,
        force_stage=list(profile.force_stages),
        reuse_forecast_data=profile.reuse_forecast_data,
        smoke_line=profile.smoke_line,
        explain_rules_dir=profile.rules_dir,
        with_explainability=True,
        llm_config=profile.config_path,
        llm_iterations=profile.llm_iterations,
        force_shap_recalculation=profile.force_shap_recalculation,
        skip_smoke=profile.skip_smoke,
        verbose=profile.verbose,
    )

