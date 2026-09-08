"""
app/main.py -- FastAPI wrapper exposing the CurriculumAgent as an InteractiveAI
agent API, following the AI4REALNET AI-agent template.

Each recommendation is returned in the InteractiveAI dictionary format
(title / description / use_case / agent_type / actions / kpis), and the two
ENN epistemic-uncertainty percentiles are added to the "kpis" field, alongside
"efficiency_of_the_reco".

Endpoint (same contract as the template):
    POST /api/v1/recommendation
        body: {"event": ..., "context": {..., "observation": <grid2op obs>}}
        ->   [ {"title", "description", "use_case", "agent_type",
                "actions": [...], "kpis": {...}}, ... ]

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

The environment must match the InteractiveAI simulator's Grid2Op version and
scenario. The API uses .env/environment configuration from project_config.py
(see Dockerfile and API.md).
"""
import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="CurriculumAgent + ENN uncertainty API")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for path in (ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from project_config import AGENT_NAME, ARTIFACTS_DIR, ASSETS_DIR, ENV_DIR, ENV_NAME

os.environ.setdefault("GRID2OP_DATA_PATH", str(ENV_DIR.parent))


# --------------------------------------------------------------------------- #
#  Request model (mirrors the template's RecommendationRequest)
# --------------------------------------------------------------------------- #
class RecommendationRequest(BaseModel):
    event: Optional[Dict[str, Any]] = None
    context: Dict[str, Any]


# --------------------------------------------------------------------------- #
#  API errors
# --------------------------------------------------------------------------- #
class ApiConfigurationError(Exception):
    """Raised when the API cannot load a consistent model/artifact bundle."""

    def __init__(
        self,
        *,
        error: str,
        stage: str,
        message: str,
        status_code: int = 503,
        hint: Optional[str] = None,
        **details: Any,
    ):
        super().__init__(message)
        self.error = error
        self.stage = stage
        self.message = message
        self.status_code = status_code
        self.hint = hint
        self.details = details

    def to_detail(self) -> Dict[str, Any]:
        payload = {
            "error": self.error,
            "stage": self.stage,
            "message": self.message,
        }
        payload.update(self.details)
        if self.hint:
            payload["hint"] = self.hint
        return payload


class ApiPayloadError(Exception):
    """Raised when the request context cannot be converted to an observation."""

    def __init__(self, *, stage: str, message: str, **details: Any):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.details = details

    def to_detail(self) -> Dict[str, Any]:
        return {
            "error": "bad_observation_payload",
            "stage": self.stage,
            "message": self.message,
            **self.details,
        }


class ApiRuntimeError(Exception):
    """Raised when the loaded service fails while computing a recommendation."""

    def __init__(self, *, stage: str, message: str, **details: Any):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.details = details

    def to_detail(self) -> Dict[str, Any]:
        return {
            "error": "recommendation_runtime_error",
            "stage": self.stage,
            "message": self.message,
            **self.details,
        }


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _file_info(path: Path) -> Dict[str, Any]:
    exists = path.is_file()
    return {
        "path": _display_path(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
    }


def _read_json(path: Path, stage: str) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ApiConfigurationError(
            error="artifact_read_error",
            stage=stage,
            message=f"Could not read JSON artifact: {_display_path(path)}",
            path=_display_path(path),
            exception_type=type(exc).__name__,
            exception=str(exc),
        ) from exc


def _has_valid_saved_model(agent_dir: Path) -> bool:
    model_dir = agent_dir / "model"
    variables_dir = model_dir / "variables"
    required_files = [
        model_dir / "saved_model.pb",
        variables_dir / "variables.index",
        variables_dir / "variables.data-00000-of-00001",
    ]
    return all(p.is_file() and p.stat().st_size > 0 for p in required_files)


def _find_agent_dir() -> Path:
    candidates = [
        ASSETS_DIR / ENV_NAME,
        ASSETS_DIR / "network36",
    ]
    checked = []
    for candidate in candidates:
        checked.append(_display_path(candidate))
        if (candidate / "model").is_dir() and (candidate / "actions").is_dir():
            if _has_valid_saved_model(candidate):
                return candidate
    raise ApiConfigurationError(
        error="artifact_missing",
        stage="find_agent_dir",
        message="No valid CurriculumAgent directory was found.",
        checked_paths=checked,
        hint="Expected model/ and actions/ with a non-empty TensorFlow "
             "SavedModel under assets/<ENV_NAME>/, assets/network36/, or "
             "another active agent directory.",
    )


def _api_artifact_paths() -> Dict[str, Path]:
    """Select one consistent API artifact bundle.

    The API intentionally selects the refactored ENN outputs as one consistent
    bundle so checkpoint, metadata, scaler, calibration, and actions match.
    """
    model_dir = ARTIFACTS_DIR / ENV_NAME / AGENT_NAME / "model"
    rollouts_dir = ARTIFACTS_DIR / ENV_NAME / AGENT_NAME / "rollouts"
    return {
        "metadata": model_dir / "enn_meta.json",
        "weights": model_dir / f"enn_{AGENT_NAME}.pth",
        "scaler": model_dir / "scaler_params.json",
        "calibration": model_dir / "enn_pctile_calib.npz",
        "actions": rollouts_dir / "actions.npy",
    }


def _validate_api_artifacts() -> Dict[str, Any]:
    paths = _api_artifact_paths()
    missing = [
        name for name, path in paths.items()
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise ApiConfigurationError(
            error="artifact_missing",
            stage="select_artifacts",
            message="The selected API artifact bundle is incomplete.",
            selected_paths={name: _file_info(path) for name, path in paths.items()},
            missing=missing,
            hint="The FastAPI service expects the refactored ENN bundle: "
                 "enn_<AGENT_NAME>.pth, enn_meta.json, scaler_params.json, "
                 "enn_pctile_calib.npz, and rollouts/actions.npy.",
        )

    meta = _read_json(paths["metadata"], "read_metadata")
    scaler_params = _read_json(paths["scaler"], "read_scaler")
    try:
        input_dim = int(meta["input_dim"])
        num_classes = int(meta["num_classes"])
        scaler_features = int(scaler_params["n_features_in"])
    except KeyError as exc:
        raise ApiConfigurationError(
            error="artifact_compatibility_error",
            stage="validate_artifacts",
            message=f"Required artifact metadata key is missing: {exc}",
            metadata_path=_display_path(paths["metadata"]),
            scaler_path=_display_path(paths["scaler"]),
        ) from exc

    if scaler_features != input_dim:
        raise ApiConfigurationError(
            error="artifact_compatibility_error",
            stage="validate_artifacts",
            message="Scaler feature count does not match ENN input dimension.",
            metadata_path=_display_path(paths["metadata"]),
            scaler_path=_display_path(paths["scaler"]),
            metadata_input_dim=input_dim,
            scaler_n_features_in=scaler_features,
        )

    try:
        actions = np.load(paths["actions"], mmap_mode="r")
        action_shape = tuple(int(v) for v in actions.shape)
    except Exception as exc:
        raise ApiConfigurationError(
            error="artifact_read_error",
            stage="read_actions",
            message=f"Could not read action set: {_display_path(paths['actions'])}",
            actions_path=_display_path(paths["actions"]),
            exception_type=type(exc).__name__,
            exception=str(exc),
        ) from exc

    if len(action_shape) != 2:
        raise ApiConfigurationError(
            error="artifact_compatibility_error",
            stage="validate_artifacts",
            message="Action set must be a 2-D NumPy array.",
            actions_path=_display_path(paths["actions"]),
            actions_shape=action_shape,
        )

    if action_shape[0] != num_classes:
        raise ApiConfigurationError(
            error="artifact_compatibility_error",
            stage="validate_artifacts",
            message="Action-set row count does not match ENN class count.",
            metadata_path=_display_path(paths["metadata"]),
            actions_path=_display_path(paths["actions"]),
            metadata_num_classes=num_classes,
            actions_rows=action_shape[0],
            hint="Use an actions.npy generated with the same ENN training run "
                 "as enn_meta.json and enn_<AGENT_NAME>.pth.",
        )

    return {
        "paths": paths,
        "metadata": meta,
        "scaler_params": scaler_params,
        "action_shape": action_shape,
    }


# --------------------------------------------------------------------------- #
#  Services: env + agent + ENN + calibration, loaded once (lazy, cached)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_services():
    import grid2op
    from lightsim2grid import LightSimBackend
    import run_example as rx                    # reuse the auto-discovery
    from recommendation_uncertainty import load_calibration

    env = grid2op.make(str(ENV_DIR), backend=LightSimBackend())
    bundle = _validate_api_artifacts()
    paths = bundle["paths"]
    meta = bundle["metadata"]
    scaler_json = paths["scaler"]
    meta_json = paths["metadata"]
    npz = paths["calibration"]
    weights = paths["weights"]
    actions_path = paths["actions"]
    agent_dir = _find_agent_dir()

    agent = rx.load_agent(env, agent_dir)
    try:
        enn = rx.load_enn(weights, meta)
    except RuntimeError as exc:
        raise ApiConfigurationError(
            error="artifact_compatibility_error",
            stage="load_enn",
            message="Selected ENN metadata and checkpoint are incompatible.",
            metadata_path=_display_path(meta_json),
            weights_path=_display_path(weights),
            metadata_input_dim=meta.get("input_dim"),
            metadata_num_classes=meta.get("num_classes"),
            exception_type=type(exc).__name__,
            exception=str(exc),
            hint="Use enn_<AGENT_NAME>.pth with the matching enn_meta.json "
                 "from the same refactored training run.",
        ) from exc
    calibration = load_calibration(
        str(npz), scaler=rx.scaler_from_json(scaler_json),
        action_set=str(actions_path), class_mapping=str(meta_json))
    return env, agent, enn, calibration


# --------------------------------------------------------------------------- #
#  Recommendation formatting
# --------------------------------------------------------------------------- #
def _base_reco_dict(action, obs) -> dict:
    """Base InteractiveAI recommendation dict for one action.

    Prefer the ExpertAgent-side helper get_parade_info(action, obs) -- which
    also computes efficiency_of_the_reco and the human-readable description. It
    is provided by ExpertOp4Grid (installed in the ExpertAgent container); wire
    its exact import below, or drop the helper into the repo as app/parade.py.
    Otherwise we return a minimal dict with efficiency_of_the_reco left null for
    the platform to fill; the uncertainty percentiles are added on top either way.
    """
    get_parade_info = None
    for mod in ("app.parade", "parade", "expertop4grid", "ExpertOp4Grid"):
        try:
            get_parade_info = __import__(mod, fromlist=["get_parade_info"]) \
                .get_parade_info
            break
        except Exception:
            continue
    if get_parade_info is not None:
        d = get_parade_info(action, obs)
        return d[0] if isinstance(d, list) else d
    return {
        "title": "Topological recommendation (CurriculumAgent)",
        "description": str(action),
        "use_case": "PowerGrid",
        "agent_type": 2,
        "actions": [action.as_serializable_dict()],
        "kpis": {"type_of_the_reco": "Topological",
                 "efficiency_of_the_reco": None},
    }


def _merge_uncertainty(reco: dict, info: dict) -> dict:
    """Add ENN epistemic-uncertainty KPIs into kpis."""
    reco.setdefault("kpis", {})
    reco["kpis"]["epistemic_uncertainty_pct"] = \
        info["epistemic_uncertainty_pct"]
    reco["kpis"]["epistemic_uncertainty_total_pctile"] = \
        info["epistemic_uncertainty_total_pctile"]
    reco["kpis"]["epistemic_uncertainty_action_pctile"] = \
        info["epistemic_uncertainty_action_pctile"]
    reco["kpis"]["epistemic_uncertainty_level"] = \
        info["epistemic_uncertainty_level"]
    reco["kpis"]["epistemic_confidence_level"] = \
        info["epistemic_confidence_level"]
    return reco


def _load_observation(env, context: dict):
    """Rebuild a Grid2Op observation from the incoming context, following the
    template (context["observation"]). Falls back to from_vect if the payload
    is a plain vector."""
    obs = env.reset()
    payload = context.get("observation")
    if payload is None:
        return obs
    if hasattr(obs, "from_json"):
        try:
            obs.from_json(payload)
            return obs
        except Exception as exc:
            if isinstance(payload, dict):
                raise ApiPayloadError(
                    stage="load_observation",
                    message="Observation JSON payload could not be loaded with "
                            "Grid2Op observation.from_json().",
                    exception_type=type(exc).__name__,
                    exception=str(exc),
                ) from exc

    try:
        vector = np.asarray(payload, dtype=float)
    except Exception as exc:
        raise ApiPayloadError(
            stage="load_observation",
            message="Observation payload is neither Grid2Op JSON nor a numeric vector.",
            exception_type=type(exc).__name__,
            exception=str(exc),
        ) from exc

    expected = int(obs.to_vect().shape[0])
    if vector.ndim != 1 or vector.shape[0] != expected:
        raise ApiPayloadError(
            stage="load_observation",
            message="Observation vector has the wrong shape.",
            expected_observation_dim=expected,
            received_shape=tuple(int(v) for v in vector.shape),
        )

    try:
        loaded = obs.from_vect(vector)
    except Exception as exc:
        raise ApiPayloadError(
            stage="load_observation",
            message="Grid2Op observation.from_vect() failed.",
            exception_type=type(exc).__name__,
            exception=str(exc),
        ) from exc
    return obs if loaded is None else loaded


def build_recommendations(context: dict) -> List[dict]:
    """Full flow for one context: rebuild obs -> agent recommends -> format ->
    attach ENN uncertainty. Returns the list of recommendation dicts."""
    from recommendation_uncertainty import assess_recommendation
    try:
        env, agent, enn, calibration = get_services()
    except ApiConfigurationError:
        raise
    except Exception as exc:
        raise ApiConfigurationError(
            error="service_initialization_error",
            stage="get_services",
            message="The API failed while loading environment, agent, ENN, or calibration.",
            exception_type=type(exc).__name__,
            exception=str(exc),
        ) from exc

    obs = _load_observation(env, context)
    try:
        action = agent.act(obs, reward=None, done=False)
    except Exception as exc:
        raise ApiRuntimeError(
            stage="agent_act",
            message="CurriculumAgent failed to compute an action.",
            exception_type=type(exc).__name__,
            exception=str(exc),
        ) from exc

    try:
        info = assess_recommendation(obs, agent, enn, calibration)
    except Exception as exc:
        raise ApiRuntimeError(
            stage="assess_recommendation",
            message="ENN uncertainty scoring failed for the selected action.",
            exception_type=type(exc).__name__,
            exception=str(exc),
        ) from exc

    try:
        reco = _merge_uncertainty(_base_reco_dict(action, obs), info)
    except Exception as exc:
        raise ApiRuntimeError(
            stage="format_recommendation",
            message="Recommendation formatting failed.",
            exception_type=type(exc).__name__,
            exception=str(exc),
        ) from exc
    return [reco]


# --------------------------------------------------------------------------- #
#  Endpoints
# --------------------------------------------------------------------------- #
@app.post("/api/v1/recommendation")
def get_recommendation(request: RecommendationRequest):
    try:
        return build_recommendations(request.context)
    except ApiPayloadError as exc:
        raise HTTPException(status_code=400, detail=exc.to_detail()) from exc
    except ApiConfigurationError as exc:
        raise HTTPException(
            status_code=exc.status_code, detail=exc.to_detail()
        ) from exc
    except ApiRuntimeError as exc:
        logger.exception("Recommendation runtime error")
        raise HTTPException(status_code=500, detail=exc.to_detail()) from exc
    except Exception as exc:
        logger.exception("Unexpected recommendation API error")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "unexpected_api_error",
                "stage": "get_recommendation",
                "message": "Unexpected API error.",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        ) from exc


@app.get("/health")
def health():
    return {"status": "ok", "env": ENV_NAME}


@app.get("/diagnostics")
def diagnostics():
    paths = _api_artifact_paths()
    result: Dict[str, Any] = {
        "env": {
            "ENV_NAME": ENV_NAME,
            "ENV_DIR": _display_path(ENV_DIR),
            "AGENT_NAME": AGENT_NAME,
            "ARTIFACTS_DIR": _display_path(ARTIFACTS_DIR),
            "ASSETS_DIR": _display_path(ASSETS_DIR),
        },
        "selected_artifacts": {
            name: _file_info(path) for name, path in paths.items()
        },
    }

    try:
        bundle = _validate_api_artifacts()
        result["metadata"] = {
            "input_dim": bundle["metadata"].get("input_dim"),
            "num_classes": bundle["metadata"].get("num_classes"),
            "n_curated_actions": bundle["metadata"].get("n_curated_actions"),
            "environment": bundle["metadata"].get("environment"),
            "agent": bundle["metadata"].get("agent"),
            "action_set": bundle["metadata"].get("action_set"),
        }
        result["scaler"] = {
            "n_features_in": bundle["scaler_params"].get("n_features_in"),
        }
        result["actions"] = {
            "shape": bundle["action_shape"],
        }
        result["artifact_validation"] = {"ok": True}
    except ApiConfigurationError as exc:
        result["artifact_validation"] = {
            "ok": False,
            "detail": exc.to_detail(),
        }

    try:
        get_services()
        result["services"] = {"can_load": True}
    except ApiConfigurationError as exc:
        result["services"] = {
            "can_load": False,
            "detail": exc.to_detail(),
        }
    except Exception as exc:
        result["services"] = {
            "can_load": False,
            "detail": {
                "error": "service_initialization_error",
                "stage": "get_services",
                "message": "Unexpected service loading error.",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        }

    return result
