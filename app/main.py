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
scenario -- set GRID2OP_ENV if it differs from the CurriculumAgent's training
environment (see the Dockerfile and API.md).
"""
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="CurriculumAgent + ENN uncertainty API")

ROOT = Path(__file__).resolve().parents[1]
ENV_NAME = os.environ.get("GRID2OP_ENV", "l2rpn_icaps_2021_small")


# --------------------------------------------------------------------------- #
#  Request model (mirrors the template's RecommendationRequest)
# --------------------------------------------------------------------------- #
class RecommendationRequest(BaseModel):
    event: Optional[Dict[str, Any]] = None
    context: Dict[str, Any]


# --------------------------------------------------------------------------- #
#  Services: env + agent + ENN + calibration, loaded once (lazy, cached)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_services():
    import grid2op
    from lightsim2grid import LightSimBackend
    import run_example as rx                    # reuse the auto-discovery
    from recommendation_uncertainty import load_calibration

    env = grid2op.make(ENV_NAME, backend=LightSimBackend())

    scaler_json, meta_json, npz = rx.find_artifact_set()
    meta = json.loads(meta_json.read_text())
    weights = rx.find_enn_weights(prefer_dir=meta_json.parent)
    actions_path = rx.find_actions_npy(meta.get("n_curated_actions"))
    agent_dir = rx.find_agent_dir()

    agent = rx.load_agent(env, agent_dir)
    enn = rx.load_enn(weights, meta)
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
    """Add the two ENN epistemic-uncertainty percentiles into kpis."""
    reco.setdefault("kpis", {})
    reco["kpis"]["epistemic_uncertainty_total_pctile"] = \
        info["epistemic_uncertainty_total_pctile"]
    reco["kpis"]["epistemic_uncertainty_action_pctile"] = \
        info["epistemic_uncertainty_action_pctile"]
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
        except Exception:
            pass
    return obs.from_vect(np.asarray(payload, dtype=float))


def build_recommendations(context: dict) -> List[dict]:
    """Full flow for one context: rebuild obs -> agent recommends -> format ->
    attach ENN uncertainty. Returns the list of recommendation dicts."""
    from recommendation_uncertainty import assess_recommendation
    env, agent, enn, calibration = get_services()

    obs = _load_observation(env, context)
    action = agent.act(obs, reward=None, done=False)
    info = assess_recommendation(obs, agent, enn, calibration)
    reco = _merge_uncertainty(_base_reco_dict(action, obs), info)
    return [reco]


# --------------------------------------------------------------------------- #
#  Endpoints
# --------------------------------------------------------------------------- #
@app.post("/api/v1/recommendation")
def get_recommendation(request: RecommendationRequest):
    return build_recommendations(request.context)


@app.get("/health")
def health():
    return {"status": "ok", "env": ENV_NAME}
