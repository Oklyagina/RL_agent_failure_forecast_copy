"""
Live demo of the trained models (``python main.py demo``).

Self-configuring: it auto-discovers the ENN weights, the curated action set,
the ENN architecture module and the agent binaries, so no paths need to be
edited. Every guess is printed; anything can still be overridden in CONFIG
below.

Pipeline: Grid2Op environment -> configured policy ->
trained ENN + training scaler (rebuilt from artifacts/, plain JSON) ->
percentile calibration -> assess_recommendation -> t+12 failure prediction
-> per-line rule explanations (when failure_forecast/explainability/rules/
exists) -> output, including the recommendations list in the InteractiveAI
format ("kpis"; per-line results in kpis["failure_forecasts"]).

Requirements: Python 3.9/3.10 + requirements.txt (see README).
Run:  python main.py demo [--no-explain] [--rules-dir PATH]
"""

import json
import sys
from pathlib import Path

from project_config import (AGENT_FACTORY, AGENT_NAME, ARTIFACTS_DIR, ASSETS_DIR,
                            configure_grid2op_warnings, ENV_DIR,
                            ENV_NAME,
                            SEED as CONFIG_SEED)
from src import runtime_artifacts
from src.agent_runtime import call_agent
from src.pipeline_support import failure_dir as _failure_dir
from src.pipeline_support import has_valid_saved_model  # noqa: F401  (legacy name)
from src.runtime_artifacts import enn_uncertainty_fn, scaler_from_json  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------
# CONFIG -- everything is auto-discovered; set a value only to override.
# ----------------------------------------------------------------------------
ENN_WEIGHTS = None          # e.g. Path("artifacts/ai4realnet_small/curriculum/model/enn_curriculum.pth")
ACTIONS_NPY = None          # e.g. Path("artifacts/ai4realnet_small/curriculum/rollouts/actions.npy")
CALIBRATION_NPZ = None      # e.g. Path("artifacts/ai4realnet_small/curriculum/model/enn_pctile_calib.npz")
SCALER_JSON = None          # e.g. Path("artifacts/ai4realnet_small/curriculum/model/scaler_params.json")
ENN_META_JSON = None        # e.g. Path("artifacts/ai4realnet_small/curriculum/model/enn_meta.json")
AGENT_DIR = None            # dir containing model/ and actions/ subfolders
N_STEPS = 13  # one hour of history plus the state being assessed
SEED = CONFIG_SEED
# ----------------------------------------------------------------------------

def _artifact_settings() -> runtime_artifacts.RuntimeArtifactSettings:
    """Discovery settings built from the CONFIG overrides above."""
    return runtime_artifacts.RuntimeArtifactSettings(
        root=ROOT,
        artifacts_dir=ARTIFACTS_DIR,
        assets_dir=ASSETS_DIR,
        env_name=ENV_NAME,
        agent_name=AGENT_NAME,
        enn_weights=ENN_WEIGHTS,
        actions_npy=ACTIONS_NPY,
        calibration_npz=CALIBRATION_NPZ,
        scaler_json=SCALER_JSON,
        enn_meta_json=ENN_META_JSON,
        agent_dir=AGENT_DIR,
    )


# Artifact discovery and loading live in src/runtime_artifacts.py; these thin
# wrappers keep the historical helper names and the CONFIG overrides above.
def find_artifact_set():
    """Locate scaler_params.json + enn_meta.json (+ calibration .npz).

    Priority: (1) CONFIG overrides; (2) artifacts/ folders produced by
    training/train_enn.py; (3) other active fallback folders. The calibration
    .npz must sit next to the selected metadata."""
    return runtime_artifacts.find_artifact_set(_artifact_settings())


def find_enn_weights(prefer_dir: Path | None = None) -> Path:
    return runtime_artifacts.find_enn_weights(prefer_dir, _artifact_settings())


def find_actions_npy(meta: dict) -> Path:
    return runtime_artifacts.find_actions_npy(meta, _artifact_settings())


def import_evidential_network():
    """Find the EvidentialNetwork class wherever it lives in the repo."""
    return runtime_artifacts.import_evidential_network(_artifact_settings())


def find_agent_dir() -> Path:
    """Locate configured bundled policy assets (model/ + actions/).

    Custom policies configured with ``AGENT_FACTORY`` do not use this helper.
    """
    return runtime_artifacts.find_agent_dir(_artifact_settings())


def load_enn(weights: Path, meta: dict, device: str = "cpu"):
    """Instantiate the ENN recorded in enn_meta.json and load its weights."""
    return runtime_artifacts.load_enn(
        weights, meta, device, settings=_artifact_settings()
    )


def load_enn_uncertainty():
    """Return ``(enn, uncertainty_fn)`` for failure-forecast ENN features."""
    return runtime_artifacts.load_enn_uncertainty(_artifact_settings())


def load_agent(env, agent_dir: Path | None = None):
    """Load the configured policy; custom agents use AGENT_FACTORY."""
    return runtime_artifacts.load_agent(
        env, agent_dir, factory_spec=AGENT_FACTORY
    )


def to_interactiveai(action, info: dict) -> dict:
    """One recommendation in the InteractiveAI format: percentiles inside
    "kpis", alongside efficiency_of_the_reco (filled by the platform)."""
    return {
        "title": f"Topological recommendation ({AGENT_NAME})",
        "description": str(action),
        "use_case": "PowerGrid",
        "agent_type": 2,
        "actions": [action.as_serializable_dict()],
        "kpis": {
            "efficiency_of_the_reco": None,
            "epistemic_uncertainty_pct": info["epistemic_uncertainty_pct"],
            "epistemic_uncertainty_total_pctile": info["epistemic_uncertainty_total_pctile"],
            "epistemic_uncertainty_action_pctile": info["epistemic_uncertainty_action_pctile"],
            "epistemic_uncertainty_level": info["epistemic_uncertainty_level"],
            "epistemic_confidence_level": info["epistemic_confidence_level"],
        },
    }


def default_rules_dir(failure_dir: Path) -> Path:
    return failure_dir / "explainability" / "rules"


def build_failure_scorer(predictor, failure_dir: Path, *, explain: bool = True,
                         rules_dir: Path | None = None):
    """Return ``(score_many, rules_dir_or_None)`` for the t+12 forecast.

    ``score_many`` has the signature of ``FailureForecastPredictor.predict_many``.
    With usable rules it is ``ExplainabilityOrchestrator.predict_and_explain_many``,
    which adds per-line explanation fields; the HGB probability and binary
    prediction are identical either way. Missing or unloadable rules only
    produce a warning. No rows CSV, OOF file, or LLM configuration is needed.
    """
    if not explain:
        print("       explanations: disabled (--no-explain)")
        return predictor.predict_many, None
    rules_dir = Path(rules_dir) if rules_dir else default_rules_dir(failure_dir)
    if not rules_dir.is_dir():
        print(f"[warn] explanations unavailable: no rules directory at {rules_dir}; "
              "continuing with HGB predictions only")
        return predictor.predict_many, None
    try:
        from src.explainability.orchestrator import ExplainabilityOrchestrator

        orchestrator = ExplainabilityOrchestrator(predictor, rules_dir=rules_dir)
    except Exception as exc:  # explanations are optional; HGB must still run
        print(f"[warn] explanations unavailable ({type(exc).__name__}: {exc}); "
              "continuing with HGB predictions only")
        return predictor.predict_many, None
    explainer = orchestrator.rule_explainer
    if explainer is None or not explainer.available_lines:
        print(f"[warn] explanations unavailable: no valid rules in {rules_dir}; "
              "continuing with HGB predictions only")
        return predictor.predict_many, None
    print(f"       explanations: {len(explainer.available_lines)} rules "
          f"from {rules_dir}")
    return orchestrator.predict_and_explain_many, rules_dir


def print_failure_forecasts(failures: list[dict]) -> None:
    """Readable per-line summary of the t+12 forecasts (and explanations)."""
    print("\n  t+12 failure forecasts (one per contingency):")
    for item in failures:
        label = "FAILURE" if item.get("failure_prediction") else "no failure"
        print(f"    line {item.get('line')}: p={item.get('failure_probability'):.3f} "
              f"-> {label}")
        if "explanation_status" not in item:
            continue
        if item["explanation_status"] == "available":
            agrees = "agrees" if item.get("surrogate_agrees") else "DISAGREES"
            print(f"      rule ({agrees} with HGB): {item['explanation']}")
        else:
            print(f"      no explanation: {item.get('explanation_error')}")


def run(explain: bool = True, rules_dir: Path | None = None) -> None:
    configure_grid2op_warnings()

    import grid2op
    import joblib
    from lightsim2grid import LightSimBackend
    from recommendation_uncertainty import (load_calibration,
                                            assess_recommendation)
    from src.failure_forecast import (FailureForecastConfig,
                                      FailureForecastPredictor)

    # 0. Auto-discovery --------------------------------------------------------
    print("[0/4] resolving artifacts:")
    scaler_json, meta_json, npz = find_artifact_set()
    meta = json.loads(meta_json.read_text())
    weights = find_enn_weights(prefer_dir=meta_json.parent)
    actions_path = find_actions_npy(meta)
    agent_dir = None if AGENT_FACTORY else find_agent_dir()
    failure_dir = _failure_dir(ARTIFACTS_DIR, ENV_NAME, AGENT_NAME)
    failure_paths = [failure_dir / "mean_forecaster.pkl",
                     failure_dir / "aleatoric_forecaster.pkl",
                     failure_dir / "failure_classifier.pkl"]
    missing = [str(path) for path in failure_paths if not path.is_file()]
    if missing:
        sys.exit("[error] missing failure artifacts: " + ", ".join(missing))

    # 1. Environment -----------------------------------------------------------
    env = grid2op.make(str(ENV_DIR), backend=LightSimBackend())
    env.seed(SEED)
    obs = env.reset()
    print(f"[1/4] environment '{ENV_NAME}' ready "
          f"(obs vector size = {obs.to_vect().shape[0]})")

    # 2. Agent -----------------------------------------------------------------
    agent = load_agent(env, agent_dir)
    print(f"[2/4] Policy agent loaded ({AGENT_NAME})")

    # 3. ENN + scaler + calibration --------------------------------------------
    enn = load_enn(weights, meta)
    calibration = load_calibration(
        str(npz),
        scaler=scaler_from_json(scaler_json),
        action_set=str(actions_path),
        class_mapping=str(meta_json),
    )
    mean_model, aleatoric_model = map(joblib.load, failure_paths[:2])
    predictor = FailureForecastPredictor.load(failure_paths[2])
    lines = list(predictor.line_map)
    failure_cfg = FailureForecastConfig.from_env(
        env, env_name=ENV_NAME, agent_name=AGENT_NAME,
        artifact_dir=failure_dir, lines_to_test=lines)
    print(f"[3/4] ENN ({meta['num_classes']} classes), scaler and "
          f"calibration loaded; failure models loaded")
    score_many, used_rules_dir = build_failure_scorer(
        predictor, failure_dir, explain=explain, rules_dir=rules_dir)

    # 4. Assess live recommendations -------------------------------------------
    print(f"[4/4] running {N_STEPS} steps:\n")
    reward, done = env.reward_range[0], False
    recommendations = []
    observations = []
    for t in range(N_STEPS):
        observations.append(obs)
        action = call_agent(agent, obs, reward=reward, done=done)
        info = assess_recommendation(obs, agent, enn, calibration, action=action)
        print(f"  step {t}: chosen_action_id={info['chosen_action_id']}  "
              f"total_pctile={info['epistemic_uncertainty_total_pctile']}  "
              f"action_pctile={info['epistemic_uncertainty_action_pctile']}")
        recommendations.append(to_interactiveai(action, info))
        if t == N_STEPS - 1:
            failures = score_many(
                obs, observations, lines,
                mean_model, aleatoric_model, failure_cfg,
                env=env,
                model_enn=enn, get_uncertainty_fn=enn_uncertainty_fn(calibration.scaler))
            kpis = recommendations[-1]["kpis"]
            if failures:
                worst = max(failures, key=lambda f: f["failure_probability"])
                kpis["failure_prediction"] = worst["failure_prediction"]
                kpis["failure_probability"] = worst["failure_probability"]
            kpis["failure_forecasts"] = failures
            kpis["failure_explanation_rules"] = (
                str(used_rules_dir) if used_rules_dir else None)
            print_failure_forecasts(failures)
            break
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
            observations = []
            done = False

    print("\nRecommendations list in the InteractiveAI format "
          "(final entry shown):")
    print(json.dumps(recommendations[-1], indent=2)[:1500])

