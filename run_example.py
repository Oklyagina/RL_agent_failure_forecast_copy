"""
run_example.py -- complete, self-contained example of the ENN
uncertainty-quantification module (repository ROOT). Self-configuring: it
auto-discovers the ENN weights, the curated action set, the ENN architecture
module and the agent binaries, so no paths need to be edited. Every guess is
printed; anything can still be overridden in CONFIG below.

Pipeline: Grid2Op environment -> CurriculumAgent (curriculumagent/) ->
trained ENN + training scaler (rebuilt from example_artifacts/, plain JSON)
-> percentile calibration -> assess_recommendation per step -> output,
including the recommendations list in the InteractiveAI format ("kpis").

Requirements: Python 3.9/3.10 + requirements.txt (see README).
Run:  python run_example.py
"""

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# CONFIG -- everything is auto-discovered; set a value only to override.
# ----------------------------------------------------------------------------
ENV_NAME = "l2rpn_icaps_2021_small"
ENN_WEIGHTS = Path("assets/enn_36.pth")          # e.g. Path("src/models/enn_36.pth")
ACTIONS_NPY = Path("assets/network36/actions/actions.npy")  # e.g. Path("src/models/network36/actions/actions.npy")
CALIBRATION_NPZ = None      # e.g. Path("models_curriculum/enn_pctile_calib.npz")
SCALER_JSON = None         # e.g. Path("models_curriculum/scaler_params.json")
ENN_META_JSON = None        # e.g. Path("models_curriculum/enn_meta.json")
AGENT_DIR = None            # dir containing model/ and actions/ subfolders
N_STEPS = 5
SEED = 0
# ----------------------------------------------------------------------------

_SKIP_DIRS = {".git", "__pycache__", "tests", "curriculumagent"}


def find_artifact_set():
    """Locate scaler_params.json + enn_meta.json (+ calibration .npz).

    Priority: (1) CONFIG overrides; (2) any folder produced by
    training/train_enn.py (contains both JSONs -- newest wins); the
    calibration .npz is taken from the same folder when present, else from
    the repository root (enn_pctile_calib.npz).

    The scaler is CREATED AT ENN TRAINING TIME -- if nothing is found, the
    pipeline must be trained first (see TRAINING.md)."""
    if SCALER_JSON and ENN_META_JSON:
        scaler_json, meta_json = Path(SCALER_JSON), Path(ENN_META_JSON)
    else:
        cands = [d for d in {p.parent for p in ROOT.rglob("enn_meta.json")}
                 if (d / "scaler_params.json").is_file()
                 and not (_SKIP_DIRS & set(d.relative_to(ROOT).parts))]
        if not cands:
            sys.exit(
                "[error] no trained artifacts found (scaler_params.json + "
                "enn_meta.json).\n        The scaler is created when the ENN "
                "is trained -- run the training pipeline first "
                "(see TRAINING.md):\n"
                "          python training/collect_rollouts.py --agent "
                "curriculum --episodes 50 --out-dir data_curriculum\n"
                "          python training/train_enn.py --data-dir "
                "data_curriculum --out-dir models_curriculum --agent-name "
                "curriculum\n        and then re-run this script.")
        cands.sort(key=lambda d: (d / "enn_meta.json").stat().st_mtime,
                   reverse=True)
        d = cands[0]
        scaler_json, meta_json = d / "scaler_params.json", d / "enn_meta.json"
        print(f"       auto: artifacts   -> {d.relative_to(ROOT)}/")
    if CALIBRATION_NPZ:
        npz = Path(CALIBRATION_NPZ)
    elif (meta_json.parent / "enn_pctile_calib.npz").is_file():
        npz = meta_json.parent / "enn_pctile_calib.npz"
    else:
        npz = ROOT / "enn_pctile_calib.npz"
    print(f"       auto: calibration -> {npz.relative_to(ROOT)}")
    return scaler_json, meta_json, npz


def _walk_files(suffixes):
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in suffixes \
                and not (_SKIP_DIRS & set(p.relative_to(ROOT).parts[:-1])):
            yield p


def find_enn_weights(prefer_dir: Path | None = None) -> Path:
    if ENN_WEIGHTS:
        return Path(ENN_WEIGHTS)
    cands = sorted(_walk_files({".pth", ".pt"}),
                   key=lambda p: (prefer_dir is not None
                                  and p.parent != prefer_dir,
                                  "enn" not in p.name.lower(), str(p)))
    if not cands:
        sys.exit("[error] no .pth/.pt ENN weights found in the repository. "
                 "Commit the trained ENN (e.g. src/models/enn_36.pth) or set "
                 "ENN_WEIGHTS in the CONFIG block.")
    print(f"       auto: ENN weights -> {cands[0].relative_to(ROOT)}"
          + (f"  (candidates: {len(cands)})" if len(cands) > 1 else ""))
    return cands[0]


def find_actions_npy(n_curated: int | None) -> Path:
    if ACTIONS_NPY:
        return Path(ACTIONS_NPY)
    cands = []
    for p in _walk_files({".npy"}):
        try:
            arr = np.load(p, mmap_mode="r")
        except Exception:
            continue
        if arr.ndim == 2:
            cands.append((p, arr.shape))
    if n_curated is not None:                     # disambiguate via enn_meta
        exact = [c for c in cands if c[1][0] == n_curated]
        if exact:
            cands = exact
    cands.sort(key=lambda c: ("action" not in c[0].name.lower(), str(c[0])))
    if not cands:
        sys.exit("[error] no 2-D .npy curated action set found. Commit "
                 "actions.npy (rows = action.to_vect()) or set ACTIONS_NPY "
                 "in the CONFIG block.")
    p, shape = cands[0]
    print(f"       auto: action set  -> {p.relative_to(ROOT)}  shape={shape}")
    return p


def import_evidential_network():
    """Find the EvidentialNetwork class wherever it lives in the repo."""
    for mod in ("src.enn_models", "enn_models", "src.models.enn_models"):
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "EvidentialNetwork"):
                print(f"       auto: ENN class   -> {mod}.EvidentialNetwork")
                return m.EvidentialNetwork
        except ModuleNotFoundError:
            continue
    for p in _walk_files({".py"}):                # last resort: scan sources
        try:
            if "class EvidentialNetwork" in p.read_text(errors="ignore"):
                spec = importlib.util.spec_from_file_location(p.stem, p)
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                print(f"       auto: ENN class   -> {p.relative_to(ROOT)}")
                return m.EvidentialNetwork
        except Exception:
            continue
    sys.exit("[error] could not find a module defining EvidentialNetwork.")


def find_agent_dir() -> Path:
    """The CurriculumAgent entrypoint make_agent(env, path) expects `path`
    to contain model/ and actions/ subfolders."""
    if AGENT_DIR:
        return Path(AGENT_DIR)
    preferred = [
        ROOT / "assets" / "network36",
        ROOT / "src" / "models" / "network36",
    ]
    base = ROOT / "curriculumagent"
    invalid = []
    for d in [*preferred, base, *sorted(p for p in base.rglob("*") if p.is_dir())]:
        if (d / "model").is_dir() and (d / "actions").is_dir():
            if not has_valid_saved_model(d):
                invalid.append(d)
                continue
            print(f"       auto: agent dir   -> {d.relative_to(ROOT)}")
            return d
    hint = ""
    if invalid:
        bad = ", ".join(str(d.relative_to(ROOT)) for d in invalid)
        hint = f"\n        Skipped invalid SavedModel artifact(s): {bad}."
    sys.exit("[error] no valid folder with model/ and actions/ subfolders "
             "found. Expected a non-empty TensorFlow SavedModel under "
             "assets/network36/ or src/models/network36/. Set AGENT_DIR in "
             f"the CONFIG block to override.{hint}")


def has_valid_saved_model(agent_dir: Path) -> bool:
    model_dir = agent_dir / "model"
    variables_dir = model_dir / "variables"
    required_files = [
        model_dir / "saved_model.pb",
        variables_dir / "variables.index",
        variables_dir / "variables.data-00000-of-00001",
    ]
    return all(p.is_file() and p.stat().st_size > 0 for p in required_files)


def scaler_from_json(path: Path):
    """Rebuild the training-time StandardScaler from exported JSON parameters
    (version-proof: no pickle involved)."""
    from sklearn.preprocessing import StandardScaler
    p = json.loads(path.read_text())
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(p["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(p["scale"], dtype=np.float64)
    scaler.var_ = np.asarray(p["var"], dtype=np.float64)
    scaler.n_features_in_ = int(p["n_features_in"])
    return scaler


def load_enn(weights: Path, meta: dict, device: str = "cpu"):
    """Instantiate the ENN with the architecture recorded in enn_meta.json
    and load the trained weights -- the `enn` for assess_recommendation."""
    import torch
    EvidentialNetwork = import_evidential_network()
    enn = EvidentialNetwork(input_dim=int(meta["input_dim"]),
                            num_classes=int(meta["num_classes"]))
    enn.load_state_dict(torch.load(weights, map_location=device))
    enn.eval()
    return enn


def load_agent(env, agent_dir: Path):
    """CurriculumAgent official entrypoint (curriculumagent 1.x):
    make_agent(env, this_directory_path) with model/ and actions/ inside."""
    from curriculumagent.submission.my_agent import make_agent
    return make_agent(env, str(agent_dir))


def to_interactiveai(action, info: dict) -> dict:
    """One recommendation in the InteractiveAI format: percentiles inside
    "kpis", alongside efficiency_of_the_reco (filled by the platform)."""
    return {
        "title": "Topological recommendation (CurriculumAgent)",
        "description": str(action),
        "use_case": "PowerGrid",
        "agent_type": 2,
        "actions": [action.as_serializable_dict()],
        "kpis": {
            "efficiency_of_the_reco": None,
            "epistemic_uncertainty_total_pctile":
                info["epistemic_uncertainty_total_pctile"],
            "epistemic_uncertainty_action_pctile":
                info["epistemic_uncertainty_action_pctile"],
        },
    }


def main() -> None:
    import grid2op
    from lightsim2grid import LightSimBackend
    from recommendation_uncertainty import (load_calibration,
                                            assess_recommendation)

    # 0. Auto-discovery --------------------------------------------------------
    print("[0/4] resolving artifacts:")
    scaler_json, meta_json, npz = find_artifact_set()
    meta = json.loads(meta_json.read_text())
    weights = find_enn_weights(prefer_dir=meta_json.parent)
    actions_path = find_actions_npy(meta.get("n_curated_actions"))
    agent_dir = find_agent_dir()

    # 1. Environment -----------------------------------------------------------
    env = grid2op.make(ENV_NAME, backend=LightSimBackend())
    env.seed(SEED)
    obs = env.reset()
    print(f"[1/4] environment '{ENV_NAME}' ready "
          f"(obs vector size = {obs.to_vect().shape[0]})")

    # 2. Agent -----------------------------------------------------------------
    agent = load_agent(env, agent_dir)
    print(f"[2/4] CurriculumAgent loaded")

    # 3. ENN + scaler + calibration --------------------------------------------
    enn = load_enn(weights, meta)
    calibration = load_calibration(
        str(npz),
        scaler=scaler_from_json(scaler_json),
        action_set=str(actions_path),
        class_mapping=str(meta_json),
    )
    print(f"[3/4] ENN ({meta['num_classes']} classes), scaler and "
          f"calibration loaded")

    # 4. Assess live recommendations -------------------------------------------
    print(f"[4/4] running {N_STEPS} steps:\n")
    reward, done = env.reward_range[0], False
    recommendations = []
    for t in range(N_STEPS):
        action = agent.act(obs, reward, done)
        info = assess_recommendation(obs, agent, enn, calibration)
        print(f"  step {t}: chosen_action_id={info['chosen_action_id']}  "
              f"total_pctile={info['epistemic_uncertainty_total_pctile']}  "
              f"action_pctile={info['epistemic_uncertainty_action_pctile']}")
        recommendations.append(to_interactiveai(action, info))
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
            done = False

    print("\nRecommendations list in the InteractiveAI format "
          "(first entry shown):")
    print(json.dumps(recommendations[0], indent=2)[:1500])


if __name__ == "__main__":
    main()
