"""
export_artifacts.py -- run ONCE on the training machine (repository ROOT).
Fully self-configuring: it finds the fitted scaler, the ENN weights and the
curated action set in the repository, infers input_dim / num_classes from
the checkpoint itself, and writes the two artifacts consumers need:

    example_artifacts/scaler_params.json   scaler mean/std as plain JSON
                                           (version-proof; no pickle shipped)
    example_artifacts/enn_meta.json        input_dim, num_classes and the
                                           class_mapping (identity by default)

Nothing to edit. Every guess is printed and cross-checked; use CLI flags to
override if needed:

    python export_artifacts.py [--scaler PATH] [--weights PATH]
                               [--actions PATH] [--mapping PATH.json|.npy]

Then commit the example_artifacts/ folder.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "example_artifacts"
_SKIP = {".git", "__pycache__", "tests", "curriculumagent", "example_artifacts"}


def _files(suffixes):
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in suffixes \
                and not (_SKIP & set(p.relative_to(ROOT).parts[:-1])):
            yield p


def find_scaler(override):
    if override:
        return joblib.load(override), Path(override)
    cands = sorted(_files({".pkl", ".joblib"}),
                   key=lambda p: ("scal" not in p.name.lower(), str(p)))
    for p in cands:
        try:
            obj = joblib.load(p)
        except Exception:
            continue
        if hasattr(obj, "mean_") and hasattr(obj, "scale_"):
            print(f"auto: scaler   -> {p.relative_to(ROOT)} "
                  f"({type(obj).__name__}, {int(obj.n_features_in_)} features)")
            return obj, p
    sys.exit("[error] no fitted scaler (.pkl/.joblib with mean_/scale_) "
             "found. Pass --scaler PATH.")


def find_weights(override):
    if override:
        return Path(override)
    cands = sorted(_files({".pth", ".pt"}),
                   key=lambda p: ("enn" not in p.name.lower(), str(p)))
    if not cands:
        sys.exit("[error] no ENN .pth/.pt checkpoint found. Pass --weights.")
    print(f"auto: weights  -> {cands[0].relative_to(ROOT)}")
    return cands[0]


def infer_dims(weights_path: Path):
    """input_dim / num_classes from the state_dict: first 2-D tensor is
    [hidden, input_dim], last 2-D tensor is [num_classes, hidden]."""
    sd = torch.load(weights_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    mats = [t for t in sd.values() if getattr(t, "ndim", 0) == 2]
    if not mats:
        sys.exit("[error] checkpoint has no 2-D weight tensors; is it a "
                 "state_dict?")
    input_dim = int(mats[0].shape[1])
    num_classes = int(mats[-1].shape[0])
    print(f"auto: inferred -> input_dim={input_dim}  num_classes={num_classes}")
    return input_dim, num_classes


def find_actions(override, num_classes):
    if override:
        p = Path(override)
        return p, np.load(p, mmap_mode="r").shape
    cands = []
    for p in _files({".npy"}):
        try:
            arr = np.load(p, mmap_mode="r")
        except Exception:
            continue
        if arr.ndim == 2:
            cands.append((p, arr.shape))
    exact = [c for c in cands if c[1][0] == num_classes]
    pool = exact or cands
    pool.sort(key=lambda c: ("action" not in c[0].name.lower(), str(c[0])))
    if not pool:
        sys.exit("[error] no 2-D .npy curated action set found. "
                 "Pass --actions PATH.")
    p, shape = pool[0]
    print(f"auto: actions  -> {p.relative_to(ROOT)}  shape={shape}")
    return p, shape


def build_class_mapping(mapping_arg, num_classes):
    if mapping_arg:
        src = Path(mapping_arg)
        if src.suffix == ".json":
            raw = json.loads(src.read_text())
            raw = raw.get("class_mapping", raw)
            return {str(k): int(v) for k, v in raw.items()}
        if src.suffix == ".npy":
            arr = np.load(src)                   # arr[i] = label of action i
            return {str(i): int(v) for i, v in enumerate(arr)}
        sys.exit(f"[error] unsupported mapping source: {src}")
    return {str(k): k for k in range(num_classes)}     # identity


def export(scaler_path=None, weights_path=None, actions_arg=None,
           mapping=None):
    """Generate example_artifacts/. Callable from other scripts
    (run_example.py calls this automatically on first run if the artifacts
    are missing)."""
    scaler, _ = find_scaler(scaler_path)
    weights = find_weights(weights_path)
    input_dim, num_classes = infer_dims(weights)
    actions_path, act_shape = find_actions(actions_arg, num_classes)

    # cross-checks -------------------------------------------------------------
    if int(scaler.n_features_in_) != input_dim:
        sys.exit(f"[error] scaler has {int(scaler.n_features_in_)} features "
                 f"but the ENN input layer expects {input_dim} -- wrong "
                 f"scaler or wrong checkpoint. Use --scaler/--weights.")
    if act_shape[0] != num_classes:
        print(f"[warn] actions.npy has {act_shape[0]} rows but the ENN has "
              f"{num_classes} classes -- fine if the mapping is not identity "
              f"(pass --mapping), otherwise check the files.")
    mapping_arg = mapping

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "scaler_params.json").write_text(json.dumps({
        "type": type(scaler).__name__,
        "mean": np.asarray(scaler.mean_, dtype=float).tolist(),
        "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        "var": np.asarray(scaler.var_, dtype=float).tolist(),
        "n_features_in": int(scaler.n_features_in_),
    }))
    (OUT_DIR / "enn_meta.json").write_text(json.dumps({
        "input_dim": input_dim,
        "num_classes": num_classes,
        "n_curated_actions": int(act_shape[0]),
        "environment": "l2rpn_icaps_2021_small",
        "grid2op_version": "1.9.8",
        "class_mapping": build_class_mapping(mapping_arg, num_classes),
    }))
    print(f"\n[ok] wrote {OUT_DIR.relative_to(ROOT)}/scaler_params.json and "
          f"enn_meta.json. Commit the example_artifacts/ folder.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaler"); ap.add_argument("--weights")
    ap.add_argument("--actions"); ap.add_argument("--mapping")
    a = ap.parse_args()
    export(a.scaler, a.weights, a.actions, a.mapping)


if __name__ == "__main__":
    main()
