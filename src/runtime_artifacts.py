"""Shared runtime discovery and loading for ENN and policy artifacts.

The example CLI, training collectors, and FastAPI service use the same
implementation through this module.  Optional settings keep the historical
``run_example.py`` override variables usable without coupling application code
to an executable script.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from project_config import (
    AGENT_FACTORY,
    AGENT_NAME,
    ARTIFACTS_DIR,
    ASSETS_DIR,
    ENV_NAME,
    ROOT as PROJECT_ROOT,
)
from src.pipeline_support import has_valid_saved_model


DEFAULT_SKIP_DIRS = frozenset(
    {".git", "__pycache__", "tests", "curriculumagent", "archive"}
)


@dataclass(frozen=True)
class RuntimeArtifactSettings:
    """Discovery roots and optional legacy overrides."""

    root: Path = PROJECT_ROOT
    artifacts_dir: Path = ARTIFACTS_DIR
    assets_dir: Path = ASSETS_DIR
    env_name: str = ENV_NAME
    agent_name: str = AGENT_NAME
    enn_weights: Path | None = None
    actions_npy: Path | None = None
    calibration_npz: Path | None = None
    scaler_json: Path | None = None
    enn_meta_json: Path | None = None
    agent_dir: Path | None = None
    skip_dirs: frozenset[str] = DEFAULT_SKIP_DIRS


def _settings(value: RuntimeArtifactSettings | None) -> RuntimeArtifactSettings:
    return value or RuntimeArtifactSettings()


def _display(path: Path, settings: RuntimeArtifactSettings) -> str:
    try:
        return path.resolve().relative_to(settings.root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _relative_parts(path: Path, settings: RuntimeArtifactSettings) -> tuple[str, ...]:
    try:
        return path.resolve().relative_to(settings.root.resolve()).parts
    except ValueError:
        return ()


def _artifact_rank(path: Path, settings: RuntimeArtifactSettings) -> int:
    parts = _relative_parts(path, settings)
    if parts and parts[0] == "artifacts":
        return 0
    if path.resolve() == settings.root.resolve():
        return 99
    return 1


def _is_skipped_path(path: Path, settings: RuntimeArtifactSettings) -> bool:
    parts = _relative_parts(path, settings)
    return bool(settings.skip_dirs & set(parts)) or any(
        part.startswith("models_") for part in parts
    )


def _walk_files(
    suffixes: Iterable[str], settings: RuntimeArtifactSettings
):
    suffix_set = set(suffixes)
    for path in settings.root.rglob("*"):
        if (
            path.is_file()
            and path.suffix in suffix_set
            and not _is_skipped_path(path, settings)
        ):
            yield path


def find_artifact_set(
    settings: RuntimeArtifactSettings | None = None,
) -> tuple[Path, Path, Path]:
    """Locate matching scaler, ENN metadata, and calibration artifacts."""
    settings = _settings(settings)
    if settings.scaler_json and settings.enn_meta_json:
        scaler_json = Path(settings.scaler_json)
        meta_json = Path(settings.enn_meta_json)
    else:
        configured_dir = (
            settings.artifacts_dir
            / settings.env_name
            / settings.agent_name
            / "model"
        )
        candidates = []
        if (
            (configured_dir / "enn_meta.json").is_file()
            and (configured_dir / "scaler_params.json").is_file()
        ):
            candidates.append(configured_dir)
        candidates.extend(
            directory
            for directory in {
                path.parent for path in settings.root.rglob("enn_meta.json")
            }
            if (directory / "scaler_params.json").is_file()
            and directory != configured_dir
            and not _is_skipped_path(directory, settings)
        )
        if not candidates:
            sys.exit(
                "[error] no trained artifacts found (scaler_params.json + "
                "enn_meta.json).\n        The scaler is created when the ENN "
                "is trained -- run the training pipeline first "
                "(see TRAINING.md):\n"
                "          python training/collect_rollouts.py\n"
                "          python training/train_enn.py\n"
                "        and then re-run this script."
            )
        candidates.sort(
            key=lambda directory: (
                directory != configured_dir,
                _artifact_rank(directory, settings),
                -(directory / "enn_meta.json").stat().st_mtime,
            )
        )
        selected = candidates[0]
        scaler_json = selected / "scaler_params.json"
        meta_json = selected / "enn_meta.json"
        print(f"       auto: artifacts   -> {_display(selected, settings)}/")

    if settings.calibration_npz:
        calibration = Path(settings.calibration_npz)
    elif (meta_json.parent / "enn_pctile_calib.npz").is_file():
        calibration = meta_json.parent / "enn_pctile_calib.npz"
    elif (
        meta_json.parent.resolve() == settings.root.resolve()
        and (settings.root / "enn_pctile_calib.npz").is_file()
    ):
        calibration = settings.root / "enn_pctile_calib.npz"
    else:
        sys.exit(
            "[error] no calibration file found next to the selected trained "
            f"artifacts: {_display(meta_json.parent, settings)}/\n"
            "        Expected enn_pctile_calib.npz there, or set "
            "CALIBRATION_NPZ in the CONFIG block."
        )
    print(f"       auto: calibration -> {_display(calibration, settings)}")
    return scaler_json, meta_json, calibration


def find_enn_weights(
    prefer_dir: Path | None = None,
    settings: RuntimeArtifactSettings | None = None,
) -> Path:
    """Locate an ENN checkpoint, preferring the selected metadata folder."""
    settings = _settings(settings)
    if settings.enn_weights:
        return Path(settings.enn_weights)
    candidates = sorted(
        _walk_files({".pth", ".pt"}, settings),
        key=lambda path: (
            prefer_dir is not None and path.parent != prefer_dir,
            "enn" not in path.name.lower(),
            str(path),
        ),
    )
    if not candidates:
        sys.exit(
            "[error] no .pth/.pt ENN weights found in the repository. "
            "Commit the trained ENN under artifacts/ or set ENN_WEIGHTS "
            "in the CONFIG block."
        )
    selected = candidates[0]
    suffix = f"  (candidates: {len(candidates)})" if len(candidates) > 1 else ""
    print(f"       auto: ENN weights -> {_display(selected, settings)}{suffix}")
    return selected


def find_actions_npy(
    meta: dict,
    settings: RuntimeArtifactSettings | None = None,
) -> Path:
    """Locate the curated 2-D action set matching ENN metadata."""
    settings = _settings(settings)
    if settings.actions_npy:
        return Path(settings.actions_npy)
    for key in ("action_set", "actions_path"):
        if meta.get(key):
            path = Path(meta[key])
            if not path.is_absolute():
                path = settings.root / path
            if path.is_file():
                print(f"       auto: action set  -> {_display(path, settings)}")
                return path
            print(f"       warn: {key} in enn_meta.json not found -> {meta[key]}")

    expected_rows = meta.get("n_curated_actions")
    candidates = []
    for path in _walk_files({".npy"}, settings):
        try:
            array = np.load(path, mmap_mode="r")
        except Exception:
            continue
        if array.ndim == 2:
            candidates.append((path, array.shape))
    if expected_rows is not None:
        exact = [item for item in candidates if item[1][0] == expected_rows]
        if exact:
            candidates = exact
    candidates.sort(
        key=lambda item: (
            _artifact_rank(item[0].parent, settings),
            "action" not in item[0].name.lower(),
            str(item[0]),
        )
    )
    if not candidates:
        sys.exit(
            "[error] no 2-D .npy curated action set found. Commit "
            "actions.npy (rows = action.to_vect()) or set ACTIONS_NPY "
            "in the CONFIG block."
        )
    selected, shape = candidates[0]
    print(
        f"       auto: action set  -> {_display(selected, settings)}  "
        f"shape={shape}"
    )
    return selected


def import_evidential_network(
    settings: RuntimeArtifactSettings | None = None,
):
    """Find the EvidentialNetwork class using the established search order."""
    settings = _settings(settings)
    for module_name in ("src.enn_models", "enn_models", "src.models.enn_models"):
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "EvidentialNetwork"):
                print(
                    "       auto: ENN class   -> "
                    f"{module_name}.EvidentialNetwork"
                )
                return module.EvidentialNetwork
        except ModuleNotFoundError:
            continue
    for path in _walk_files({".py"}, settings):
        try:
            if "class EvidentialNetwork" in path.read_text(errors="ignore"):
                spec = importlib.util.spec_from_file_location(path.stem, path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"       auto: ENN class   -> {_display(path, settings)}")
                return module.EvidentialNetwork
        except Exception:
            continue
    sys.exit("[error] could not find a module defining EvidentialNetwork.")


def find_agent_dir(settings: RuntimeArtifactSettings | None = None) -> Path:
    """Locate configured bundled policy assets (model/ + actions/)."""
    settings = _settings(settings)
    if settings.agent_dir:
        return Path(settings.agent_dir)
    preferred = [
        settings.assets_dir / settings.env_name,
        settings.assets_dir / "network36",
    ]
    base = settings.root / "curriculumagent"
    invalid = []
    nested = sorted(path for path in base.rglob("*") if path.is_dir())
    for directory in [*preferred, base, *nested]:
        if (directory / "model").is_dir() and (directory / "actions").is_dir():
            if not has_valid_saved_model(directory):
                invalid.append(directory)
                continue
            print(f"       auto: agent dir   -> {_display(directory, settings)}")
            return directory
    hint = ""
    if invalid:
        bad = ", ".join(_display(path, settings) for path in invalid)
        hint = f"\n        Skipped invalid SavedModel artifact(s): {bad}."
    sys.exit(
        "[error] no valid folder with model/ and actions/ subfolders "
        "found. Expected a non-empty TensorFlow SavedModel under "
        f"assets/{settings.env_name}/, assets/network36/ or "
        "another configured active agent directory. Set AGENT_DIR in "
        f"the CONFIG block to override.{hint}"
    )


def scaler_from_json(path: Path):
    """Rebuild the training-time StandardScaler from JSON parameters."""
    from sklearn.preprocessing import StandardScaler

    payload = json.loads(Path(path).read_text())
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(payload["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(payload["scale"], dtype=np.float64)
    scaler.var_ = np.asarray(payload["var"], dtype=np.float64)
    scaler.n_features_in_ = int(payload["n_features_in"])
    return scaler


def load_enn(
    weights: Path,
    meta: dict,
    device: str = "cpu",
    settings: RuntimeArtifactSettings | None = None,
):
    """Instantiate the recorded ENN architecture and load its checkpoint."""
    import torch

    network_class = import_evidential_network(settings)
    enn = network_class(
        input_dim=int(meta["input_dim"]),
        num_classes=int(meta["num_classes"]),
        hidden_dim=int(meta.get("hidden_dim", 256)),
        dropout=float(meta.get("dropout", 0.05)),
    )
    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    enn.load_state_dict(state)
    enn.to(device).eval()
    return enn


def load_agent(
    env,
    agent_dir: Path | None = None,
    *,
    factory_spec: str | None = None,
):
    """Load a configured policy without coupling callers to an entry script."""
    from src.agent_runtime import build_agent

    spec = AGENT_FACTORY if factory_spec is None else factory_spec
    return build_agent(
        env,
        factory_spec=spec or None,
        agent_path=agent_dir if not spec else None,
    )


def enn_uncertainty_fn(scaler):
    """Build the ENN vacuity callback used by failure-forecast features."""
    import torch

    def uncertainty_fn(model, vector):
        scaled = scaler.transform(vector).astype(np.float32)
        with torch.no_grad():
            return float(model(torch.as_tensor(scaled))["uncertainty"][0].item())

    return uncertainty_fn


def load_enn_uncertainty(
    settings: RuntimeArtifactSettings | None = None,
):
    """Return the trained ENN and matching raw-observation uncertainty callback."""
    settings = _settings(settings)
    scaler_json, meta_json, _ = find_artifact_set(settings)
    meta = json.loads(meta_json.read_text())
    enn = load_enn(
        find_enn_weights(prefer_dir=meta_json.parent, settings=settings),
        meta,
        settings=settings,
    )
    return enn, enn_uncertainty_fn(scaler_from_json(scaler_json))
