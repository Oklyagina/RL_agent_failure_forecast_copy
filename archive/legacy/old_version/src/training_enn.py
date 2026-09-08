"""Training and inference utilities for the Evidential Neural Network (ENN).

The ENN is trained by behavior cloning of the policy.  Data source resolution is
``auto`` by default: existing tutor splits are preferred for backwards
compatibility; if they are absent, the configured agent is executed in Grid2Op
and rollout data are collected automatically.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

try:
    from .config import CFG, DEVICE, ENN_PARAMS
    from .enn_models import EvidentialNetwork, evidential_loss
    from .enn_data import load_npz_split, load_or_collect_enn_data
except ImportError:  # pragma: no cover - direct script execution
    from config import CFG, DEVICE, ENN_PARAMS
    from enn_models import EvidentialNetwork, evidential_loss
    from enn_data import load_npz_split, load_or_collect_enn_data


# ==============================================================================
# Artifact paths
# ==============================================================================

def _model_dir() -> Path:
    return Path(CFG.MODEL_ENN_PATH).parent


def _safe_env_token() -> str:
    return Path(str(CFG.ENV_NAME)).name or "environment"


def _scaler_path() -> str:
    return str(_model_dir() / f"scaler_{_safe_env_token()}_enn.pkl")


def _best_weights_path() -> str:
    return str(_model_dir() / f"enn_best_{_safe_env_token()}.pth")


def _meta_path() -> str:
    return str(_model_dir() / f"enn_meta_{_safe_env_token()}.json")


def _portable_meta_path() -> Path:
    return _model_dir() / "enn_meta.json"


def _portable_scaler_path() -> Path:
    return _model_dir() / "scaler_params.json"


def _calibration_path() -> Path:
    return _model_dir() / "enn_pctile_calib.npz"


def _portable_action_path() -> Path:
    return _model_dir() / "actions.npy"


# ==============================================================================
# Inference
# ==============================================================================

def get_uncertainty(model: EvidentialNetwork, obs_array: np.ndarray) -> float:
    """Return ENN vacuity in [0, 1] for one or more raw observation vectors."""
    scaler = _load_scaler()
    input_dim = int(getattr(model, "input_dim", scaler.n_features_in_))
    arr = np.asarray(obs_array, dtype=np.float32)
    if arr.size % input_dim != 0:
        raise ValueError(
            f"Observation contains {arr.size} values, which cannot be reshaped to ENN input_dim={input_dim}."
        )
    obs_scaled = scaler.transform(arr.reshape(-1, input_dim))
    obs_tensor = torch.as_tensor(obs_scaled, dtype=torch.float32, device=DEVICE)

    model.eval()
    with torch.no_grad():
        u = model(obs_tensor)["uncertainty"].squeeze(-1)
    return float(u.mean().item())


def load_trained_enn() -> EvidentialNetwork:
    """Initialize and load a trained ENN using persisted architecture metadata."""
    meta = _load_enn_meta()
    num_classes = int(meta.get("num_classes", ENN_PARAMS["num_classes"]))
    input_dim = meta.get("input_dim")
    if input_dim is None:
        try:
            input_dim = int(_load_scaler().n_features_in_)
        except (FileNotFoundError, AttributeError):
            input_dim = ENN_PARAMS["input_dim"]

    model = EvidentialNetwork(
        input_dim=int(input_dim),
        num_classes=num_classes,
        hidden_dim=int(meta.get("hidden_dim", ENN_PARAMS["hidden_dim"])),
        dropout=float(meta.get("dropout", CFG.ENN_DROPOUT)),
    ).to(DEVICE)

    load_errors = []
    for path in (_best_weights_path(), CFG.MODEL_ENN_PATH):
        if os.path.exists(path):
            try:
                state = torch.load(path, map_location=DEVICE)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state)
                break
            except (RuntimeError, KeyError, ValueError) as exc:
                load_errors.append(f"{path}: {exc}")
    else:
        detail = "; ".join(load_errors) or "no checkpoint files exist"
        raise RuntimeError(f"No compatible ENN checkpoint could be loaded: {detail}")

    model.eval()
    return model


# ==============================================================================
# Scaler & metadata
# ==============================================================================

def _load_scaler() -> StandardScaler:
    path = _scaler_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scaler not found at {path}. Train the ENN bundle before inference."
        )
    return joblib.load(path)


def _fit_and_save_scaler(obs: np.ndarray) -> StandardScaler:
    scaler = StandardScaler().fit(obs)
    path = Path(_scaler_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    return scaler


def _load_enn_meta() -> dict:
    for path in (Path(_meta_path()), _portable_meta_path()):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {"num_classes": CFG.ENN_NUM_CLASSES}


def _load_tutor_split(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper around the generic split loader."""
    return load_npz_split(path)


def _write_portable_scaler(scaler: StandardScaler) -> None:
    payload = {
        "type": "StandardScaler",
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_features_in": int(scaler.n_features_in_),
    }
    _portable_scaler_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ==============================================================================
# Mathematical/data helpers
# ==============================================================================

def _remap_topk(
    act_tr: np.ndarray,
    act_val: Optional[np.ndarray],
    act_te: Optional[np.ndarray],
    top_k: int,
) -> Tuple:
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    counts = Counter(np.asarray(act_tr, dtype=np.int64).tolist())
    topk = [cls for cls, _ in counts.most_common(top_k)]
    if not topk:
        raise ValueError("ENN training split contains no action labels.")
    mapping = {old: new for new, old in enumerate(topk)}

    def apply(arr):
        if arr is None:
            return None, None
        arr = np.asarray(arr, dtype=np.int64)
        mask = np.isin(arr, topk)
        remapped = np.fromiter((mapping[int(a)] for a in arr[mask]), dtype=np.int64)
        return remapped, mask

    act_tr_r, mask_tr = apply(act_tr)
    act_val_r, mask_val = apply(act_val)
    act_te_r, mask_te = apply(act_te)
    return act_tr_r, act_val_r, act_te_r, mask_tr, mask_val, mask_te, mapping, len(topk)


def _print_split_diagnostics(split_rows: Dict[str, Tuple[np.ndarray, np.ndarray]], source: str) -> None:
    print(f"\n[ENN] Data diagnostics (source={source})")
    for name, (_, actions) in split_rows.items():
        print(f"  {name:10s}: rows={len(actions):6d}, unique_actions={len(np.unique(actions)):4d}")

    train_actions = set(split_rows["train"][1].tolist())
    for name in ("validation", "test"):
        other = set(split_rows[name][1].tolist())
        overlap = len(train_actions & other)
        print(
            f"  train/{name:10s} action overlap: {overlap}/{len(other)} "
            f"({len(other - train_actions)} outside train)"
        )
    all_actions = np.concatenate([actions for _, actions in split_rows.values()])
    top_counts = ", ".join(
        f"{int(action)}:{count}" for action, count in Counter(all_actions.tolist()).most_common(10)
    )
    print(f"  top action frequencies: {top_counts}")


def _print_topk_diagnostics(
    act_tr: np.ndarray,
    act_val: Optional[np.ndarray],
    act_te: Optional[np.ndarray],
    mask_tr: np.ndarray,
    mask_val: Optional[np.ndarray],
    mask_te: Optional[np.ndarray],
    n_cls: int,
) -> None:
    val_rows = 0 if mask_val is None else int(mask_val.sum())
    test_rows = 0 if mask_te is None else int(mask_te.sum())
    print("\n[ENN] Top-K filter diagnostics")
    print(f"  num_classes={n_cls}, chance_accuracy={1.0 / max(n_cls, 1):.4f}")
    print(
        f"  retained rows: train={int(mask_tr.sum())}/{len(mask_tr)}, "
        f"validation={val_rows}/{0 if mask_val is None else len(mask_val)}, "
        f"test={test_rows}/{0 if mask_te is None else len(mask_te)}"
    )
    print(
        "  retained unique actions: "
        f"train={len(np.unique(act_tr))}, "
        f"validation={0 if act_val is None else len(np.unique(act_val))}, "
        f"test={0 if act_te is None else len(np.unique(act_te))}"
    )


def compute_effective_weights(actions: np.ndarray, num_classes: int, beta: float = 0.999) -> torch.Tensor:
    counts = np.bincount(actions, minlength=num_classes)
    counts = np.maximum(counts, 1)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * num_classes
    return torch.as_tensor(weights, dtype=torch.float32, device=DEVICE)


class NoisyDataset(Dataset):
    def __init__(self, obs, actions, noise_std):
        self.obs = torch.as_tensor(obs, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.long)
        self.noise_std = float(noise_std)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        x = self.obs[idx]
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x, self.actions[idx]


def _cosine_with_warmup(optimizer, warmup, total):
    def lr_fn(epoch):
        if epoch < warmup:
            return epoch / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def _loader_batch_size(n_rows: int) -> int:
    if n_rows < 2:
        raise ValueError("ENN training needs at least two retained training rows.")
    return min(int(CFG.ENN_BATCH_SIZE), n_rows)


def _export_uncertainty_artifacts(
    model: EvidentialNetwork,
    scaler: StandardScaler,
    calibration_rows_scaled: np.ndarray,
    meta: dict,
    action_set: Optional[np.ndarray],
) -> None:
    """Export the portable artifact names used by the API and example scripts."""
    _model_dir().mkdir(parents=True, exist_ok=True)
    _write_portable_scaler(scaler)

    if action_set is not None:
        np.save(_portable_action_path(), np.asarray(action_set, dtype=np.float32))
        meta["action_set"] = str(_portable_action_path())
        meta["n_curated_actions"] = int(len(action_set))

    meta_text = json.dumps(meta, indent=2)
    Path(_meta_path()).write_text(meta_text, encoding="utf-8")
    _portable_meta_path().write_text(meta_text, encoding="utf-8")

    # Keep a stable, agent-labelled alias without duplicating training logic.
    agent_name = str(getattr(CFG, "AGENT_NAME", "agent"))
    portable_weights = _model_dir() / f"enn_{agent_name}.pth"
    if Path(CFG.MODEL_ENN_PATH).resolve() != portable_weights.resolve():
        shutil.copyfile(CFG.MODEL_ENN_PATH, portable_weights)

    from recommendation_uncertainty import build_calibration, save_calibration

    model_cpu = model.to("cpu").eval()
    total_ref, action_ref = build_calibration(model_cpu, calibration_rows_scaled)
    save_calibration(_calibration_path(), total_ref, action_ref)
    model.to(DEVICE)


# ==============================================================================
# Main training loop
# ==============================================================================

def train_enn(
    top_k: int = CFG.ENN_TOP_K,
    *,
    data_source: str = "auto",
    rollout_episodes: Optional[int] = None,
    rollout_max_steps: Optional[int] = None,
    env_factory: Optional[Callable[[], Any]] = None,
    agent_factory: Optional[Callable[[Any], Any]] = None,
    agent_factory_spec: Optional[str] = None,
) -> EvidentialNetwork:
    """Train an ENN from tutor data or policy rollouts.

    ``data_source='auto'`` uses configured tutor splits when they are available.
    Otherwise it executes the configured agent and collects behavior-cloning
    data directly; ``curriculumagent/tutor`` output is not required.
    """
    episodes = int(
        rollout_episodes
        if rollout_episodes is not None
        else getattr(CFG, "ENN_ROLLOUT_EPISODES", 50)
    )
    rollout_dir = Path(
        getattr(CFG, "ENN_ROLLOUT_DIR", _model_dir() / "enn_rollouts")
    )
    max_steps = rollout_max_steps
    if max_steps is None:
        max_steps = getattr(CFG, "ENN_ROLLOUT_MAX_STEPS", None)
    bundle = load_or_collect_enn_data(
        CFG,
        source=data_source,
        rollout_dir=rollout_dir,
        episodes=episodes,
        seed=int(getattr(CFG, "SEED", 0)),
        max_steps=max_steps,
        env_factory=env_factory,
        agent_factory=agent_factory,
        agent_factory_spec=(agent_factory_spec or getattr(CFG, "AGENT_FACTORY", None)),
    )
    print(f"[ENN] Data source: {bundle.source}")
    if bundle.source == "agent_rollout":
        print(f"[ENN] Tutor data unavailable; collected policy rollouts in {rollout_dir}")

    obs_tr, act_tr = bundle.train
    obs_val, act_val = bundle.validation
    obs_te, act_te = bundle.test
    if bundle.action_set is not None:
        all_action_ids = np.concatenate([act_tr, act_val, act_te]).astype(np.int64)
        if len(all_action_ids) and (
            int(all_action_ids.min()) < 0 or int(all_action_ids.max()) >= len(bundle.action_set)
        ):
            raise ValueError(
                "ENN action labels reference rows outside the supplied action set: "
                f"label range=[{int(all_action_ids.min())}, {int(all_action_ids.max())}], "
                f"action rows={len(bundle.action_set)}."
            )
    split_rows = {
        "train": (obs_tr, act_tr),
        "validation": (obs_val, act_val),
        "test": (obs_te, act_te),
    }
    _print_split_diagnostics(split_rows, bundle.source)

    input_dim = int(obs_tr.shape[1])
    for split_name, (obs, _) in split_rows.items():
        if obs.ndim != 2 or obs.shape[1] != input_dim:
            raise ValueError(
                f"ENN {split_name} feature width {obs.shape[1] if obs.ndim == 2 else obs.shape} "
                f"does not match training width {input_dim}."
            )

    (
        act_tr, act_val, act_te,
        mask_tr, mask_val, mask_te,
        cls_mapping, n_cls,
    ) = _remap_topk(act_tr, act_val, act_te, top_k=top_k)
    _print_topk_diagnostics(act_tr, act_val, act_te, mask_tr, mask_val, mask_te, n_cls)

    obs_tr = obs_tr[mask_tr]
    obs_val = obs_val[mask_val] if mask_val is not None else None
    obs_te = obs_te[mask_te] if mask_te is not None else None
    if obs_val is not None and len(obs_val) == 0:
        print("[ENN] Validation split has no retained top-K actions; using test split for evaluation.")
        obs_val, act_val = None, None
    if obs_te is not None and len(obs_te) == 0:
        print("[ENN] Test split has no retained top-K actions; disabling test evaluation.")
        obs_te, act_te = None, None

    class_weights = compute_effective_weights(act_tr, num_classes=n_cls, beta=0.999)
    print(
        f"[ENN] Class weights: min={class_weights.min().item():.3f}, "
        f"max={class_weights.max().item():.3f}"
    )

    scaler = _fit_and_save_scaler(obs_tr)
    obs_tr_s = scaler.transform(obs_tr).astype(np.float32)
    obs_val_s = scaler.transform(obs_val).astype(np.float32) if obs_val is not None else None
    obs_te_s = scaler.transform(obs_te).astype(np.float32) if obs_te is not None else None

    batch_size = _loader_batch_size(len(obs_tr_s))
    drop_last = len(obs_tr_s) > batch_size and len(obs_tr_s) % batch_size == 1
    train_loader = DataLoader(
        NoisyDataset(obs_tr_s, act_tr, noise_std=CFG.ENN_NOISE_STD),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=drop_last,
    )
    val_loader = None
    if obs_val_s is not None:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(obs_val_s), torch.as_tensor(act_val, dtype=torch.long)),
            batch_size=min(batch_size, len(obs_val_s)),
            shuffle=False,
        )
    test_loader = None
    if obs_te_s is not None:
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(obs_te_s), torch.as_tensor(act_te, dtype=torch.long)),
            batch_size=min(batch_size, len(obs_te_s)),
            shuffle=False,
        )

    model = EvidentialNetwork(
        input_dim=input_dim,
        num_classes=n_cls,
        hidden_dim=ENN_PARAMS["hidden_dim"],
        dropout=CFG.ENN_DROPOUT,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.ENN_MAX_LR, weight_decay=CFG.ENN_WEIGHT_DECAY
    )
    scheduler = _cosine_with_warmup(optimizer, CFG.ENN_WARMUP, CFG.ENN_EPOCHS)

    best_val_loss = float("inf")
    best_epoch = None
    patience_counter = 0
    best_path = Path(_best_weights_path())
    best_path.unlink(missing_ok=True)

    print("\n" + "=" * 80)
    print(f"  ENN TRAINING — policy behavior cloning ({bundle.source})")
    print("=" * 80)
    t0 = time.time()

    progress = tqdm(range(1, CFG.ENN_EPOCHS + 1), desc="Training ENN", unit="epoch")
    for epoch in progress:
        model.train()
        tr_loss = tr_correct = tr_n = 0
        for obs_b, act_b in train_loader:
            obs_b, act_b = obs_b.to(DEVICE), act_b.to(DEVICE)
            out = model(obs_b)
            loss = evidential_loss(
                out["alpha"], act_b, epoch, CFG.ENN_EPOCHS,
                class_weights=class_weights, lam=CFG.ENN_EDL_LAMBDA,
                anneal_epochs=getattr(CFG, "ENN_ANNEAL_EPOCHS", 20),
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = obs_b.size(0)
            tr_loss += float(loss.item()) * bs
            tr_correct += int((out["prob"].argmax(-1) == act_b).sum().item())
            tr_n += bs
        if tr_n == 0:
            raise RuntimeError("ENN training DataLoader yielded no batches.")
        scheduler.step()

        eval_loader = val_loader if val_loader is not None else test_loader
        val_info = {}
        if eval_loader is not None:
            model.eval()
            v_loss = v_correct = v_n = 0
            uncertainties = []
            with torch.no_grad():
                for obs_b, act_b in eval_loader:
                    obs_b, act_b = obs_b.to(DEVICE), act_b.to(DEVICE)
                    out = model(obs_b)
                    loss = evidential_loss(
                        out["alpha"], act_b, epoch, CFG.ENN_EPOCHS,
                        class_weights=class_weights, lam=CFG.ENN_EDL_LAMBDA,
                        anneal_epochs=getattr(CFG, "ENN_ANNEAL_EPOCHS", 20),
                    )
                    bs = obs_b.size(0)
                    v_loss += float(loss.item()) * bs
                    v_correct += int((out["prob"].argmax(-1) == act_b).sum().item())
                    v_n += bs
                    uncertainties.append(out["uncertainty"].squeeze(-1).cpu())
            if v_n:
                u_arr = torch.cat(uncertainties)
                val_info = {
                    "val_loss": v_loss / v_n,
                    "val_acc": v_correct / v_n,
                    "u_mean": float(u_arr.mean()),
                    "u_std": float(u_arr.std(unbiased=False)),
                }
                if val_info["val_loss"] < best_val_loss - 1e-5:
                    best_val_loss = val_info["val_loss"]
                    patience_counter = 0
                    best_epoch = epoch
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), best_path)
                else:
                    patience_counter += 1

        postfix = {
            "tr_loss": f"{tr_loss / tr_n:.4f}",
            "tr_acc": f"{tr_correct / tr_n:.3f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
            "elapsed": f"{time.time() - t0:.0f}s",
        }
        if val_info:
            postfix.update({
                "val_loss": f"{val_info['val_loss']:.4f}",
                "val_acc": f"{val_info['val_acc']:.3f}",
                "U": f"{val_info['u_mean']:.3f}+/-{val_info['u_std']:.3f}",
            })
        progress.set_postfix(postfix)
        if patience_counter >= CFG.ENN_PATIENCE:
            tqdm.write(f"\n[ENN] Early stopping at epoch {epoch}.")
            break

    if best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    if best_epoch is not None:
        print(f"[ENN] Best epoch: {best_epoch} | validation loss: {best_val_loss:.4f}")

    _model_dir().mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CFG.MODEL_ENN_PATH)

    action_set = bundle.action_set
    # Class mapping is original action id -> retained ENN label.
    meta = {
        "input_dim": input_dim,
        "num_classes": n_cls,
        "hidden_dim": int(ENN_PARAMS["hidden_dim"]),
        "dropout": float(CFG.ENN_DROPOUT),
        "class_mapping": {str(k): int(v) for k, v in cls_mapping.items()},
        "top_k": n_cls,
        "environment": str(CFG.ENV_NAME),
        "agent": str(getattr(CFG, "AGENT_NAME", "agent")),
        "data_source": bundle.source,
    }
    if bundle.action_set_path is not None:
        meta["source_action_set"] = str(bundle.action_set_path)

    calibration_rows = obs_val_s if obs_val_s is not None and len(obs_val_s) else obs_tr_s
    n_cal = min(len(calibration_rows), 5000)
    calibration_rows = calibration_rows[:n_cal]
    _export_uncertainty_artifacts(model, scaler, calibration_rows, meta, action_set)
    print(f"[ENN] Wrote model, scaler, metadata, action set and calibration in {_model_dir()}")
    return model


if __name__ == "__main__":
    model = train_enn(top_k=CFG.ENN_TOP_K)
    print("\n[ENN] Pipeline Completed.")
