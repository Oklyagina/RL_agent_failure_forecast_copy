"""Training-data sources for the Evidential Neural Network (ENN).

The original project assumed CurriculumAgent tutor splits were always present.
This module makes the source explicit and generic:

* ``tutor``: use existing train/validation/test NPZ files when available;
* ``rollout``: run the configured policy in Grid2Op and behavior-clone its
  observation/action pairs;
* ``auto`` (default): prefer tutor data, otherwise collect policy rollouts.

Only the ``act`` interface is required from the agent.  A custom agent can be
provided via an injected factory or ``AGENT_FACTORY=module:function``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    from .agent_runtime import build_agent, call_agent
except ImportError:  # pragma: no cover - script execution via src/ on PYTHONPATH
    from agent_runtime import build_agent, call_agent


@dataclass
class ENNDataBundle:
    train: Tuple[np.ndarray, np.ndarray]
    validation: Tuple[np.ndarray, np.ndarray]
    test: Tuple[np.ndarray, np.ndarray]
    source: str
    action_set: Optional[np.ndarray] = None
    action_set_path: Optional[Path] = None


def load_npz_split(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a tutor-style NPZ split using tolerant state/action key matching."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(str(path), allow_pickle=False) as data:
        keys = list(data.files)
        state_key = next((k for k in keys if k.startswith("s_") or k in {"obs", "observations"}), None)
        action_key = next((k for k in keys if k.startswith("a_") or k in {"act", "actions", "labels"}), None)
        if state_key is None or action_key is None:
            raise ValueError(
                f"Invalid ENN split {path}: expected state/action arrays, found keys {keys}."
            )
        obs = np.asarray(data[state_key], dtype=np.float32)
        actions = np.asarray(data[action_key], dtype=np.int64).reshape(-1)
    if obs.ndim != 2 or actions.ndim != 1 or len(obs) == 0 or len(obs) != len(actions):
        raise ValueError(
            f"Invalid ENN split {path}: obs shape={obs.shape}, action shape={actions.shape}."
        )
    return obs, actions


def tutor_bundle_if_available(cfg: Any) -> Optional[ENNDataBundle]:
    """Return valid tutor data, or ``None`` when any required split is absent."""
    paths = [Path(cfg.TRAIN_FILE), Path(cfg.VAL_FILE), Path(cfg.TEST_FILE)]
    if not all(path.is_file() for path in paths):
        return None
    train, validation, test = (load_npz_split(path) for path in paths)

    tutor_dir = Path(getattr(cfg, "TUTOR_DIR", paths[0].parent))
    action_candidates = [
        Path(cfg.AGENT_PATH) / "actions" / "actions.npy",
        tutor_dir / "actions.npy",
        tutor_dir.parent / "actions.npy",
        tutor_dir.parent / "actions" / "actions.npy",
    ]
    action_path = next((path for path in action_candidates if path.is_file()), None)
    action_set = None
    if action_path is not None:
        arr = np.load(action_path, allow_pickle=False)
        if arr.ndim == 2:
            action_set = np.asarray(arr, dtype=np.float32)
        else:
            action_path = None

    return ENNDataBundle(
        train=train,
        validation=validation,
        test=test,
        source="tutor",
        action_set=action_set,
        action_set_path=action_path,
    )


def _stable_unique_rows(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Deduplicate 2-D rows while preserving first-observed action order."""
    mapping: Dict[bytes, int] = {}
    unique = []
    labels = np.empty(len(rows), dtype=np.int64)
    for i, row in enumerate(np.asarray(rows)):
        contiguous = np.ascontiguousarray(row)
        key = contiguous.tobytes()
        label = mapping.get(key)
        if label is None:
            label = len(unique)
            mapping[key] = label
            unique.append(contiguous.copy())
        labels[i] = label
    return np.stack(unique).astype(np.float32), labels


def _split_indices(labels: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create robust 70/15/15-ish splits while keeping every class in train."""
    n = len(labels)
    if n < 3:
        raise ValueError("At least three rollout samples are required to split ENN data.")
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels, dtype=np.int64)

    # Anchor one example of every observed action in the training split. This
    # prevents a random split from silently deleting rare policy actions from
    # the ENN class mapping.
    anchors = []
    remaining_mask = np.ones(n, dtype=bool)
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        chosen = int(rng.choice(candidates))
        anchors.append(chosen)
        remaining_mask[chosen] = False

    remaining = rng.permutation(np.flatnonzero(remaining_mask))
    if len(remaining) < 2:
        raise ValueError(
            "Rollout data need at least two non-anchor samples to create validation and "
            "test splits while retaining every observed action in training. Collect more rollouts."
        )

    target_test = max(1, int(round(0.15 * n)))
    target_val = max(1, int(round(0.15 * n)))
    n_test = min(target_test, len(remaining) - 1)
    n_val = min(target_val, len(remaining) - n_test)
    if n_val < 1:
        n_val = 1
        n_test = max(1, len(remaining) - 1)

    test_idx = remaining[:n_test]
    val_idx = remaining[n_test:n_test + n_val]
    train_idx = np.concatenate([np.asarray(anchors, dtype=np.int64), remaining[n_test + n_val:]])
    train_idx = rng.permutation(train_idx)
    return train_idx, val_idx, test_idx


def collect_policy_rollouts(
    env: Any,
    agent: Any,
    *,
    episodes: int,
    seed: int = 0,
    max_steps: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``agent`` and return observations, integer labels, and action vectors."""
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps <= 0:
            max_steps = None

    obs_rows = []
    action_rows = []
    for episode in range(episodes):
        try:
            obs = env.reset(seed=seed + episode)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed + episode)
            obs = env.reset()

        if hasattr(agent, "reset"):
            try:
                agent.reset(obs)
            except Exception:
                pass

        reward = getattr(env, "reward_range", (0.0, 0.0))[0]
        done = False
        step = 0
        while not done and (max_steps is None or step < max_steps):
            action = call_agent(agent, obs, reward=float(reward), done=done)
            obs_rows.append(np.asarray(obs.to_vect(), dtype=np.float32))
            action_rows.append(np.asarray(action.to_vect(), dtype=np.float32))
            obs, reward, done, _ = env.step(action)
            step += 1
        if progress_callback is not None:
            progress_callback(episode, step, len(obs_rows))

    if not obs_rows:
        raise RuntimeError("Agent rollout collection produced no observation/action pairs.")

    observations = np.stack(obs_rows).astype(np.float32)
    action_vectors = np.stack(action_rows).astype(np.float32)
    action_set, labels = _stable_unique_rows(action_vectors)
    if action_set.shape[0] < 2:
        raise RuntimeError(
            "Agent rollouts contained fewer than two distinct actions. An ENN trained on a "
            "single action cannot provide a meaningful policy-familiarity uncertainty signal."
        )
    return observations, labels, action_set


def save_rollout_bundle(
    out_dir: Path,
    observations: np.ndarray,
    labels: np.ndarray,
    action_set: np.ndarray,
    *,
    seed: int,
) -> ENNDataBundle:
    """Persist generic rollout data and return train/validation/test views."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "observations.npy", observations)
    np.save(out_dir / "labels.npy", labels.astype(np.int64))
    np.save(out_dir / "actions.npy", action_set.astype(np.float32))

    tr_idx, va_idx, te_idx = _split_indices(labels, seed)
    splits = {
        "train": (observations[tr_idx], labels[tr_idx]),
        "validation": (observations[va_idx], labels[va_idx]),
        "test": (observations[te_idx], labels[te_idx]),
    }
    # Also save tutor-compatible NPZ files so older tooling can inspect/reuse them.
    np.savez(out_dir / "train.npz", obs=splits["train"][0], act=splits["train"][1])
    np.savez(out_dir / "validation.npz", obs=splits["validation"][0], act=splits["validation"][1])
    np.savez(out_dir / "test.npz", obs=splits["test"][0], act=splits["test"][1])

    return ENNDataBundle(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        source="agent_rollout",
        action_set=action_set,
        action_set_path=out_dir / "actions.npy",
    )


def load_or_collect_enn_data(
    cfg: Any,
    *,
    source: str = "auto",
    rollout_dir: Optional[Path] = None,
    episodes: int = 50,
    seed: int = 0,
    max_steps: Optional[int] = None,
    env_factory: Optional[Callable[[], Any]] = None,
    agent_factory: Optional[Callable[[Any], Any]] = None,
    agent_factory_spec: Optional[str] = None,
) -> ENNDataBundle:
    """Resolve ENN data from tutor files or by executing the configured policy."""
    source = source.lower().strip()
    if source not in {"auto", "tutor", "rollout"}:
        raise ValueError("source must be one of: auto, tutor, rollout")

    if source in {"auto", "tutor"}:
        tutor = tutor_bundle_if_available(cfg)
        if tutor is not None and tutor.action_set is not None:
            return tutor
        if source == "tutor":
            if tutor is None:
                raise FileNotFoundError(
                    "Tutor data was explicitly requested but one or more ENN tutor splits are missing."
                )
            raise FileNotFoundError(
                "Tutor splits exist, but no compatible 2-D policy action set was found. "
                "The action vectors are required to preserve the ENN class/action contract."
            )

    if rollout_dir is None:
        configured = getattr(cfg, "ENN_ROLLOUT_DIR", None)
        rollout_dir = Path(configured) if configured else Path(cfg.MODEL_ENN_PATH).parent / "enn_rollouts"

    if env_factory is None:
        import grid2op
        try:
            from lightsim2grid import LightSimBackend
            env_factory = lambda: grid2op.make(cfg.ENV_NAME, backend=LightSimBackend())
        except ImportError:  # pragma: no cover
            env_factory = lambda: grid2op.make(cfg.ENV_NAME)

    env = env_factory()
    try:
        agent = build_agent(
            env,
            factory=agent_factory,
            factory_spec=agent_factory_spec,
            agent_path=getattr(cfg, "AGENT_PATH", None),
        )
        observations, labels, actions = collect_policy_rollouts(
            env, agent, episodes=episodes, seed=seed, max_steps=max_steps
        )
    finally:
        try:
            env.close()
        except Exception:
            pass

    return save_rollout_bundle(
        rollout_dir,
        observations,
        labels,
        actions,
        seed=seed,
    )
