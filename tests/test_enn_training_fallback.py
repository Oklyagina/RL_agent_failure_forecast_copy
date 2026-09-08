"""Regression test: the real ENN trainer must work without tutor files."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import training_enn as T


class Action:
    def __init__(self, v):
        self.v = np.asarray(v, dtype=np.float32)

    def to_vect(self):
        return self.v


class Obs:
    def __init__(self, step):
        self.current_step = step
        self._v = np.asarray(
            [step % 2, step / 10.0, np.sin(step), np.cos(step)], dtype=np.float32
        )

    def to_vect(self):
        return self._v


class Env:
    reward_range = (-1.0, 1.0)

    def __init__(self):
        self.t = 0

    def reset(self, seed=None):
        self.t = 0
        return Obs(0)

    def step(self, action):
        self.t += 1
        return Obs(self.t), 0.0, self.t >= 10, {}

    def close(self):
        pass


class Agent:
    def act(self, obs, reward=0.0, done=False):
        return Action([1, 0, 0] if obs.current_step % 2 else [0, 1, 0])


def _run_smoke(td: Path) -> None:
    cfg = T.CFG
    names = [
        "MODEL_ENN_PATH", "TRAIN_FILE", "VAL_FILE", "TEST_FILE", "AGENT_PATH",
        "ENV_NAME", "AGENT_NAME", "ENN_ROLLOUT_DIR", "ENN_ROLLOUT_EPISODES",
        "ENN_EPOCHS", "ENN_BATCH_SIZE", "ENN_PATIENCE", "ENN_TOP_K", "ENN_DROPOUT",
        "ENN_NOISE_STD", "ENN_WARMUP", "AGENT_FACTORY", "SEED",
    ]
    sentinel = object()
    old = {name: getattr(cfg, name, sentinel) for name in names}
    try:
        cfg.MODEL_ENN_PATH = str(td / "model" / "enn_36.pth")
        cfg.TRAIN_FILE = str(td / "missing_train.npz")
        cfg.VAL_FILE = str(td / "missing_val.npz")
        cfg.TEST_FILE = str(td / "missing_test.npz")
        cfg.AGENT_PATH = str(td / "no_agent_assets")
        cfg.ENV_NAME = "fake_env"
        cfg.AGENT_NAME = "fake_agent"
        cfg.ENN_ROLLOUT_DIR = str(td / "rollouts")
        cfg.ENN_ROLLOUT_EPISODES = 4
        cfg.ENN_EPOCHS = 2
        cfg.ENN_BATCH_SIZE = 8
        cfg.ENN_PATIENCE = 3
        cfg.ENN_TOP_K = 2
        cfg.ENN_DROPOUT = 0.0
        cfg.ENN_NOISE_STD = 0.0
        cfg.ENN_WARMUP = 0
        cfg.AGENT_FACTORY = ""
        cfg.SEED = 0

        T.train_enn(
            top_k=2,
            data_source="auto",
            rollout_episodes=4,
            env_factory=Env,
            agent_factory=lambda env: Agent(),
        )

        model_dir = td / "model"
        for name in (
            "enn_36.pth", "scaler_fake_env_enn.pkl", "enn_meta_fake_env.json",
            "enn_meta.json", "scaler_params.json", "enn_pctile_calib.npz", "actions.npy",
        ):
            assert (model_dir / name).is_file(), name
        meta = json.loads((model_dir / "enn_meta.json").read_text())
        assert meta["data_source"] == "agent_rollout"
        assert meta["num_classes"] == 2
    finally:
        for name, value in old.items():
            if value is sentinel:
                delattr(cfg, name)
            else:
                setattr(cfg, name, value)


def test_enn_training_without_tutor(tmp_path: Path):
    _run_smoke(tmp_path)


def main():
    with tempfile.TemporaryDirectory() as td:
        _run_smoke(Path(td))
    print("test_enn_training_fallback: PASSED")


if __name__ == "__main__":
    main()
