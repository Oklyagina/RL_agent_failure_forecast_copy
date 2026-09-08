"""Targeted regression tests for the generic ENN fallback and public risk APIs."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from recommendation_uncertainty import confidence_level, uncertainty_level
from src.enn_data import load_or_collect_enn_data
from src.failure_probability import FailureProbabilityPredictor


class FakeAction:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def to_vect(self):
        return self.vector


class FakeObs:
    def __init__(self, vector, step=0):
        self._vector = np.asarray(vector, dtype=np.float32)
        self.current_step = step
        self.load_p = np.asarray([10.0, 11.0], dtype=float)
        self.load_q = np.asarray([2.0, 3.0], dtype=float)
        self.gen_p = np.asarray([22.0], dtype=float)
        self.gen_q = np.asarray([0.5], dtype=float)
        self.rho = np.asarray([0.5, 0.9, 1.05], dtype=float)
        self.name_line = np.asarray(["L0", "L1", "L2"])

    def to_vect(self):
        return self._vector


class FakeEnv:
    reward_range = (-1.0, 1.0)

    def __init__(self, steps=8):
        self.steps = steps
        self.t = 0

    def reset(self, seed=None):
        self.t = 0
        return FakeObs([0.0, 1.0, 0.0, 1.0], 0)

    def step(self, action):
        self.t += 1
        obs = FakeObs([self.t % 2, 1.0, float(self.t), -float(self.t)], self.t)
        return obs, 0.0, self.t >= self.steps, {}

    def close(self):
        pass


class FakeAgent:
    def act(self, obs, reward=0.0, done=False):
        if int(obs.current_step) % 2:
            return FakeAction([1.0, 0.0, 0.0])
        return FakeAction([0.0, 1.0, 0.0])


class DummyCfg:
    AGENT_PATH = "/does/not/matter"
    ENV_NAME = "fake"
    MODEL_ENN_PATH = "/tmp/fake/enn.pth"
    TRAIN_FILE = "/missing/train.npz"
    VAL_FILE = "/missing/val.npz"
    TEST_FILE = "/missing/test.npz"


class FakeClassifier:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, frame):
        # deterministic probability based on max rho; exercises DataFrame contract
        p = float(np.clip(frame.iloc[0]["max_line_rho"] / 1.2, 0.0, 1.0))
        return np.asarray([[1.0 - p, p]])


def test_rollout_fallback(tmp_path: Path):
    bundle = load_or_collect_enn_data(
        DummyCfg,
        source="auto",
        rollout_dir=tmp_path / "rollouts",
        episodes=3,
        seed=5,
        env_factory=lambda: FakeEnv(steps=8),
        agent_factory=lambda env: FakeAgent(),
    )
    assert bundle.source == "agent_rollout"
    assert bundle.action_set.shape == (2, 3)
    assert len(bundle.train[0]) > 0
    assert (tmp_path / "rollouts" / "observations.npy").is_file()
    assert (tmp_path / "rollouts" / "actions.npy").is_file()


def test_failure_probability():
    predictor = FailureProbabilityPredictor(
        FakeClassifier(),
        metadata={
            "line_map": {"L0": 0, "L1": 1, "L2": 2},
            "decision_threshold": 0.7,
        },
    )
    obs = FakeObs([1, 2, 3, 4], step=10)
    probability = predictor.predict_proba(obs, "L1")
    assert 0.0 <= probability <= 1.0
    detail = predictor.predict(obs, 1)
    assert detail["line"] == "L1"
    assert abs(detail["failure_probability"] - probability) < 1e-12
    assert detail["failure_probability_pct"] == round(probability * 100.0, 2)
    try:
        predictor.predict_proba(obs, "UNSEEN_LINE")
    except ValueError as exc:
        assert "training data" in str(exc)
    else:
        raise AssertionError("unseen categorical line must be rejected")


def test_kpi_bands():
    assert uncertainty_level(0.0) == "low"
    assert confidence_level(0.0) == "high"
    assert uncertainty_level(50.0) == "medium"
    assert confidence_level(50.0) == "medium"
    assert uncertainty_level(100.0) == "high"
    assert confidence_level(100.0) == "low"


def main():
    with tempfile.TemporaryDirectory() as td:
        test_rollout_fallback(Path(td))
    test_failure_probability()
    test_kpi_bands()
    print("test_new_components: PASSED")


if __name__ == "__main__":
    main()
