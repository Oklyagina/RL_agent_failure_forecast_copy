"""Synthetic checks for the failure dataset CSV helpers.

Run with:
    python tests/test_failure_dataset.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.failure_dataset import (  # noqa: E402
    collect_fresh_failure_rows,
    iter_artifact_feature_rows,
    map_failure_label,
    rollout_artifacts_available,
    write_failure_dataset_csv,
)


class FakeAction:
    def __init__(self, value: float = 0.0):
        self.value = value

    def to_vect(self):
        return np.array([self.value], dtype=np.float32)


class FakeObs:
    def __init__(self, current_step: int = 0, max_step: int = 3, rho: tuple[float, float] = (0.4, 0.8)):
        self.current_step = current_step
        self.max_step = max_step
        self.month = 1
        self.day = 2
        self.hour_of_day = 3
        self.minute_of_hour = 4
        self.day_of_week = 5
        self.rho = np.asarray(rho, dtype=np.float32)
        self.line_status = np.array([True, False])
        self.timestep_overflow = np.array([0, 1])
        self.load_p = np.array([10.0])
        self.load_q = np.array([2.0])
        self.load_v = np.array([141.0])
        self.gen_p = np.array([12.0])
        self.gen_q = np.array([1.0])
        self.gen_v = np.array([142.0])
        self.p_or = np.array([1.0, 2.0])
        self.q_or = np.array([3.0, 4.0])
        self.v_or = np.array([5.0, 6.0])
        self.a_or = np.array([7.0, 8.0])
        self.p_ex = np.array([9.0, 10.0])
        self.q_ex = np.array([11.0, 12.0])
        self.v_ex = np.array([13.0, 14.0])
        self.a_ex = np.array([15.0, 16.0])
        self.topo_vect = np.array([1, 2, 1])
        self.time_before_cooldown_line = np.array([0, 3])
        self.time_before_cooldown_sub = np.array([0])
        self.time_next_maintenance = np.array([-1, 4])
        self.duration_next_maintenance = np.array([0, 2])

    def to_vect(self):
        return np.array([self.current_step, self.rho[0], self.rho[1]], dtype=np.float32)

    def copy(self):
        return FakeObs(self.current_step, self.max_step, tuple(self.rho.tolist()))

    def from_vect(self, vector):
        self.current_step = int(vector[0])
        self.rho = np.asarray(vector[1:3], dtype=np.float32)
        return self


class FakeAgent:
    def act(self, obs, reward=0.0, done=False):
        return FakeAction(float(obs.current_step))


class FakeEnv:
    reward_range = (0.0, 1.0)

    def __init__(self):
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        return FakeObs(current_step=0, max_step=3)

    def step(self, action):
        del action
        self.step_count += 1
        done = self.step_count >= 2
        # Ends at step 2 while max_step is 3, so it is an abnormal stop.
        obs = FakeObs(current_step=self.step_count, max_step=3)
        return obs, float(self.step_count), done, {"exception": []}


def test_failure_mapping():
    assert map_failure_label(False, FakeObs(current_step=1, max_step=3), {}) == 0
    assert map_failure_label(True, FakeObs(current_step=3, max_step=3), {}) == 0
    assert map_failure_label(True, FakeObs(current_step=2, max_step=3), {}) == 1


def test_fresh_rows_and_csv():
    rows = collect_fresh_failure_rows(FakeEnv(), FakeAgent(), episodes=1, seed=0)
    assert len(rows) == 2
    assert rows[0]["failure"] == 0
    assert rows[1]["failure"] == 1
    assert "rho_0" in rows[0]
    assert "load_p_0" in rows[0]

    with TemporaryDirectory() as raw_tmp:
        output = Path(raw_tmp) / "failure_dataset.csv"
        count = write_failure_dataset_csv(rows, output)
        assert count == 2
        with output.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames is not None
            assert "failure" in reader.fieldnames
            assert "rho_1" in reader.fieldnames
            loaded = list(reader)
        assert loaded[-1]["failure"] == "1"


def test_artifact_feature_rows_do_not_invent_failure_label():
    with TemporaryDirectory() as raw_tmp:
        rollout_dir = Path(raw_tmp)
        observations = np.array(
            [
                [0.0, 0.2, 0.3],
                [1.0, 0.4, 0.5],
            ],
            dtype=np.float32,
        )
        labels = np.array([1, 0], dtype=np.int64)
        actions = np.array([[0.0], [1.0]], dtype=np.float32)
        np.save(rollout_dir / "observations.npy", observations)
        np.save(rollout_dir / "labels.npy", labels)
        np.save(rollout_dir / "actions.npy", actions)

        assert rollout_artifacts_available(rollout_dir)
        rows = list(iter_artifact_feature_rows(rollout_dir, FakeObs()))
        assert len(rows) == 2
        assert rows[0]["agent_action_class"] == 1
        assert rows[0]["rho_0"] == np.float32(0.2)
        assert "failure" not in rows[0]


def main() -> None:
    test_failure_mapping()
    test_fresh_rows_and_csv()
    test_artifact_feature_rows_do_not_invent_failure_label()
    print("test_failure_dataset: PASSED")


if __name__ == "__main__":
    main()
