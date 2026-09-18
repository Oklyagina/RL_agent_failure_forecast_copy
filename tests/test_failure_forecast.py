"""Synthetic checks for the active failure-forecast module.

Run with:
    python tests/test_failure_forecast.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.failure_forecast import (  # noqa: E402
    CLASSIFIER_FEATURES,
    FailureForecastConfig,
    FailureForecastPredictor,
    collect_failure_forecast_rows,
    get_features_with_history,
    prepare_classifier_dataframe,
    run_t12_forecast,
    train_failure_classifier,
)


class FakeAction:
    def __init__(self, kind="noop", line_id=None):
        self.kind = kind
        self.line_id = line_id

    def to_vect(self):
        code = {"noop": 0, "disconnect": 1, "agent": 2}[self.kind]
        return np.asarray([code, -1 if self.line_id is None else self.line_id], dtype=np.float32)


class FakeActionSpace:
    def __call__(self, payload):
        if not payload:
            return FakeAction()
        status = payload["set_line_status"]
        if isinstance(status, list):
            line_id = int(status[0][0])
        else:
            line_id = int(np.flatnonzero(np.asarray(status) == -1)[0])
        return FakeAction("disconnect", line_id)


class FakeObs:
    def __init__(self, step=0, *, attacked=False):
        self.current_step = step
        self.max_step = 100
        self.load_p = np.asarray([10.0 + step], dtype=np.float32)
        self.load_q = np.asarray([2.0 + step], dtype=np.float32)
        self.gen_p = np.asarray([20.0 + step], dtype=np.float32)
        self.gen_q = np.asarray([3.0], dtype=np.float32)
        self.gen_v = np.asarray([142.0], dtype=np.float32)
        self.rho = np.asarray([0.40, 0.96 if attacked else 0.50], dtype=np.float32)
        self.day = 7
        self.hour_of_day = 10
        self.minute_of_hour = 5
        self.day_of_week = 2
        self._forecasted_inj = []
        self.attacked = attacked

    def to_vect(self):
        return np.asarray([self.current_step, *self.rho], dtype=np.float32)

    def copy(self):
        copied = FakeObs(self.current_step, attacked=self.attacked)
        copied._forecasted_inj = list(self._forecasted_inj)
        return copied

    def get_time_stamp(self):
        return dt.datetime(2026, 1, 1, self.hour_of_day, self.minute_of_hour)

    def simulate(self, action):
        if action.kind == "noop":
            obs = FakeObs(self.current_step + 12)
            obs._forecasted_inj = list(self._forecasted_inj)
            return obs, 0.0, False, {"exception": []}
        if action.kind == "disconnect":
            return FakeObs(self.current_step + 1, attacked=True), 0.0, False, {"exception": []}
        if action.kind == "agent":
            return FakeObs(self.current_step + 1, attacked=True), 0.0, True, {
                "exception": [RuntimeError("blackout")]
            }
        raise AssertionError(action.kind)


class FakeEnv:
    action_space = FakeActionSpace()
    reward_range = (0.0, 1.0)
    n_load = 1
    n_gen = 1
    n_line = 2
    gen_pmin = np.asarray([0.0])
    gen_pmax = np.asarray([50.0])
    name_line = ["line_ok", "line_bad"]

    def reset(self, seed=None):
        del seed
        self.step_count = 0
        return FakeObs(0)

    def step(self, action):
        del action
        self.step_count += 1
        return FakeObs(self.step_count), 0.0, self.step_count >= 14, {"exception": []}


class FakeAgent:
    def act(self, obs, reward=0.0, done=False):
        del obs, reward, done
        return FakeAction("agent")


class FakeMeanModel:
    def predict(self, rows):
        assert len(rows) == 1
        return np.asarray([[11.0, 4.0, 99.0]], dtype=np.float32)


class FakeAleatoricModel:
    def predict(self, rows):
        assert len(rows) == 1
        return np.log1p(np.asarray([[1.0, 4.0, 9.0]], dtype=np.float32))


def _cfg():
    return FailureForecastConfig.from_env(
        FakeEnv(),
        env_name="fake_grid",
        agent_name="curriculum",
        artifact_dir=Path("unused"),
        lines_to_test=["line_bad"],
    )


def test_history_features_and_forecast_injection():
    obs = FakeObs(2016)
    observations = [FakeObs(i) for i in range(2017)]
    x = get_features_with_history(observations, obs)
    assert x.shape == (19,)
    assert x[0] == obs.load_p[0]
    assert x[3] == observations[2004].load_p[0]
    assert x[6] == observations[1728].load_p[0]
    assert x[9] == observations[0].load_p[0]

    forecast = run_t12_forecast(
        FakeEnv(), obs, observations, FakeMeanModel(), FakeAleatoricModel(), _cfg()
    )
    assert forecast["predicted_gen_p"][0] == 50.0
    injections = forecast["forecast_obs"]._forecasted_inj
    assert len(injections) == 2
    assert injections[1][0] - injections[0][0] == dt.timedelta(minutes=60)
    assert injections[1][1]["injection"]["prod_p"][0] == 50.0
    assert forecast["aleatoric_gen_p_mean"] > forecast["aleatoric_load_p_mean"]


def test_collection_labels_agent_response_blackout():
    rows = collect_failure_forecast_rows(
        FakeEnv(),
        FakeAgent(),
        FakeMeanModel(),
        FakeAleatoricModel(),
        _cfg(),
        episodes=1,
        sampling_stride=12,
        max_steps=13,
    )
    assert len(rows) == 1
    assert rows[0]["failed"] == 1
    assert rows[0]["label_reason"] == "agent_action_blackout"
    assert rows[0]["line_disconnected"] == "line_bad"


def test_classifier_metadata_and_prediction_round_trip():
    try:
        import joblib  # noqa: F401
        import pandas as pd
        import sklearn  # noqa: F401
    except ModuleNotFoundError as exc:
        print(f"test_classifier_metadata_and_prediction_round_trip: SKIPPED ({exc})")
        return

    rows = []
    for i in range(30):
        line = "line_bad" if i % 2 else "line_ok"
        failed = 1 if i >= 15 else 0
        row = {
            "line_disconnected": line,
            "failed": failed,
        }
        for j, feature in enumerate(CLASSIFIER_FEATURES):
            if feature in {"line_id_encoded", "load_gen_ratio"}:
                continue
            row[feature] = float(i + j)
        rows.append(row)
    df = prepare_classifier_dataframe(pd.DataFrame(rows))

    with TemporaryDirectory() as raw_tmp:
        model_path = Path(raw_tmp) / "failure_classifier.pkl"
        _, info = train_failure_classifier(
            df,
            model_path=model_path,
            threshold=0.4,
            seed=0,
            env_name="fake_grid",
            agent_name="curriculum",
            max_iter=20,
        )
        predictor = FailureForecastPredictor.load(model_path, info["metadata_path"])
        out = predictor.predict_from_features(df.iloc[-1].to_dict())
        assert set(out) == {"failure_prediction", "failure_probability"}
        assert out["failure_prediction"] in {0, 1}
        assert 0.0 <= out["failure_probability"] <= 1.0
        assert predictor.metadata["environment"] == "fake_grid"


def main() -> None:
    test_history_features_and_forecast_injection()
    test_collection_labels_agent_response_blackout()
    test_classifier_metadata_and_prediction_round_trip()
    print("test_failure_forecast: PASSED")


if __name__ == "__main__":
    main()
