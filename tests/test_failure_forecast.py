"""Synthetic checks for the historical-label failure-forecast module.

Run with:
    python -m pytest tests/test_failure_forecast.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.failure_forecast import (  # noqa: E402
    CLASSIFIER_FEATURES,
    HORIZON_STEPS,
    FailureForecastConfig,
    FailureForecastPredictor,
    _fixed_tuning_indices,
    _split_classifier_rows,
    build_forecast_pairs,
    collect_failure_forecast_rows,
    collect_forecaster_training_data,
    find_attack_events,
    forecast_injections,
    get_features_with_history,
    label_attack_event,
    prepare_classifier_dataframe,
    run_historical_episode,
    train_failure_classifier,
)

LINE_NAMES = ["line_ok", "line_bad", "line_trip"]


class FakeObs:
    """Observation whose injections encode the step, so alignment is checkable."""

    def __init__(self, step, line_status):
        self.step = step
        self.load_p = np.asarray([10.0 + step], dtype=np.float32)
        self.load_q = np.asarray([2.0 + step], dtype=np.float32)
        self.gen_p = np.asarray([20.0 + step], dtype=np.float32)
        self.rho = np.asarray([0.4, 0.5, 0.97], dtype=np.float32)
        self.line_status = np.asarray(line_status, dtype=bool)
        self.day = 7
        self.hour_of_day = 10
        self.minute_of_hour = 5
        self.day_of_week = 2

    def to_vect(self):
        return np.asarray([self.step, *self.rho], dtype=np.float32)


class FakeEnv:
    """Scripted episode.

    ``attacks``/``trips`` map step -> line ids that go down at that step (an
    attack reports ``opponent_attack_line``, a trip does not).  The episode ends
    with a blackout at ``blackout_at`` or naturally at ``natural_end``.
    """

    reward_range = (0.0, 1.0)
    n_load = 1
    n_gen = 1
    gen_pmin = np.asarray([0.0])
    gen_pmax = np.asarray([50.0])
    name_line = LINE_NAMES

    def __init__(self, attacks=None, trips=None, blackout_at=None, natural_end=None):
        self.attacks = attacks or {}
        self.trips = trips or {}
        self.blackout_at = blackout_at
        self.natural_end = natural_end
        self.t = 0
        self.down = set()

    def reset(self, seed=None):
        self.t = 0
        self.down = set()
        return FakeObs(0, [True] * 3)

    def step(self, action):
        del action
        self.t += 1
        attacked = np.zeros(3, dtype=bool)
        for line in self.attacks.get(self.t, []):
            attacked[line] = True
            self.down.add(line)
        self.down.update(self.trips.get(self.t, []))
        status = [i not in self.down for i in range(3)]
        info = {"exception": [], "opponent_attack_line": attacked if attacked.any() else None}
        done = False
        if self.t == self.blackout_at:
            done, info["exception"] = True, [RuntimeError("blackout")]
        elif self.t == self.natural_end:
            done = True
        return FakeObs(self.t, status), 0.0, done, info


class FakeAgent:
    def __init__(self):
        self.resets = 0

    def reset(self, obs):
        self.resets += 1

    def act(self, obs, reward=0.0, done=False):
        return "noop"


class RecordingMeanModel:
    def __init__(self):
        self.inputs = []

    def predict(self, rows):
        assert len(rows) == 1
        self.inputs.append(np.asarray(rows[0]))
        return np.asarray([[11.0, 4.0, 99.0]], dtype=np.float32)


class FakeAleatoricModel:
    def predict(self, rows):
        assert len(rows) == 1
        return np.log1p(np.asarray([[1.0, 4.0, 9.0]], dtype=np.float32))


class RecordingClassifier:
    classes_ = np.asarray([0, 1])

    def __init__(self):
        self.rows = []

    def predict_proba(self, rows):
        self.rows.append(rows.copy())
        return np.asarray([[0.25, 0.75]], dtype=float)


def make_cfg(lines=("line_bad",)):
    return FailureForecastConfig.from_env(
        FakeEnv(), env_name="fake", agent_name="agent",
        artifact_dir=Path("unused"), lines_to_test=list(lines),
    )


def collect(env, **kwargs):
    kwargs.setdefault("episodes", 1)
    mean_model = kwargs.pop("mean_model", RecordingMeanModel())
    rows, stats = collect_failure_forecast_rows(
        FakeAgent(), env, mean_model, FakeAleatoricModel(), make_cfg(), **kwargs)
    return rows, stats, mean_model


class ForecastPairTests(unittest.TestCase):
    def test_pairs_are_twelve_steps_apart(self):
        observations = [FakeObs(i, [True] * 3) for i in range(40)]
        X, y = build_forecast_pairs(observations, stride=12, offset=0)
        self.assertEqual(HORIZON_STEPS, 12)
        self.assertEqual(X.shape, (3, 19))
        self.assertEqual(y.shape, (3, 3))
        for row, (origin, target) in enumerate(zip(X[:, 0], y[:, 0])):
            self.assertEqual(target - origin, 12)
            self.assertEqual(y[row, 2], origin + 22)  # gen_p = 20 + (t + 12), origin = 10 + t
        self.assertEqual(list(X[:, 0] - 10), [0, 12, 24])

    def test_short_episode_gives_no_pairs(self):
        observations = [FakeObs(i, [True] * 3) for i in range(12)]
        X, _ = build_forecast_pairs(observations, stride=12, offset=0)
        self.assertEqual(len(X), 0)

    def test_no_pair_crosses_an_episode_boundary(self):
        X, y, episode_ids = collect_forecaster_training_data(
            FakeEnv(natural_end=30), FakeAgent(), episodes=3, return_episode_ids=True)
        self.assertEqual(len(X), 6)
        for x_row, y_row, episode in zip(X, y, episode_ids):
            origin = x_row[0] - 10
            target = y_row[0] - 10
            self.assertEqual(target - origin, 12)
            self.assertLessEqual(target, 29)  # last valid observation of the episode
            self.assertEqual(origin % 12, episode % 12)  # rotating phase

    def test_history_features_use_only_the_past(self):
        observations = [FakeObs(i, [True] * 3) for i in range(2017)]
        x = get_features_with_history(observations)
        self.assertEqual(x[0], observations[2016].load_p[0])
        self.assertEqual(x[3], observations[2004].load_p[0])
        self.assertEqual(x[6], observations[1728].load_p[0])
        self.assertEqual(x[9], observations[0].load_p[0])
        early = get_features_with_history(observations[:5])
        self.assertTrue(np.isnan(early[3:12]).all())


class EpisodeTraceTests(unittest.TestCase):
    def test_blackout_episode_excludes_invalid_final_observation(self):
        trace = run_historical_episode(FakeEnv(blackout_at=20), FakeAgent(), seed=0)
        self.assertTrue(trace.ended_in_blackout)
        self.assertEqual(len(trace.observations), 20)

    def test_natural_end_is_not_a_blackout(self):
        trace = run_historical_episode(FakeEnv(natural_end=20), FakeAgent(), seed=0)
        self.assertFalse(trace.ended_in_blackout)

    def test_max_steps_truncates_without_blackout(self):
        trace = run_historical_episode(FakeEnv(blackout_at=50), FakeAgent(), 0, max_steps=15)
        self.assertEqual(len(trace.observations), 16)
        self.assertFalse(trace.ended_in_blackout)

    def test_agent_is_reset_with_first_observation(self):
        agent = FakeAgent()
        run_historical_episode(FakeEnv(natural_end=5), agent, seed=0)
        self.assertEqual(agent.resets, 1)


class LabelTests(unittest.TestCase):
    def labels(self, env, **kwargs):
        rows, stats, _ = collect(env, **kwargs)
        return rows, stats

    def test_blackout_after_attack_is_positive(self):
        rows, stats = self.labels(FakeEnv(attacks={30: [1]}, blackout_at=36))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["failed"], row["label_reason"]), (1, "blackout_after_attack"))
        self.assertEqual((row["event_step"], row["step"]), (30, 18))
        self.assertEqual(row["line_disconnected"], "line_bad")
        self.assertEqual((stats["positive"], stats["negative"]), (1, 0))

    def test_survived_attack_is_negative(self):
        rows, _ = self.labels(FakeEnv(attacks={30: [1]}, natural_end=80))
        self.assertEqual([(r["failed"], r["label_reason"]) for r in rows],
                         [(0, "survived_after_attack")])

    def test_blackout_after_the_window_is_negative(self):
        rows, _ = self.labels(FakeEnv(attacks={30: [1]}, blackout_at=60))
        self.assertEqual([r["failed"] for r in rows], [0])

    def test_outcome_window_is_configurable(self):
        env = FakeEnv(attacks={30: [1]}, blackout_at=60)
        rows, _ = self.labels(env, outcome_steps=40)
        self.assertEqual([r["failed"] for r in rows], [1])

    def test_unobserved_outcome_is_not_labeled(self):
        for env, kwargs in (
            (FakeEnv(attacks={30: [1]}, natural_end=38), {}),
            (FakeEnv(attacks={30: [1]}, blackout_at=200), {"max_steps": 35}),
        ):
            rows, stats = self.labels(env, **kwargs)
            self.assertEqual(rows, [])
            self.assertEqual(stats["skipped_outcome_not_observed"], 1)
            self.assertEqual(stats["negative"], 0)

    def test_overlapping_attacks_are_not_labeled(self):
        rows, stats = self.labels(FakeEnv(attacks={30: [1], 33: [0]}, natural_end=80),
                                  )
        self.assertEqual(rows, [])
        self.assertEqual(stats["skipped_overlapping_attack"], 1)

    def test_attack_causing_immediate_blackout_is_not_labeled(self):
        rows, stats = self.labels(FakeEnv(attacks={30: [1]}, blackout_at=30))
        self.assertEqual(rows, [])
        self.assertEqual(stats["attack_events"], 0)

    def test_attack_without_forecast_origin_is_skipped(self):
        rows, stats = self.labels(FakeEnv(attacks={5: [1]}, natural_end=80))
        self.assertEqual(rows, [])
        self.assertEqual(stats["skipped_no_forecast_origin"], 1)

    def test_trips_and_repeated_attacks_are_not_events(self):
        env = FakeEnv(attacks={20: [1], 60: [1]}, trips={25: [2]}, natural_end=100)
        trace = run_historical_episode(env, FakeAgent(), 0)
        events = find_attack_events(trace, make_cfg(LINE_NAMES), LINE_NAMES)
        self.assertEqual(events, [(20, 1)])

    def test_lines_outside_the_configured_set_are_ignored(self):
        trace = run_historical_episode(
            FakeEnv(attacks={30: [0]}, natural_end=80), FakeAgent(), 0)
        self.assertEqual(find_attack_events(trace, make_cfg(["line_bad"]), ["line_bad"]), [])

    def test_label_function_directly(self):
        trace = run_historical_episode(FakeEnv(attacks={30: [1]}, blackout_at=36), FakeAgent(), 0)
        self.assertEqual(label_attack_event(trace, 30, 1, 12), (1, "blackout_after_attack"))
        self.assertEqual(label_attack_event(trace, 30, 1, 3), (0, "survived_after_attack"))
        self.assertEqual(label_attack_event(trace, 30, 1, 20), (1, "blackout_after_attack"))


class RowConstructionTests(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv(attacks={30: [1]}, natural_end=80)
        self.rows, _, self.mean_model = collect(self.env)

    def test_row_uses_the_forecast_origin_and_horizon(self):
        row = self.rows[0]
        self.assertEqual(row["horizon_steps"], 12)
        self.assertEqual(row["horizon_minutes"], 60)
        self.assertEqual(row["event_step"] - row["step"], HORIZON_STEPS)
        self.assertEqual(row["forecast_source"], "forecaster")

    def test_no_future_leakage(self):
        row = self.rows[0]
        origin = row["step"]
        x = self.mean_model.inputs[0]
        self.assertEqual(x[0], 10.0 + origin)  # forecaster saw obs at t, not t+12
        self.assertEqual(row["sum_gen_p"], 20.0 + origin)
        self.assertEqual(row["max_line_rho"], np.float32(0.97))
        self.assertEqual(row["fcast_sum_gen_p"], 50.0)  # model output, clipped

    def test_features_do_not_change_when_the_future_changes(self):
        trace = run_historical_episode(FakeEnv(natural_end=80), FakeAgent(), 0)
        origin = 18
        before = get_features_with_history(trace.observations[: origin + 1])
        for obs in trace.observations[origin + 1:]:
            obs.load_p = obs.load_p * 1000
        after = get_features_with_history(trace.observations[: origin + 1])
        np.testing.assert_array_equal(before, after)

    def test_perfect_forecast_benchmark_uses_measured_future(self):
        rows, _, model = collect(
            FakeEnv(attacks={30: [1]}, natural_end=80), perfect_forecast=True)
        self.assertEqual(rows[0]["forecast_source"], "measured_future")
        self.assertEqual(rows[0]["fcast_sum_gen_p"], 20.0 + 30)
        self.assertEqual(model.inputs, [])

    def test_stats_and_progress_callback(self):
        calls = []
        _, stats, _ = collect(
            FakeEnv(attacks={30: [1]}, blackout_at=36), episodes=2,
            progress_callback=lambda ep, steps, st: calls.append((ep, steps, st["rows"])))
        self.assertEqual(stats["episodes"], 2)
        self.assertEqual(stats["blackout_episodes"], 2)
        self.assertEqual(calls, [(0, 36, 1), (1, 36, 2)])


class ForecastTests(unittest.TestCase):
    def test_forecast_shapes_and_clipping(self):
        observations = [FakeObs(i, [True] * 3) for i in range(20)]
        forecast = forecast_injections(
            observations, RecordingMeanModel(), FakeAleatoricModel(), make_cfg())
        self.assertEqual(forecast["load_p"].shape, (1,))
        self.assertEqual(forecast["gen_p"][0], 50.0)
        self.assertGreater(forecast["aleatoric_gen_p_mean"], forecast["aleatoric_load_p_mean"])
        self.assertTrue(np.isnan(forecast["epistemic_before"]))

    def test_enn_requires_an_uncertainty_function(self):
        observations = [FakeObs(0, [True] * 3)]
        with self.assertRaises(ValueError):
            forecast_injections(observations, RecordingMeanModel(), FakeAleatoricModel(),
                                make_cfg(), model_enn=object())


class PredictorTests(unittest.TestCase):
    def make_predictor(self, classifier=None):
        metadata = {"features": CLASSIFIER_FEATURES,
                    "line_map": {"ok": 0, "bad": 1}, "decision_threshold": 0.5}
        return FailureForecastPredictor(classifier or RecordingClassifier(), metadata)

    def test_inference_matches_training_features(self):
        env = FakeEnv(attacks={30: [1]}, natural_end=80)
        rows, _, _ = collect(env)
        trace = run_historical_episode(FakeEnv(natural_end=80), FakeAgent(), 0)
        history = trace.observations[: rows[0]["step"] + 1]
        classifier = RecordingClassifier()
        predictor = self.make_predictor(classifier)
        predictions = predictor.predict_many(
            history[-1], history, ["line_bad"], RecordingMeanModel(),
            FakeAleatoricModel(), make_cfg())
        self.assertEqual(predictions, [{
            "line": "line_bad", "failure_prediction": 1, "failure_probability": 0.75}])
        used = classifier.rows[0]
        self.assertEqual(list(used.columns), CLASSIFIER_FEATURES)
        training = predictor.features_from_row(rows[0])
        pd.testing.assert_frame_equal(used.reset_index(drop=True), training, check_dtype=False)

    def test_one_forecast_serves_all_lines_and_skips_disconnected_ones(self):
        mean_model = RecordingMeanModel()
        obs = FakeObs(30, [True, True, False])
        predictions = self.make_predictor().predict_many(
            obs, [obs], ["line_ok", "line_bad", "line_trip"] if False else ["line_ok", "line_bad"],
            mean_model, FakeAleatoricModel(), make_cfg(["line_ok", "line_bad"]))
        self.assertEqual([p["line"] for p in predictions], ["line_ok", "line_bad"])
        self.assertEqual(len(mean_model.inputs), 1)
        disconnected = self.make_predictor().predict_many(
            obs, [obs], ["line_bad"], RecordingMeanModel(), FakeAleatoricModel(),
            make_cfg(["line_bad"]))
        self.assertEqual(len(disconnected), 1)
        obs_down = FakeObs(30, [True, False, True])
        self.assertEqual(self.make_predictor().predict_many(
            obs_down, [obs_down], ["line_bad"], RecordingMeanModel(),
            FakeAleatoricModel(), make_cfg()), [])

    def test_unknown_line_raises(self):
        obs = FakeObs(30, [True] * 3)
        with self.assertRaises(ValueError):
            self.make_predictor().predict_many(
                obs, [obs], ["line_trip"], RecordingMeanModel(), FakeAleatoricModel(),
                make_cfg(["line_trip"]))

    def test_history_must_end_with_current_observation(self):
        obs = FakeObs(30, [True] * 3)
        with self.assertRaises(ValueError):
            self.make_predictor().predict_many(
                obs, [FakeObs(29, [True] * 3)], ["line_bad"], RecordingMeanModel(),
                FakeAleatoricModel(), make_cfg())

    def test_missing_feature_and_old_artifacts_are_rejected(self):
        predictor = self.make_predictor()
        with self.assertRaises(ValueError):
            predictor.features_from_row({"line_disconnected": "line_bad"})
        with self.assertRaises(ValueError):
            FailureForecastPredictor(RecordingClassifier(), {"features": ["old"]})


class ClassifierTests(unittest.TestCase):
    @staticmethod
    def synthetic_rows(episodes=12, per_episode=4):
        rng = np.random.default_rng(0)
        rows = []
        for episode in range(episodes):
            for _ in range(per_episode):
                failed = int(episode % 2 == 0)
                row = {name: float(rng.normal()) + failed for name in CLASSIFIER_FEATURES
                       if name not in ("line_id_encoded", "load_gen_ratio")}
                row.update({
                    "episode": episode, "line_disconnected": "line_bad",
                    "failed": failed, "label_reason": "blackout_after_attack" if failed
                    else "survived_after_attack", "outcome_steps": 12,
                })
                rows.append(row)
        return pd.DataFrame(rows)

    def test_train_save_and_reload_round_trip(self):
        df = prepare_classifier_dataframe(self.synthetic_rows())
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure_classifier.pkl"
            _, info = train_failure_classifier(df, model_path=path, seed=1, max_iter=20)
            self.assertEqual(info["features"], CLASSIFIER_FEATURES)
            self.assertEqual(info["outcome_steps"], 12)
            self.assertEqual(info["label_counts"], {"1": 24, "0": 24})
            self.assertEqual(info["forecast_horizon_steps"], 12)
            predictor = FailureForecastPredictor.load(path)
            result = predictor.predict_from_features(self.synthetic_rows().iloc[0].to_dict())
            self.assertIn(result["failure_prediction"], (0, 1))
            self.assertTrue(0.0 <= result["failure_probability"] <= 1.0)

    def test_split_keeps_episodes_apart(self):
        df = prepare_classifier_dataframe(self.synthetic_rows())
        X_train, X_test, _, _, info = _split_classifier_rows(df, seed=3)
        train_eps = set(df.loc[X_train.index, "episode"])
        test_eps = set(df.loc[X_test.index, "episode"])
        self.assertFalse(train_eps & test_eps)
        self.assertEqual(info["strategy"], "episode_grouped")

    def test_missing_columns_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare_classifier_dataframe(self.synthetic_rows().drop(columns=["fcast_sum_gen_p"]))

    def test_tuning_split_is_episode_disjoint(self):
        episode_ids = np.repeat(np.arange(10), 5)
        train_idx, val_idx = _fixed_tuning_indices(
            len(episode_ids), episode_ids, max_subsample=100, seed=0)
        self.assertFalse(set(episode_ids[train_idx]) & set(episode_ids[val_idx]))


if __name__ == "__main__":
    unittest.main()
