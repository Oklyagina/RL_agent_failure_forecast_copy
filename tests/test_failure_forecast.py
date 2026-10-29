"""Synthetic checks for the legacy-logic failure-forecast module (t+12 horizon).

Run with:
    python -m pytest tests/test_failure_forecast.py
"""
from __future__ import annotations

import copy
import datetime
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.failure_forecast import (  # noqa: E402
    CLASSIFIER_FEATURES,
    HORIZON_STEPS,
    LABEL_DISCONNECTION_TEST,
    LABEL_EPISODE_ENDED,
    LEGACY_PHASES,
    FailureForecastConfig,
    FailureForecastPredictor,
    _classifier_f2,
    _f2_optimal_threshold,
    _fit_multioutput_with_progress,
    _metrics,
    _fixed_tuning_indices,
    _split_classifier_rows,
    build_forecast_pairs,
    collect_failure_forecast_phases,
    collect_failure_forecast_rows,
    collect_forecaster_training_data,
    forecast_injections,
    get_features_with_history,
    prepare_classifier_dataframe,
    run_historical_episode,
    train_failure_classifier,
    train_mean_forecaster,
)

LINE_NAMES = ["line_ok", "line_bad", "line_trip"]
START = datetime.datetime(2020, 1, 1)


class FakeObs:
    """Observation whose injections encode the step, so alignment is checkable."""

    gen_v = np.asarray([1.0], dtype=np.float32)

    def __init__(self, step, line_status):
        self.step = step
        self.current_step = step
        self.load_p = np.asarray([10.0 + step], dtype=np.float32)
        self.load_q = np.asarray([2.0 + step], dtype=np.float32)
        self.gen_p = np.asarray([20.0 + step], dtype=np.float32)
        self.rho = np.asarray([0.4, 0.5, 0.97], dtype=np.float32)
        self.line_status = np.asarray(line_status, dtype=bool)
        self.day = 7
        self.hour_of_day = 10
        self.minute_of_hour = 5
        self.day_of_week = 2
        self._forecasted_inj = []

    def to_vect(self):
        return np.asarray([self.step, *self.rho], dtype=np.float32)

    def get_time_stamp(self):
        return START + datetime.timedelta(minutes=5 * self.step)

    def copy(self):
        return copy.copy(self)

    def simulate(self, action):
        """Powerflow stand-in: the state takes the injected t+12 injections."""
        del action
        injection = self._forecasted_inj[1][1]["injection"]
        sim = FakeObs(self.step + 1, self.line_status)
        sim.load_p = np.asarray(injection["load_p"], dtype=np.float32)
        sim.load_q = np.asarray(injection["load_q"], dtype=np.float32)
        sim.gen_p = np.asarray(injection["prod_p"], dtype=np.float32)
        sim.rho = np.asarray([0.5, 0.6, 0.99], dtype=np.float32)
        return sim, 0.0, False, {}


class FakeEnv:
    """Scripted episode with copy() and forced line disconnection.

    ``fragile`` lines blackout the grid when forcibly disconnected;
    ``trips`` maps step -> line ids that go down by themselves.  The episode
    ends with a blackout at ``blackout_at`` or naturally at ``natural_end``.
    ``forced_at`` records the step at which a disconnection was forced.
    """

    reward_range = (0.0, 1.0)
    n_load = 1
    n_gen = 1
    gen_pmin = np.asarray([0.0])
    gen_pmax = np.asarray([50.0])
    name_line = LINE_NAMES

    def __init__(self, fragile=(), trips=None, blackout_at=None, natural_end=None):
        self.fragile = set(fragile)
        self.trips = trips or {}
        self.blackout_at = blackout_at
        self.natural_end = natural_end
        self.t = 0
        self.down = set()
        self.forced_at = []
        self.closed = False

    def action_space(self, content):
        return dict(content)

    def copy(self):
        clone = copy.copy(self)
        clone.down = set(self.down)
        return clone

    def close(self):
        self.closed = True

    def reset(self, seed=None):
        self.t = 0
        self.down = set()
        return FakeObs(0, [True] * 3)

    def step(self, action):
        self.t += 1
        self.down.update(self.trips.get(self.t, []))
        done, info = False, {"exception": []}
        forced = action.get("set_line_status", []) if isinstance(action, dict) else []
        for line_id, _ in forced:
            self.forced_at.append(self.t)
            self.down.add(line_id)
            if line_id in self.fragile:
                done, info["exception"] = True, [RuntimeError("blackout")]
        if self.t == self.blackout_at:
            done, info["exception"] = True, [RuntimeError("blackout")]
        elif self.t == self.natural_end:
            done = True
        status = [i not in self.down for i in range(3)]
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


def collect(env, lines=("line_bad",), **kwargs):
    kwargs.setdefault("episodes", 1)
    mean_model = kwargs.pop("mean_model", RecordingMeanModel())
    rows, stats = collect_failure_forecast_rows(
        FakeAgent(), env, mean_model, FakeAleatoricModel(), make_cfg(lines), **kwargs)
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

    def test_every_step_is_a_forecast_origin_with_stride_one(self):
        observations = [FakeObs(i, [True] * 3) for i in range(30)]
        X, y = build_forecast_pairs(observations, stride=1, offset=0)
        self.assertEqual(len(X), 30 - HORIZON_STEPS)
        np.testing.assert_array_equal(y[:, 0] - X[:, 0], np.full(len(X), 12.0))

    def test_short_episode_gives_no_pairs(self):
        observations = [FakeObs(i, [True] * 3) for i in range(12)]
        X, _ = build_forecast_pairs(observations, stride=12, offset=0)
        self.assertEqual(len(X), 0)

    def test_collection_defaults_to_every_step_and_never_crosses_episodes(self):
        X, y, episode_ids = collect_forecaster_training_data(
            FakeEnv(natural_end=30), FakeAgent(), episodes=3, return_episode_ids=True)
        self.assertEqual(len(X), 3 * (30 - HORIZON_STEPS))
        for x_row, y_row in zip(X, y):
            self.assertEqual(y_row[0] - x_row[0], 12)
            self.assertLessEqual(y_row[0] - 10, 29)  # last valid observation
        self.assertEqual(sorted(set(episode_ids.tolist())), [0, 1, 2])

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

    def test_agent_is_reset_with_first_observation(self):
        agent = FakeAgent()
        run_historical_episode(FakeEnv(natural_end=5), agent, seed=0)
        self.assertEqual(agent.resets, 1)


class LabelTests(unittest.TestCase):
    def test_survived_forced_disconnection_is_negative(self):
        rows, stats, _ = collect(FakeEnv(natural_end=80))
        self.assertEqual([r["step"] for r in rows], [20, 40, 60])
        self.assertEqual({(r["failed"], r["label_reason"]) for r in rows},
                         {(0, LABEL_DISCONNECTION_TEST)})
        self.assertEqual((stats["positive"], stats["negative"]), (0, 3))
        self.assertEqual(stats["sample_points"], 3)

    def test_fragile_line_blackout_after_forced_disconnection_is_positive(self):
        rows, stats, _ = collect(FakeEnv(fragile={1}, natural_end=80))
        self.assertEqual({(r["failed"], r["label_reason"]) for r in rows},
                         {(1, LABEL_DISCONNECTION_TEST)})
        self.assertEqual(stats["positive"], 3)

    def test_disconnection_is_forced_at_t_plus_12(self):
        env = FakeEnv(natural_end=80)
        collect(env)
        # copies share the log: origin 20/40/60 -> 12 agent steps + the forced step
        self.assertEqual(env.forced_at, [33, 53, 73])
    def test_episode_ended_before_t12_labels_every_line_failed(self):
        rows, stats, _ = collect(FakeEnv(blackout_at=25),
                                 lines=("line_ok", "line_bad"))
        self.assertEqual(len(rows), 2)
        self.assertEqual({(r["failed"], r["label_reason"]) for r in rows},
                         {(1, LABEL_EPISODE_ENDED)})
        self.assertEqual(stats["blackout_episodes"], 1)

    def test_lines_down_at_t_are_skipped(self):
        rows, _, _ = collect(FakeEnv(trips={15: [1]}, natural_end=80),
                             lines=("line_ok", "line_bad"))
        self.assertEqual({r["line_disconnected"] for r in rows}, {"line_ok"})

    def test_sampling_only_after_min_step_every_sample_every(self):
        rows, _, _ = collect(FakeEnv(natural_end=100), sample_every=10, min_step=12)
        self.assertEqual([r["step"] for r in rows], [20, 30, 40, 50, 60, 70, 80, 90])

    def test_row_carries_phase_episode_and_horizon(self):
        rows, _, _ = collect(FakeEnv(natural_end=80), phase="HeavyAttack_1", seed=700)
        row = rows[0]
        self.assertEqual(row["phase"], "HeavyAttack_1")
        self.assertEqual(row["episode"], 700)
        self.assertEqual(row["horizon_steps"], 12)
        self.assertEqual(row["horizon_minutes"], 60)
        self.assertEqual(row["event_step"] - row["step"], HORIZON_STEPS)

    def test_stats_and_progress_callback(self):
        calls = []
        _, stats, _ = collect(
            FakeEnv(blackout_at=25), episodes=2,
            progress_callback=lambda ep, steps, st: calls.append((ep, steps, st["rows"])))
        self.assertEqual(stats["episodes"], 2)
        self.assertEqual(calls, [(0, 25, 1), (1, 25, 2)])


class PhaseTests(unittest.TestCase):
    def test_every_legacy_phase_is_collected_with_its_env_kwargs(self):
        self.assertEqual([name for name, _ in LEGACY_PHASES],
                         ["Standard", "HeavyAttack_1", "HeavyAttack_2"])
        made = []

        def make_env(**kwargs):
            made.append(kwargs)
            return FakeEnv(natural_end=40)

        with mock.patch("src.failure_forecast.opponent_env_kwargs",
                        side_effect=lambda lines, cooldown: {"cooldown": cooldown}):
            rows, stats = collect_failure_forecast_phases(
                FakeAgent(), make_env, RecordingMeanModel(), FakeAleatoricModel(),
                make_cfg(), episodes=1, seed=700)
        self.assertEqual(made, [{}, {"cooldown": 288}, {"cooldown": 128}])
        self.assertEqual([r["phase"] for r in rows],
                         ["Standard", "HeavyAttack_1", "HeavyAttack_2"])
        self.assertEqual({r["episode"] for r in rows}, {700})
        self.assertEqual(stats["episodes"], 3)


class RowConstructionTests(unittest.TestCase):
    def setUp(self):
        self.rows, _, self.mean_model = collect(FakeEnv(natural_end=80))

    def test_features_cover_the_legacy_set(self):
        row = self.rows[0]
        for feature in CLASSIFIER_FEATURES:
            if feature != "line_id_encoded":
                self.assertIn(feature, row)
        self.assertEqual(len(CLASSIFIER_FEATURES), 21)

    def test_no_future_leakage_and_forecast_comes_from_the_simulation(self):
        row = self.rows[0]
        origin = row["step"]
        self.assertEqual(self.mean_model.inputs[0][0], 10.0 + origin)
        self.assertEqual(row["sum_gen_p"], 20.0 + origin)
        self.assertEqual(row["max_line_rho"], np.float32(0.97))
        self.assertEqual(row["fcast_sum_gen_p"], 50.0)  # forecast, clipped, via simulate
        self.assertAlmostEqual(row["fcast_max_line_rho"], 0.99, places=5)
        self.assertEqual(row["fcast_nb_rho_ge_0.95"], 1)
        self.assertTrue(np.isnan(row["epistemic_after"]))

    def test_features_do_not_change_when_the_future_changes(self):
        trace = run_historical_episode(FakeEnv(natural_end=80), FakeAgent(), 0)
        origin = 18
        before = get_features_with_history(trace.observations[: origin + 1])
        for obs in trace.observations[origin + 1:]:
            obs.load_p = obs.load_p * 1000
        after = get_features_with_history(trace.observations[: origin + 1])
        np.testing.assert_array_equal(before, after)


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

    def predict_many(self, predictor, obs, history, lines, cfg=None, mean_model=None):
        return predictor.predict_many(
            obs, history, lines, mean_model or RecordingMeanModel(),
            FakeAleatoricModel(), cfg or make_cfg(lines), env=FakeEnv())

    def test_inference_matches_training_features(self):
        rows, _, _ = collect(FakeEnv(natural_end=80))
        trace = run_historical_episode(FakeEnv(natural_end=80), FakeAgent(), 0)
        history = trace.observations[: rows[0]["step"] + 1]
        classifier = RecordingClassifier()
        predictor = self.make_predictor(classifier)
        predictions = self.predict_many(predictor, history[-1], history, ["line_bad"])
        self.assertEqual(predictions, [{
            "line": "line_bad", "failure_prediction": 1, "failure_probability": 0.75}])
        used = classifier.rows[0]
        self.assertEqual(list(used.columns), CLASSIFIER_FEATURES)
        training = predictor.features_from_row(rows[0])
        pd.testing.assert_frame_equal(used.reset_index(drop=True), training, check_dtype=False)

    def test_one_forecast_serves_all_lines_and_skips_disconnected_ones(self):
        mean_model = RecordingMeanModel()
        obs = FakeObs(30, [True, True, False])
        predictions = self.predict_many(
            self.make_predictor(), obs, [obs], ["line_ok", "line_bad"], mean_model=mean_model)
        self.assertEqual([p["line"] for p in predictions], ["line_ok", "line_bad"])
        self.assertEqual(len(mean_model.inputs), 1)
        obs_down = FakeObs(30, [True, False, True])
        self.assertEqual(
            self.predict_many(self.make_predictor(), obs_down, [obs_down], ["line_bad"]), [])

    def test_unknown_line_raises(self):
        obs = FakeObs(30, [True] * 3)
        with self.assertRaises(ValueError):
            self.predict_many(self.make_predictor(), obs, [obs], ["line_trip"])

    def test_history_must_end_with_current_observation(self):
        obs = FakeObs(30, [True] * 3)
        with self.assertRaises(ValueError):
            self.predict_many(self.make_predictor(), obs, [FakeObs(29, [True] * 3)],
                              ["line_bad"])

    def test_missing_feature_and_old_artifacts_are_rejected(self):
        predictor = self.make_predictor()
        with self.assertRaises(ValueError):
            predictor.features_from_row({"line_disconnected": "line_bad"})
        with self.assertRaises(ValueError):
            FailureForecastPredictor(RecordingClassifier(), {"features": ["old"]})


class ForecasterTrainingTests(unittest.TestCase):
    @staticmethod
    def arrays():
        rng = np.random.default_rng(7)
        X = rng.normal(size=(60, 4)).astype(np.float32)
        y = rng.normal(size=(60, 2)).astype(np.float32)
        episode_ids = np.repeat(np.arange(6), 10)
        return X, y, episode_ids

    def test_checkpoint_survives_failed_final_fit_and_skips_retuning(self):
        X, y, episode_ids = self.arrays()
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            params_path = directory / "mean_forecaster_best_params.json"
            model_path = directory / "mean_forecaster.pkl"

            def fail_only_full_fit(*args, **kwargs):
                if len(args[1]) == len(X):
                    raise RuntimeError("simulated full-fit failure")
                return _fit_multioutput_with_progress(*args, **kwargs)

            with mock.patch(
                "src.failure_forecast._fit_multioutput_with_progress",
                side_effect=fail_only_full_fit,
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    train_mean_forecaster(
                        X,
                        y,
                        model_path=model_path,
                        n_trials=1,
                        max_subsample=20,
                        seed=3,
                        episode_ids=episode_ids,
                        native_threads=1,
                        best_params_path=params_path,
                    )
            self.assertTrue(params_path.is_file())
            self.assertFalse(model_path.exists())

            with mock.patch(
                "optuna.create_study",
                side_effect=AssertionError("Optuna must not run on resume"),
            ):
                model = train_mean_forecaster(
                    X,
                    y,
                    model_path=model_path,
                    n_trials=1,
                    max_subsample=20,
                    seed=3,
                    episode_ids=episode_ids,
                    native_threads=1,
                    best_params_path=params_path,
                )
            self.assertEqual(len(model.estimators_), y.shape[1])
            self.assertTrue(model_path.is_file())

    def test_progress_fit_limits_openmp_and_blas_threads(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

        X, y, _ = self.arrays()
        limiter = mock.MagicMock()
        limiter.return_value.__enter__.return_value = None
        with mock.patch("threadpoolctl.threadpool_limits", limiter):
            model = _fit_multioutput_with_progress(
                HistGradientBoostingRegressor(max_iter=2),
                X,
                y,
                native_threads=4,
                description="test",
            )
        self.assertEqual(len(model.estimators_), 2)
        self.assertIn(
            mock.call(limits=4, user_api="openmp"), limiter.call_args_list
        )
        self.assertIn(mock.call(limits=1, user_api="blas"), limiter.call_args_list)


class ClassifierTests(unittest.TestCase):
    @staticmethod
    def synthetic_rows(episodes=20, per_episode=4):
        rng = np.random.default_rng(0)
        rows = []
        for episode in range(episodes):
            for _ in range(per_episode):
                failed = int(episode % 2 == 0)
                row = {name: float(rng.normal()) + failed for name in CLASSIFIER_FEATURES
                       if name not in ("line_id_encoded", "load_gen_ratio")}
                row.update({
                    "episode": episode, "line_disconnected": "line_bad",
                    "failed": failed, "label_reason": LABEL_DISCONNECTION_TEST,
                })
                rows.append(row)
        return pd.DataFrame(rows)

    def test_f2_uses_sampled_probability_threshold(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.2, 0.6, 0.4, 0.8])
        low_threshold = _classifier_f2(labels, probabilities, 0.3)
        high_threshold = _classifier_f2(labels, probabilities, 0.7)
        self.assertNotEqual(low_threshold, high_threshold)
        self.assertAlmostEqual(low_threshold, 10.0 / 11.0)
        self.assertAlmostEqual(high_threshold, 5.0 / 9.0)

    def test_exact_threshold_selection_prefers_lower_threshold_on_f2_tie(self):
        labels = np.array([1, 0])
        probabilities = np.array([0.1, 0.2])
        threshold, f2 = _f2_optimal_threshold(labels, probabilities)
        self.assertEqual(threshold, 0.0)
        self.assertAlmostEqual(f2, 5.0 / 6.0)

    def test_metrics_report_rare_event_counts_and_scores(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        predictions = np.array([0, 1, 0, 1, 0, 1])
        probabilities = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.7])
        metrics = _metrics(labels, predictions, probabilities)
        self.assertEqual(metrics["false_alarms"], 1)
        self.assertEqual(metrics["missed_failures"], 1)
        self.assertAlmostEqual(metrics["precision"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["recall"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["f2"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["pr_auc"], 29.0 / 36.0)

    def test_train_with_optuna_save_and_reload_round_trip(self):
        df = prepare_classifier_dataframe(self.synthetic_rows())
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "failure_classifier.pkl"
            _, info = train_failure_classifier(
                df, model_path=path, seed=1, n_trials=2, n_jobs=1)
            self.assertEqual(info["features"], CLASSIFIER_FEATURES)
            self.assertEqual(info["label_counts"], {"1": 40, "0": 40})
            self.assertEqual(info["forecast_horizon_steps"], 12)
            self.assertEqual(info["optuna"]["n_trials"], 2)
            self.assertEqual(info["optuna"]["direction"], "maximize")
            self.assertEqual(
                info["decision_threshold"], info["optuna"]["best_threshold"]
            )
            self.assertGreaterEqual(info["optuna"]["best_f2"], 0.0)
            self.assertIn("class_weight", info["model_params"])
            self.assertEqual(info["split"]["strategy"], "episode_grouped")
            predictor = FailureForecastPredictor.load(path)
            self.assertEqual(predictor.threshold, info["decision_threshold"])
            result = predictor.predict_from_features(self.synthetic_rows().iloc[0].to_dict())
            self.assertIn(result["failure_prediction"], (0, 1))
            self.assertTrue(0.0 <= result["failure_probability"] <= 1.0)

    def test_zero_trials_uses_fixed_parameters(self):
        df = prepare_classifier_dataframe(self.synthetic_rows())
        with TemporaryDirectory() as tmp:
            _, info = train_failure_classifier(
                df, model_path=Path(tmp) / "c.pkl", seed=1, n_trials=0)
            self.assertEqual(info["optuna"]["direction"], "maximize")
            self.assertEqual(
                info["decision_threshold"], info["optuna"]["best_threshold"]
            )
            self.assertTrue(0.0 <= info["decision_threshold"] <= 1.0)
            self.assertIn("baseline_threshold_0_5_metrics", info)

    def test_split_is_70_15_15_and_keeps_episodes_apart(self):
        df = prepare_classifier_dataframe(self.synthetic_rows())
        X_train, X_val, X_test, _, _, _, groups, info = _split_classifier_rows(df, seed=3)
        sets = [set(df.loc[X.index, "episode"]) for X in (X_train, X_val, X_test)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set(groups), sets[0])
        self.assertEqual(info["strategy"], "episode_grouped")
        self.assertEqual((info["train_episodes"], info["val_episodes"], info["test_episodes"]),
                         (14, 3, 3))

    def test_missing_columns_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare_classifier_dataframe(
                self.synthetic_rows().drop(columns=["fcast_max_line_rho"]))

    def test_tuning_split_is_episode_disjoint(self):
        episode_ids = np.repeat(np.arange(10), 5)
        train_idx, val_idx = _fixed_tuning_indices(
            len(episode_ids), episode_ids, max_subsample=100, seed=0)
        self.assertFalse(set(episode_ids[train_idx]) & set(episode_ids[val_idx]))


if __name__ == "__main__":
    unittest.main()
