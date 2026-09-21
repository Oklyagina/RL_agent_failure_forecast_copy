"""Fast tests for the active ENN uncertainty and recommendation pipeline."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommendation_uncertainty import (  # noqa: E402
    _percentile,
    assess_recommendation,
    build_calibration,
    confidence_level,
    load_calibration,
    save_calibration,
    uncertainty_level,
)
from src.enn_models import EvidentialNetwork  # noqa: E402


class FakeAction:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=float)

    def to_vect(self):
        return self.vector


class FakeObservation:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def to_vect(self):
        return self.vector


class FakeAgent:
    def __init__(self, action):
        self.action = action

    def act(self, observation, reward=0.0, done=False):
        del observation, reward, done
        return self.action


class EvidentialNetworkTests(unittest.TestCase):
    INPUT_DIM = 8
    NUM_CLASSES = 4
    ACTION_DIM = 5

    def setUp(self):
        self.rng = np.random.RandomState(7)
        torch.manual_seed(7)
        self.enn = EvidentialNetwork(
            self.INPUT_DIM,
            self.NUM_CLASSES,
            hidden_dim=16,
            dropout=0.0,
        ).eval()

    def test_inference_returns_valid_dirichlet_outputs(self):
        inputs = torch.as_tensor(
            self.rng.randn(3, self.INPUT_DIM), dtype=torch.float32
        )

        output = self.enn.predict(inputs)

        self.assertEqual(
            set(output), {"alpha", "evidence", "S", "prob", "uncertainty"}
        )
        self.assertEqual(tuple(output["prob"].shape), (3, self.NUM_CLASSES))
        self.assertTrue(torch.all(output["evidence"] >= 0))
        self.assertTrue(torch.all(output["alpha"] >= 1))
        torch.testing.assert_close(
            output["prob"].sum(dim=1), torch.ones(3), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            output["uncertainty"],
            self.NUM_CLASSES / output["S"],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_calibration_round_trip_and_recommendation_assessment(self):
        training_rows = self.rng.randn(64, self.INPUT_DIM)
        scaler = StandardScaler().fit(training_rows)
        scaled_rows = scaler.transform(training_rows).astype(np.float32)
        actions = self.rng.randn(self.NUM_CLASSES, self.ACTION_DIM)
        total_ref, action_ref = build_calibration(self.enn, scaled_rows)

        with TemporaryDirectory() as temporary:
            artifact_dir = Path(temporary)
            calibration_path = artifact_dir / "enn_pctile_calib.npz"
            actions_path = artifact_dir / "actions.npy"
            metadata_path = artifact_dir / "enn_meta.json"
            save_calibration(calibration_path, total_ref, action_ref)
            np.save(actions_path, actions)
            metadata_path.write_text(
                json.dumps(
                    {
                        "input_dim": self.INPUT_DIM,
                        "num_classes": self.NUM_CLASSES,
                        "class_mapping": {
                            str(index): index for index in range(self.NUM_CLASSES)
                        },
                    }
                ),
                encoding="utf-8",
            )

            calibration = load_calibration(
                calibration_path,
                scaler=scaler,
                action_set=actions_path,
                class_mapping=metadata_path,
            )

        observation = FakeObservation(self.rng.randn(self.INPUT_DIM))
        known = assess_recommendation(
            observation,
            FakeAgent(FakeAction(actions[2])),
            self.enn,
            calibration,
        )
        unknown = assess_recommendation(
            observation,
            FakeAgent(FakeAction(np.zeros(self.ACTION_DIM))),
            self.enn,
            calibration,
        )

        self.assertEqual(known["chosen_action_id"], 2)
        self.assertIsNotNone(known["epistemic_uncertainty_action_pctile"])
        self.assertIsNone(unknown["chosen_action_id"])
        self.assertIsNone(unknown["epistemic_uncertainty_action_pctile"])
        for result in (known, unknown):
            self.assertGreaterEqual(result["epistemic_uncertainty_pct"], 0.0)
            self.assertLessEqual(result["epistemic_uncertainty_pct"], 100.0)
            self.assertGreaterEqual(
                result["epistemic_uncertainty_total_pctile"], 0.0
            )
            self.assertLessEqual(
                result["epistemic_uncertainty_total_pctile"], 100.0
            )
            expected_confidence = {
                "low": "high",
                "medium": "medium",
                "high": "low",
            }[result["epistemic_uncertainty_level"]]
            self.assertEqual(
                result["epistemic_confidence_level"], expected_confidence
            )

    def test_percentile_and_kpi_bands_have_stable_boundaries(self):
        reference = np.asarray([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(_percentile(0.0, reference), 0.0)
        self.assertEqual(_percentile(4.0, reference), 100.0)
        self.assertEqual(_percentile(5.0, reference), 100.0)
        self.assertEqual(uncertainty_level(0.0), "low")
        self.assertEqual(uncertainty_level(50.0), "medium")
        self.assertEqual(uncertainty_level(100.0), "high")
        self.assertEqual(confidence_level(0.0), "high")
        self.assertEqual(confidence_level(100.0), "low")


if __name__ == "__main__":
    unittest.main()
