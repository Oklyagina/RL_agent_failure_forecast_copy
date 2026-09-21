"""Fast API contract test using synthetic observations and model inputs."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.main as api_module  # noqa: E402
from recommendation_uncertainty import Calibration, build_calibration  # noqa: E402
from src.enn_models import EvidentialNetwork  # noqa: E402


OBSERVATION_DIM = 40
ACTION_COUNT = 20
ACTION_DIM = 25


class FakeAction:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=float)

    def to_vect(self):
        return self.vector

    def to_json(self):
        return {"_set_topo_vect": self.vector.tolist()}

    def as_serializable_dict(self):
        return self.to_json()

    def impact_on_objects(self):
        return {
            "force_line": {
                "changed": False,
                "reconnections": {"count": 0, "powerlines": []},
                "disconnections": {"count": 0, "powerlines": []},
            },
            "switch_line": {"changed": False, "count": 0, "powerlines": []},
            "topology": {
                "bus_switch": [],
                "assigned_bus": [
                    {
                        "bus": 1,
                        "object_type": "line",
                        "object_id": 11,
                        "substation": 3,
                    }
                ],
                "disconnect_bus": [],
            },
        }

    def __str__(self):
        return "Assign bus 1 to line id 11 (example)"


class FakeSimulatedObservation:
    rho = np.asarray([0.23, 0.87, 0.42], dtype=np.float32)


class FakeObservation:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def to_vect(self):
        return self.vector

    def from_vect(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)
        return self

    def simulate(self, action, time_step=1):
        del action, time_step
        return FakeSimulatedObservation(), 0.0, False, {}


class FakeEnvironment:
    def __init__(self, observation):
        self.observation = observation

    def reset(self):
        return self.observation


class FakeAgent:
    def __init__(self, action):
        self.action = action
        self.calls = 0

    def act(self, observation, reward=None, done=False):
        del observation, reward, done
        self.calls += 1
        return self.action


class RecommendationApiTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(11)
        torch.manual_seed(11)
        actions = rng.randn(ACTION_COUNT, ACTION_DIM)
        observation = FakeObservation(rng.randn(OBSERVATION_DIM))
        enn = EvidentialNetwork(
            OBSERVATION_DIM,
            ACTION_COUNT,
            hidden_dim=32,
            dropout=0.0,
        ).eval()
        scaler = StandardScaler().fit(rng.randn(80, OBSERVATION_DIM))
        calibration_rows = scaler.transform(
            rng.randn(40, OBSERVATION_DIM)
        ).astype(np.float32)
        total_ref, action_ref = build_calibration(enn, calibration_rows)
        calibration = Calibration(
            total_ref,
            action_ref,
            scaler=scaler,
            action_set=actions,
            class_mapping={str(index): index for index in range(ACTION_COUNT)},
        )
        self.agent = FakeAgent(FakeAction(actions[7]))
        self.context = {"observation": observation.to_vect().tolist()}
        self.original_get_services = api_module.get_services
        self.original_get_services.cache_clear()
        api_module.get_services = lambda: (
            FakeEnvironment(observation),
            self.agent,
            enn,
            calibration,
        )

    def tearDown(self):
        api_module.get_services = self.original_get_services
        self.original_get_services.cache_clear()

    def test_recommendation_endpoint_contract(self):
        response = TestClient(api_module.app).post(
            "/api/v1/recommendation",
            json={"event": {"id": "evt-1"}, "context": self.context},
        )

        self.assertEqual(response.status_code, 200, response.text)
        recommendations = response.json()
        self.assertEqual(len(recommendations), 1)
        recommendation = recommendations[0]
        expected_fields = {
            "title", "description", "use_case", "agent_type", "actions", "kpis"
        }
        self.assertTrue(expected_fields.issubset(recommendation))
        self.assertTrue(
            {"data", "criticality", "start_date"}.isdisjoint(recommendation)
        )
        self.assertEqual(recommendation["use_case"], "PowerGrid")
        self.assertEqual(recommendation["agent_type"], 2)
        self.assertTrue(recommendation["actions"])
        self.assertEqual(self.agent.calls, 1)

        kpis = recommendation["kpis"]
        self.assertEqual(kpis["efficiency_of_the_reco"], np.float32(0.87).item())
        self.assertEqual(kpis["uncertainty"], kpis["epistemic_uncertainty_pct"])
        self.assertGreaterEqual(kpis["uncertainty"], 0.0)
        self.assertLessEqual(kpis["uncertainty"], 100.0)
        self.assertGreaterEqual(kpis["epistemic_uncertainty_total_pctile"], 0.0)
        self.assertLessEqual(kpis["epistemic_uncertainty_total_pctile"], 100.0)
        self.assertIsNotNone(kpis["epistemic_uncertainty_action_pctile"])
        self.assertIn(kpis["epistemic_uncertainty_level"], {"low", "medium", "high"})
        self.assertIn(kpis["epistemic_confidence_level"], {"low", "medium", "high"})


if __name__ == "__main__":
    unittest.main()
