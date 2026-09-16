"""
tests/test_api.py -- validates the agent API (app/main.py) end-to-end with a
synthetic ENN and fake Grid2Op objects, no trained weights or real environment
needed. Confirms the recommendation output shape and all uncertainty KPI fields.

    python tests/test_api.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from enn_models_synthetic import EvidentialNetwork
from recommendation_uncertainty import build_calibration, Calibration
import app.main as M

rng = np.random.RandomState(0)
torch.manual_seed(0)
OBS, K, ACT = 40, 20, 25


class FakeAction:
    def __init__(self, v): self.v = np.asarray(v, float)
    def to_vect(self): return self.v
    def to_json(self): return {"_set_topo_vect": self.v.tolist()}
    def as_serializable_dict(self): return {"_set_topo_vect": self.v.tolist()}
    def impact_on_objects(self):
        return {
            "force_line": {
                "changed": False,
                "reconnections": {"count": 0, "powerlines": []},
                "disconnections": {"count": 0, "powerlines": []},
            },
            "switch_line": {
                "changed": False,
                "count": 0,
                "powerlines": [],
            },
            "topology": {
                "bus_switch": [],
                "assigned_bus": [{
                    "bus": 1,
                    "object_type": "line",
                    "object_id": 11,
                    "substation": 3,
                }],
                "disconnect_bus": [],
            },
        }
    def __str__(self): return "Assign bus 1 to line id 11 (example)"


class FakeSimulatedObs:
    def __init__(self):
        self.rho = np.asarray([0.23, 0.87, 0.42], dtype="float32")


class FakeObs:
    def __init__(self, v): self.v = np.asarray(v, "float32")
    def to_vect(self): return self.v
    def from_vect(self, v):
        self.v = np.asarray(v, "float32")
        return self
    def simulate(self, action, time_step=1):
        return FakeSimulatedObs(), 0.0, False, {}


class FakeEnv:
    def reset(self): return FakeObs(rng.randn(OBS))


class FakeAgent:
    def __init__(self):
        self.calls = 0
    def act(self, o, reward=None, done=False):
        self.calls += 1
        return FakeAction(ACTIONS[7])


ACTIONS = rng.randn(K, ACT)


def test_api_contract():
    enn = EvidentialNetwork(OBS, K).eval()
    scaler = StandardScaler().fit(rng.randn(300, OBS))
    tot, act = build_calibration(
        enn, scaler.transform(rng.randn(200, OBS)).astype("float32"))
    calib = Calibration(tot, act, scaler=scaler, action_set=ACTIONS,
                        class_mapping={str(k): k for k in range(K)})

    fake_agent = FakeAgent()
    M.get_services.cache_clear()
    M.get_services = lambda: (FakeEnv(), fake_agent, enn, calib)

    context = {"observation": rng.randn(OBS).tolist()}
    recos = M.build_recommendations(context)
    r = recos[0]

    assert isinstance(recos, list) and len(recos) == 1
    for key in ("title", "description", "use_case", "agent_type",
                "actions", "kpis"):
        assert key in r, f"missing {key}"
    assert "data" not in r
    assert r["use_case"] == "PowerGrid"
    k = r["kpis"]
    assert "efficiency_of_the_reco" in k
    assert k["efficiency_of_the_reco"] == np.float32(0.87).item()
    assert fake_agent.calls == 1
    assert "uncertainty" in k
    assert "epistemic_uncertainty_pct" in k
    assert "epistemic_uncertainty_total_pctile" in k
    assert "epistemic_uncertainty_action_pctile" in k
    assert k["epistemic_uncertainty_level"] in {"low", "medium", "high"}
    assert k["epistemic_confidence_level"] in {"low", "medium", "high"}
    assert k["uncertainty"] == k["epistemic_uncertainty_pct"]
    assert 0.0 <= k["uncertainty"] <= 100.0
    assert 0.0 <= k["epistemic_uncertainty_pct"] <= 100.0
    assert 0.0 <= k["epistemic_uncertainty_total_pctile"] <= 100.0
    assert json.dumps(recos)

    from fastapi.testclient import TestClient
    client = TestClient(M.app)
    resp = client.post(
        "/api/v1/recommendation",
        json={"event": {"id": "evt-1"}, "context": context},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()[0]
    for key in ("title", "description", "use_case", "agent_type",
                "actions", "kpis"):
        assert key in payload, f"missing response field {key}"
    assert "data" not in payload
    assert "criticality" not in payload
    assert "start_date" not in payload
    assert payload["use_case"] == "PowerGrid"
    assert payload["actions"]
    assert payload["agent_type"] == 2
    assert payload["kpis"]["efficiency_of_the_reco"] == np.float32(0.87).item()
    assert payload["kpis"]["uncertainty"] == \
        payload["kpis"]["epistemic_uncertainty_pct"]
    assert payload["kpis"]["epistemic_uncertainty_total_pctile"] is not None
    assert fake_agent.calls == 2

    print("test_api: PASSED")


def main():
    test_api_contract()


if __name__ == "__main__":
    main()
