"""
tests/test_api.py -- validates the agent API (app/main.py) end-to-end with a
synthetic ENN and fake Grid2Op objects, no trained weights or real environment
needed. Confirms the InteractiveAI output shape and that the two uncertainty
percentiles land inside "kpis".

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
    def as_serializable_dict(self): return {"_set_topo_vect": self.v.tolist()}
    def __str__(self): return "Assign bus 1 to line id 11 (example)"


class FakeObs:
    def __init__(self, v): self.v = np.asarray(v, "float32")
    def to_vect(self): return self.v
    def from_vect(self, v):
        self.v = np.asarray(v, "float32")
        return self


class FakeEnv:
    def reset(self): return FakeObs(rng.randn(OBS))


class FakeAgent:
    def act(self, o, reward=None, done=False): return FakeAction(ACTIONS[7])


ACTIONS = rng.randn(K, ACT)


def main():
    enn = EvidentialNetwork(OBS, K).eval()
    scaler = StandardScaler().fit(rng.randn(300, OBS))
    tot, act = build_calibration(
        enn, scaler.transform(rng.randn(200, OBS)).astype("float32"))
    calib = Calibration(tot, act, scaler=scaler, action_set=ACTIONS,
                        class_mapping={str(k): k for k in range(K)})

    M.get_services.cache_clear()
    M.get_services = lambda: (FakeEnv(), FakeAgent(), enn, calib)

    context = {"observation": rng.randn(OBS).tolist()}
    recos = M.build_recommendations(context)
    r = recos[0]

    assert isinstance(recos, list) and len(recos) == 1
    for key in ("title", "description", "use_case", "agent_type",
                "actions", "kpis"):
        assert key in r, f"missing {key}"
    k = r["kpis"]
    assert "efficiency_of_the_reco" in k
    assert "epistemic_uncertainty_total_pctile" in k
    assert "epistemic_uncertainty_action_pctile" in k
    assert 0.0 <= k["epistemic_uncertainty_total_pctile"] <= 100.0
    assert json.dumps(recos)

    from fastapi.testclient import TestClient
    client = TestClient(M.app)
    resp = client.post("/api/v1/recommendation", json={"context": context})
    assert resp.status_code == 200, resp.text
    assert resp.json()[0]["kpis"]["epistemic_uncertainty_total_pctile"] \
        is not None

    print("test_api: PASSED")


if __name__ == "__main__":
    main()
