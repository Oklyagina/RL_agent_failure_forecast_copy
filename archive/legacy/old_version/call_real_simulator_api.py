"""
call_real_simulator_api.py

Create a real Grid2Op simulator observation and send it to the local
CurriculumAgent + ENN uncertainty FastAPI endpoint.

Description
-----------
This script is a client-side smoke test for the API exposed by app/main.py.
It starts a Grid2Op environment using the same project_config.py settings as
run_pipeline.py and the API server, advances the simulator with do-nothing
actions if requested, serializes the current observation as obs.to_vect(), and
posts it to:

    POST /api/v1/recommendation

The API response should be a list of InteractiveAI-style recommendations with
actions and KPIs, including epistemic uncertainty percentiles.

Assumptions
-----------
- The API server is already running, for example:
      uvicorn app.main:app --host 0.0.0.0 --port 8000
- The local .env points to the same Grid2Op environment and artifacts used by
  the trained models.
- ENV_LOCATION / ENV_NAME resolves to a valid Grid2Op scenario directory.
- The trained artifacts exist under artifacts/<ENV_NAME>/<AGENT_NAME>/.
- Sending obs.to_vect().tolist() is acceptable because app/main.py explicitly
  supports reconstructing observations from plain vectors.
- This script is intended as an integration check, not as a replacement for the
  real InteractiveAI simulator payload.
"""

import argparse
import json

import grid2op
import requests
from lightsim2grid import LightSimBackend

from project_config import ENV_DIR, ENV_NAME, SEED


def build_real_observation(seed: int, step: int):
    env = grid2op.make(str(ENV_DIR), backend=LightSimBackend())
    env.seed(seed)

    obs = env.reset()
    reward = env.reward_range[0]
    done = False

    for _ in range(step):
        action = env.action_space({})
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
            reward = env.reward_range[0]
            done = False

    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    obs = build_real_observation(args.seed, args.step)

    payload = {
        "event": {
            "source": "local_grid2op_simulator",
            "env": ENV_NAME,
            "seed": args.seed,
            "step": args.step,
        },
        "context": {
            "observation": obs.to_vect().astype(float).tolist(),
        },
    }

    response = requests.post(
        f"{args.url.rstrip('/')}/api/v1/recommendation",
        json=payload,
        timeout=args.timeout,
    )
    if not response.ok:
        print(f"Request failed with HTTP {response.status_code}.")
        print(response.text)
    response.raise_for_status()

    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
