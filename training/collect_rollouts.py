"""
training/collect_rollouts.py -- step 1 of training the ENN for a NEW agent
(e.g. the ExpertAgent). Agent-agnostic: it only needs an object exposing
`agent.act(obs, reward, done) -> grid2op action`.

It rolls the trained agent in the environment and collects (observation,
action) pairs; then it CURATES the action set by deduplicating the action
vectors seen during the rollouts, and assigns each step the index of its
action in that curated set. This produces exactly the three arrays the ENN
training step needs:

    out_dir/observations.npy     float32 [N, obs_dim]   obs.to_vect()
    out_dir/labels.npy           int64   [N]            index in curated set
    out_dir/actions.npy          float32 [K, act_dim]   curated action set
                                                        (action.to_vect())

Usage:
    python training/collect_rollouts.py

Then point training/train_enn.py at out_dir.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import grid2op
from lightsim2grid import LightSimBackend

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_config import (AGENT_NAME, ARTIFACTS_DIR, ASSETS_DIR, ENV_DIR,
                            ENV_NAME, ROLLOUT_EPISODES, SEED)


def default_rollout_dir(agent_name: str) -> Path:
    return ARTIFACTS_DIR / ENV_NAME / agent_name / "rollouts"


def _has_valid_saved_model(agent_dir: Path) -> bool:
    model_dir = agent_dir / "model"
    variables_dir = model_dir / "variables"
    required_files = [
        model_dir / "saved_model.pb",
        variables_dir / "variables.index",
        variables_dir / "variables.data-00000-of-00001",
    ]
    return all(p.is_file() and p.stat().st_size > 0 for p in required_files)


def _candidate_agent_dirs() -> list[Path]:
    preferred = [
        ASSETS_DIR / ENV_NAME,
        ASSETS_DIR / "network36",
        ROOT / "src" / "models" / "network36",
    ]
    base = ROOT / "curriculumagent"
    discovered = [base, *sorted(x for x in base.rglob("*") if x.is_dir())]
    return preferred + discovered


def make_curriculum_agent(env):
    """Load the CurriculumAgent from the binaries committed in
    curriculumagent/ (official entrypoint; the folder must contain model/
    and actions/ subfolders -- auto-located)."""
    from curriculumagent.submission.my_agent import make_agent
    invalid = []
    for d in _candidate_agent_dirs():
        if (d / "model").is_dir() and (d / "actions").is_dir():
            if not _has_valid_saved_model(d):
                invalid.append(d)
                continue
            print(f"agent dir: {d.relative_to(ROOT)}")
            return make_agent(env, str(d))
    hint = ""
    if invalid:
        bad = ", ".join(str(d.relative_to(ROOT)) for d in invalid)
        hint = f" Skipped invalid SavedModel artifact(s): {bad}."
    raise FileNotFoundError(
        "no valid folder with model/ and actions/ found. Expected a "
        f"non-empty TensorFlow SavedModel under assets/{ENV_NAME}/, "
        "assets/network36/ or "
        f"src/models/network36/.{hint}")


def make_expert_agent(env):
    """Build the ExpertAgent. Replace with its real constructor, e.g.:
        from expert_agent import ExpertAgent
        return ExpertAgent(env.action_space, ...)
    Any object with .act(obs, reward, done) -> grid2op action works."""
    raise NotImplementedError("plug the ExpertAgent constructor here")


AGENTS = {"curriculum": make_curriculum_agent, "expert": make_expert_agent}


def collect(agent_name: str, episodes: int, out_dir: Path, seed: int = 0,
            max_steps: int | None = None) -> None:

    print()

    env = grid2op.make(str(ENV_DIR), backend=LightSimBackend())
    env.seed(seed)
    agent = AGENTS[agent_name](env)

    obs_rows, act_rows = [], []
    for ep in range(episodes):
        obs = env.reset()
        reward, done, t = env.reward_range[0], False, 0
        while not done and (max_steps is None or t < max_steps):
            action = agent.act(obs, reward=reward, done=done)
            obs_rows.append(obs.to_vect().astype(np.float32))
            act_rows.append(action.to_vect().astype(np.float32))
            obs, reward, done, _ = env.step(action)
            t += 1
        print(f"episode {ep + 1}/{episodes}: {t} steps "
              f"(total pairs: {len(obs_rows)})")

    X = np.stack(obs_rows)                       # [N, obs_dim]
    A = np.stack(act_rows)                       # [N, act_dim]

    # Curate the action set: unique action vectors, stable order of first
    # appearance; labels[i] = index of A[i] in the curated set.
    curated, labels = np.unique(A, axis=0, return_inverse=True)
    print(f"\ncollected {len(X)} pairs | curated action set: "
          f"{curated.shape[0]} distinct actions")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "observations.npy", X)
    np.save(out_dir / "labels.npy", labels.astype(np.int64))
    np.save(out_dir / "actions.npy", curated)
    print(f"[ok] wrote observations.npy, labels.npy, actions.npy in {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=("curriculum", "expert"),
                    default=AGENT_NAME)
    ap.add_argument("--episodes", type=int, default=ROLLOUT_EPISODES)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help=f"default: artifacts/{ENV_NAME}/<agent>/rollouts")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = default_rollout_dir(args.agent)
    collect(args.agent, args.episodes, args.out_dir, args.seed, args.max_steps)
