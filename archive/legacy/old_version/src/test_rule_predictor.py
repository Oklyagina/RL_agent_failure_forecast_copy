import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .config import CFG, OUTPUT_DIR_LLM, ROOT
except ImportError:
    from config import CFG, OUTPUT_DIR_LLM, ROOT

import grid2op
from curriculumagent.baseline.baseline import CurriculumAgent
from lightsim2grid import LightSimBackend

try:
    from .rule_predictor import RulePredictor
except ImportError:
    from rule_predictor import RulePredictor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EPISODE_SEED: int = getattr(CFG, "LLM_RULES_EPISODE", getattr(CFG, "PROBA_TEST_EPISODE_ID", 50))
_rules_dir = Path(getattr(CFG, "LLM_RULES_DIR", Path(OUTPUT_DIR_LLM) / "temp_0.8"))
if not _rules_dir.is_absolute():
    _rules_dir = ROOT / _rules_dir
RESULTS_DIR: str = str(_rules_dir)
HORIZON: int = 12   # steps = 1 hour at 5 min resolution

from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import RandomLineOpponent

OPP_KWARGS: Dict[str, Any] = {
    "opponent_attack_duration": 288,
    "opponent_attack_cooldown": 288,
    "opponent_budget_per_ts":   0.5,
    "opponent_init_budget":     10_000.0,
    "opponent_action_class":    PowerlineSetAction,
    "opponent_class":           RandomLineOpponent,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_models() -> Tuple[Any, Any, Any]:
    import joblib
    import torch
    try:
        from .enn_models import EvidentialNetwork
    except ImportError:
        from enn_models import EvidentialNetwork

    mp = joblib.load(CFG.MODEL_MEAN_PATH)
    ma = joblib.load(CFG.MODEL_ALEATORIC_PATH)

    checkpoint = torch.load(CFG.MODEL_ENN_PATH, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    hidden_dim = checkpoint["embedding.0.weight"].shape[0]
    input_dim = checkpoint["embedding.0.weight"].shape[1]
    num_classes = checkpoint["head.3.bias"].shape[0]
    me = EvidentialNetwork(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=CFG.ENN_DROPOUT,
    )
    me.load_state_dict(checkpoint)
    me.eval()
    return mp, ma, me


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _line_name_to_idx(env: Any, line_name: str) -> Optional[int]:
    clean = line_name.replace("line_", "")
    for i, name in enumerate(env.name_line):
        if str(name) == clean or str(name).replace(" ", "_") == clean:
            return i
    return None


def _simulate_disconnection(obs: Any, line_idx: int, env: Any) -> bool:
    """
    Simulates disconnecting *line_idx* and checks if the grid fails.
    Returns True if the contingency causes a game over.
    """
    try:
        act = env.action_space({"set_line_status": [(line_idx, -1)]})
        _, _, done, info = obs.simulate(act)
        return done or info.get("is_illegal", False)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(use_attack: bool) -> None:
    label = "HeavyAttack_1" if use_attack else "Normal"
    print(f"\n{'═'*65}")
    print(f"  Scenario: {label}  |  seed={EPISODE_SEED}")
    print(f"{'═'*65}\n")

    # Build environment
    env_kwargs: Dict[str, Any] = {"backend": LightSimBackend()}
    if use_attack:
        env_kwargs.update(OPP_KWARGS)
    env = grid2op.make(CFG.ENV_NAME, **env_kwargs)

    # Load models and agent
    mp, ma, me = _load_models()
    agent = CurriculumAgent(env.action_space, env.observation_space, name="CA")
    try:
        agent.load(CFG.AGENT_PATH)
    except Exception:
        print("[WARN] CurriculumAgent not loaded — using do-nothing fallback.")

    # Import project functions explicitly so RulePredictor has them
    try:
        from .utils import compute_grid_stats
        from .training_enn import get_uncertainty
        from .collect_data import get_features_with_history
    except ImportError:
        from utils import compute_grid_stats
        from training_enn import get_uncertainty
        from collect_data import get_features_with_history

    # Build predictor
    observations_array: List[Any] = []
    predictor = RulePredictor(
        rules_dir=RESULTS_DIR,
        model_predict=mp,
        model_aleatoric=ma,
        model_enn=me,
        observations_array=observations_array,
        compute_grid_stats_fn=compute_grid_stats,
        get_uncertainty_fn=get_uncertainty,
        get_features_with_history_fn=get_features_with_history,
    )
    lines = predictor.available_lines
    if not lines:
        print(f"[ERROR] No rules found in '{RESULTS_DIR}'.")
        return

    # Resolve line names to environment indices
    line_indices: Dict[str, int] = {}
    for ln in lines:
        idx = _line_name_to_idx(env, ln)
        if idx is not None:
            line_indices[ln] = idx
    lines = list(line_indices.keys())

    print(f"[INFO] Lines monitored: {lines}\n")

    # ── Episode loop ─────────────────────────────────────────────────────
    obs               = env.reset(seed=EPISODE_SEED)
    done              = False
    step              = 0
    gameover_step: Optional[int] = None

    confirmed_alerts: Dict[str, List[int]] = {ln: [] for ln in lines}

    print(f"[INFO] Episode started. Running...\n")

    while not done:
        step += 1
        observations_array.append(obs)

        if step % 100 == 0:
            print(f"  [step {step:5d}] running... (no alerts so far)" if not any(confirmed_alerts.values())
                  else f"  [step {step:5d}] running...")

        for line_name in lines:
            result    = predictor.predict(obs, line_name)
            predicted = result["prediction"]

            if predicted == 1:
                # Verify with actual contingency simulation
                idx         = line_indices[line_name]
                actual_fail = _simulate_disconnection(obs, idx, env)

                if actual_fail:
                    confirmed_alerts[line_name].append(step)
                    print(f"  Step {step:4d} | [{line_name}] FAILURE PREDICTED")
                    print(f"  {result['sentence']}\n")

        # Step the environment with the agent
        try:
            action = agent.act(obs, 0.0, done)
        except Exception:
            action = env.action_space({})

        obs, _, done, _ = env.step(action)

        if done:
            gameover_step = step

    # ── End-of-episode summary ───────────────────────────────────────────
    print(f"\n{'─'*65}")

    if gameover_step is None:
        print(f"  Agent completed the episode without failure ({step} steps).\n")
        return

    print(f"  Agent failed at step {gameover_step}.\n")
    print(f"  Did the rule warn in the {HORIZON} steps before failure?\n")
    print(f"  {'Line':<20}  {'Warned?':<8}  Warning steps")
    print(f"  {'─'*20}  {'─'*8}  {'─'*25}")
    for ln in lines:
        warning_steps = [
            s for s in confirmed_alerts[ln]
            if gameover_step - HORIZON <= s < gameover_step
        ]
        warned = len(warning_steps) > 0
        tick   = "YES " if warned else "NO  "
        print(f"  {ln:<20}  {tick:<8}  {warning_steps if warning_steps else '—'}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack", action="store_true",
        help="Run with HeavyAttack_1 opponent.",
    )
    args = parser.parse_args()
    run_episode(use_attack=args.attack)
