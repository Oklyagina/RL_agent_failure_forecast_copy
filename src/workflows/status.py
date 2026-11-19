"""Report which artifacts exist and what to run next (``python main.py status``).

Read-only: it prints the same stage table as ``train`` (without running or
marking anything) and checks whether ``demo`` has the files it needs.
"""
from __future__ import annotations

from project_config import AGENT_FACTORY, AGENT_NAME
from src.workflows import train

_IDLE_ACTIONS = ("reuse", "keep", "skip")


def _missing_demo_files(agent_name: str) -> list[str]:
    """Files ``demo`` loads; empty when the demo can run."""
    failures = train.failure_dir(agent_name)
    required = [
        train.model_dir(agent_name) / name
        for name in ("enn_meta.json", "scaler_params.json", "enn_pctile_calib.npz",
                     f"enn_{agent_name}.pth")
    ] + [
        failures / name
        for name in ("mean_forecaster.pkl", "aleatoric_forecaster.pkl",
                     "failure_classifier.pkl", "failure_classifier_metadata.json")
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if not AGENT_FACTORY and not train.pipeline_support.has_complete_policy(
        train.default_policy_dir()
    ):
        missing.append(f"policy assets in {train.default_policy_dir()}")
    return missing


def run(agent_name: str = AGENT_NAME) -> None:
    plan = train.plan_stages(agent_name)
    train.print_preflight(plan.preflight_rows(), agent_name)

    pending = [stage for stage, _, action in plan.preflight_rows()
               if not action.startswith(_IDLE_ACTIONS)]
    missing_demo = _missing_demo_files(agent_name)
    n_rules = train.count_rules(plan.rules_dir)
    print()
    if plan.blocked:
        print("  'python main.py train' would stop before running anything:")
        for reason in plan.blocked:
            print(f"    - {reason}")
    elif pending:
        print(f"  Training needed for: {', '.join(pending)}")
        print("  Next: python main.py train")
    else:
        print("  All training stages are up to date.")
    if n_rules:
        print(f"  Explanation rules: {n_rules} lines in {plan.rules_dir}")
    else:
        print(f"  Explanation rules: none in {plan.rules_dir} "
              "(demo runs without explanations; generate with "
              "python main.py train --profile)")
    if missing_demo:
        print("  Demo not ready; missing:")
        for item in missing_demo:
            print(f"    - {item}")
    else:
        suffix = " (with rule explanations)" if n_rules else ""
        print(f"  Demo ready{suffix}: python main.py demo")
