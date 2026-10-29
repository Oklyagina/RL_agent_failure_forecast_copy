"""Train the t+12 failure classifier and select its threshold by F2."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_config import (  # noqa: E402
    AGENT_NAME,
    ARTIFACTS_DIR,
    ENV_NAME,
    FAILURE_CLASSIFIER_TRIALS,
    SEED,
)
from src.failure_forecast import (  # noqa: E402
    prepare_classifier_dataframe,
    train_failure_classifier,
)


def default_dir(agent_name: str) -> Path:
    return ARTIFACTS_DIR / ENV_NAME / agent_name / "failure_forecast"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-name", default=AGENT_NAME)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--trials", type=int, default=FAILURE_CLASSIFIER_TRIALS,
                        help="Optuna trials (0 = fixed parameters, no search)")
    args = parser.parse_args()

    out_dir = default_dir(args.agent_name)
    input_csv = args.input_csv or (out_dir / "failure_forecast_rows.csv")
    model_path = args.model_path or (out_dir / "failure_classifier.pkl")
    df = prepare_classifier_dataframe(input_csv)
    _, info = train_failure_classifier(
        df,
        model_path=model_path,
        seed=args.seed,
        env_name=ENV_NAME,
        agent_name=args.agent_name,
        n_trials=args.trials,
    )
    metrics = info["metrics"]
    print(f"[ok] wrote classifier -> {model_path}")
    print(f"[ok] wrote metadata -> {info['metadata_path']}")
    print(f"[labels] rows by reason: {info['label_reason_counts']}")
    print(f"[labels] rows by class: {info['label_counts']}")
    print(
        f"[test] threshold={info['decision_threshold']:.6f} "
        f"F2={metrics['f2']:.3f} precision={metrics['precision']:.3f} "
        f"recall={metrics['recall']:.3f} PR-AUC={metrics['pr_auc']:.3f} "
        f"FA={metrics['false_alarms']} ({metrics['false_alarm_pct']:.2f}%) "
        f"missed={metrics['missed_failures']} "
        f"({metrics['missed_failure_pct']:.2f}%)"
    )
    baseline = info["baseline_threshold_0_5_metrics"]
    print(
        f"[baseline@0.5] F2={baseline['f2']:.3f} "
        f"precision={baseline['precision']:.3f} recall={baseline['recall']:.3f} "
        f"FA={baseline['false_alarms']} missed={baseline['missed_failures']}"
    )


if __name__ == "__main__":
    main()
