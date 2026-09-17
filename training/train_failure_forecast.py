"""Train the one-hour-ahead failure classifier from collected rows."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_config import AGENT_NAME, ARTIFACTS_DIR, ENV_NAME, SEED  # noqa: E402
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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()

    out_dir = default_dir(args.agent_name)
    input_csv = args.input_csv or (out_dir / "failure_forecast_rows.csv")
    model_path = args.model_path or (out_dir / "failure_classifier.pkl")
    df = prepare_classifier_dataframe(input_csv)
    _, info = train_failure_classifier(
        df,
        model_path=model_path,
        threshold=args.threshold,
        seed=args.seed,
        env_name=ENV_NAME,
        agent_name=args.agent_name,
        max_iter=args.max_iter,
    )
    metrics = info["metrics"]
    print(f"[ok] wrote classifier -> {model_path}")
    print(f"[ok] wrote metadata -> {info['metadata_path']}")
    print(
        f"[test] FA={metrics['false_alarm_pct']:.2f}% "
        f"oversight={metrics['oversight_pct']:.2f}% "
        f"AUC={metrics['roc_auc']:.3f}"
    )


if __name__ == "__main__":
    main()
