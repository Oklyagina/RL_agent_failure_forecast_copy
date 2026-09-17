"""Synthetic checks for the standalone ENN tuning pipeline.

Run with:
    python tests/test_tune_enn.py
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TUNE_ENN_PATH = ROOT / "training" / "tune_enn.py"
if not TUNE_ENN_PATH.is_file():
    raise FileNotFoundError(
        f"Expected standalone tuner at {TUNE_ENN_PATH}. "
        "Copy training/tune_enn.py to the server before running this test."
    )

spec = importlib.util.spec_from_file_location("tune_enn", TUNE_ENN_PATH)
tune_enn = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = tune_enn
spec.loader.exec_module(tune_enn)


def make_rollout_bundle(path: Path) -> None:
    rng = np.random.RandomState(7)
    observations = rng.randn(72, 10).astype(np.float32)
    labels = np.tile(np.arange(4, dtype=np.int64), 18)
    actions = rng.randn(4, 6).astype(np.float32)
    np.save(path / "observations.npy", observations)
    np.save(path / "labels.npy", labels)
    np.save(path / "actions.npy", actions)


def test_device_fallback():
    with patch.object(tune_enn.torch.cuda, "is_available", return_value=False):
        device = tune_enn.resolve_device("cuda")
    assert str(device) == "cpu"


def test_tuning_smoke_search_outputs():
    with TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        data_dir = tmp / "rollouts"
        out_dir = tmp / "tuning"
        data_dir.mkdir()
        make_rollout_bundle(data_dir)

        observations, labels, actions = tune_enn.load_rollout_arrays(data_dir)
        prepared = tune_enn.prepare_data(
            observations,
            labels,
            actions,
            val_frac=0.25,
            seed=3,
        )
        configs = [
            tune_enn.CandidateConfig(lr=1e-3, hidden_dim=8, dropout=0.0, batch_size=12),
            tune_enn.CandidateConfig(lr=3e-3, hidden_dim=12, dropout=0.05, batch_size=12),
        ]

        outcomes = [
            tune_enn.train_candidate(
                config,
                prepared,
                run_id=index,
                total_runs=len(configs),
                epochs=1,
                anneal_epochs=1,
                seed=11,
                device=torch.device("cpu"),
            )
            for index, config in enumerate(configs, start=1)
        ]
        results = [result for result, _state in outcomes]
        best_result, best_state = min(
            outcomes,
            key=lambda outcome: outcome[0].validation_loss,
        )
        csv_path, json_path, best_params_path, best_path = tune_enn.write_results(
            results,
            output_dir=out_dir,
            best_checkpoint_name="best_enn_test.pth",
            best_state=best_state,
        )

        assert csv_path.is_file()
        assert json_path.is_file()
        assert best_params_path.is_file()
        assert best_path.is_file()
        assert not (out_dir / "candidate_checkpoints").exists()

        payload = tune_enn.json.loads(json_path.read_text(encoding="utf-8"))
        expected_best = min(results, key=lambda result: result.validation_loss)
        assert expected_best.run_id == best_result.run_id
        assert payload["selection_metric"] == "validation_loss"
        assert payload["best_run_id"] == expected_best.run_id
        assert payload["best_validation_loss"] == expected_best.validation_loss
        assert payload["best_parameters"]["hidden_dim"] == expected_best.hidden_dim
        assert "test" not in payload["selection_metric"]

        rows = list(tune_enn.csv.DictReader(best_params_path.open(encoding="utf-8")))
        by_key = {row["env_key"]: row["value"] for row in rows}
        assert by_key["ENN_LR"] == f"{expected_best.lr:g}"
        assert by_key["ENN_HIDDEN_DIM"] == str(expected_best.hidden_dim)
        assert by_key["ENN_DROPOUT"] == f"{expected_best.dropout:g}"
        assert by_key["ENN_BATCH_SIZE"] == str(expected_best.batch_size)


def main():
    test_device_fallback()
    test_tuning_smoke_search_outputs()
    print("test_tune_enn: PASSED")


if __name__ == "__main__":
    main()
