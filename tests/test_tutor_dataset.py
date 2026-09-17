"""Validation checks for CurriculumAgent tutor dataset splitting."""

import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from curriculumagent.tutor.collect_tutor_experience import create_dataset


def _rows(n_rows: int, n_features: int = 4) -> np.ndarray:
    actions = np.arange(n_rows, dtype=np.float32).reshape(-1, 1)
    states = np.arange(n_rows * n_features, dtype=np.float32).reshape(n_rows, n_features)
    return np.hstack([actions, states])


def _split_shapes(root: Path) -> dict[str, tuple[int, ...]]:
    paths = {
        "train": root / "test_train.npz",
        "val": root / "test_val.npz",
        "test": root / "test_test.npz",
    }
    with np.load(paths["train"]) as train, np.load(paths["val"]) as val, np.load(paths["test"]) as test:
        return {
            "s_train": train["s_train"].shape,
            "a_train": train["a_train"].shape,
            "s_validate": val["s_validate"].shape,
            "a_validate": val["a_validate"].shape,
            "s_test": test["s_test"].shape,
            "a_test": test["a_test"].shape,
        }


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        try:
            create_dataset(root, _rows(5), dataset_name="test", min_unique_rows=10)
        except ValueError as exc:
            assert "only 5 unique rows" in str(exc)
        else:
            raise AssertionError("small tutor datasets must fail clearly")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        create_dataset(root, _rows(10), dataset_name="test", min_unique_rows=10)
        shapes = _split_shapes(root)
        assert shapes == {
            "s_train": (8, 4),
            "a_train": (8, 1),
            "s_validate": (1, 4),
            "a_validate": (1, 1),
            "s_test": (1, 4),
            "a_test": (1, 1),
        }

    print("test_tutor_dataset: PASSED")


if __name__ == "__main__":
    main()
