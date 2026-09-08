"""Synthetic validation for ENN split loading and dynamic class sizing."""

import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from training_enn import _load_tutor_split, _remap_topk


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        split = root / "test_train.npz"
        states = np.arange(30, dtype=np.float32).reshape(5, 6)
        actions = np.array([[7], [7], [3], [9], [3]], dtype=np.int64)
        np.savez_compressed(split, s_train=states, a_train=actions)

        loaded_states, loaded_actions = _load_tutor_split(split)
        assert loaded_states.shape == (5, 6)
        assert loaded_actions.shape == (5,)

        result = _remap_topk(
            loaded_actions, loaded_actions.copy(), loaded_actions.copy(), top_k=120
        )
        assert result[-1] == 3, "num_classes must equal observed classes, not requested top_k"

        try:
            _load_tutor_split(root / "missing.npz")
        except FileNotFoundError as exc:
            assert "Required ENN tutor split" in str(exc)
        else:
            raise AssertionError("missing tutor data must fail explicitly")

        empty = root / "empty.npz"
        np.savez_compressed(
            empty,
            s_validate=np.empty((0, 6), dtype=np.float32),
            a_validate=np.empty((0, 1), dtype=np.int64),
        )
        try:
            _load_tutor_split(empty)
        except ValueError as exc:
            assert "non-empty" in str(exc)
        else:
            raise AssertionError("empty tutor data must fail explicitly")

    print("test_training_enn_data: PASSED")


if __name__ == "__main__":
    main()
