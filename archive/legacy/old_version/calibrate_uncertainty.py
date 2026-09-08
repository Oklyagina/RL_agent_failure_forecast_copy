"""
calibrate_uncertainty.py
========================
Build and save the percentile reference distributions for recommendation_uncertainty.py,
using the SAME model, scaler and data as your ENN training pipeline (src/training_enn.py).

It loads your trained ENN (load_trained_enn), your fitted StandardScaler (_load_scaler) and
your distillation training states (_load_npz), scales them exactly as in training, calibrates
the two reference distributions (total vacuity + chosen-action variance), and saves them.

Run ONCE, from the project root:
    python calibrate_uncertainty.py
"""
import numpy as np
from pathlib import Path

from src.config import CFG
from src.training_enn import load_trained_enn, _load_scaler, _load_npz

from recommendation_uncertainty import RecommendationUncertainty

CALIB_OUT = "enn_pctile_calib.npz"        # output calibration file


def main():
    # 1) Load the trained ENN and the fitted scaler EXACTLY as your pipeline does.
    #    (load_trained_enn builds the architecture from enn_meta_*.json, so it always matches.)
    model = load_trained_enn()
    scaler = _load_scaler()

    # 2) Load the distillation training states (raw) and scale them the same way as training.
    obs_raw, _ = _load_npz(Path(CFG.DISTILL_TRAIN_FILE))     # (N, input_dim) raw states
    if obs_raw is None:
        raise FileNotFoundError(f"Training states not found: {CFG.DISTILL_TRAIN_FILE}")
    obs_scaled = scaler.transform(obs_raw).astype(np.float32)
    print(f"[calib] reference states: {obs_scaled.shape}")

    # 3) Wrap the loaded model (architecture guaranteed correct) and calibrate.
    #    input_vectors are already scaled, so they are fed to the ENN as-is.
    #    The action reference uses the ENN's top action per state (no action remapping needed
    #    for calibration); at live scoring you pass the agent's chosen action label.
    ru = RecommendationUncertainty.from_model(model, scaler=scaler)
    ru.calibrate(input_vectors=obs_scaled)
    ru.save_calibration(CALIB_OUT)
    print(f"[calib] saved -> {CALIB_OUT}  ({len(obs_scaled)} samples)")

    # 4) Sanity check on a few reference states.
    for i in (0, len(obs_scaled) // 2, len(obs_scaled) - 1):
        tp, ap = ru.scores_vector(obs_scaled[i])
        print(f"  sample {i}: total_pctile={tp:.1f}, action_pctile={ap:.1f}")


if __name__ == "__main__":
    main()