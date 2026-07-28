"""
training/train_enn.py -- step 2: train the ENN by behavior cloning on the
(observation, action) pairs collected by collect_rollouts.py, then export
every artifact the uncertainty module needs.

Outputs (in --out-dir):
    enn_<agent>.pth              trained ENN weights (state_dict)
    scaler_params.json           mean/std of the StandardScaler (JSON)
    enn_meta.json                input_dim, num_classes, class_mapping
                                 (identity: label k == curated action k)

After this, run the existing calibrate_uncertainty.py on a set of reference
observations to produce the percentile calibration (.npz). See TRAINING.md.

NOTE(margarida): the loss below is the standard evidential-classification
objective (Bayes-risk cross-entropy + annealed KL to the uniform Dirichlet,
Sensoy et al. 2018). If the CurriculumAgent ENN was trained with a different
objective or schedule, align this file with the original training script so
both ENNs are comparable.

Usage:
    python training/train_enn.py --data-dir data_expert --out-dir models_expert \
        --agent-name expert --epochs 100
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# TODO(margarida): confirm import path of the ENN architecture.
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.enn_models import EvidentialNetwork


# --- evidential loss ---------------------------------------------------------
def alpha_from_output(out) -> torch.Tensor:
    """The EvidentialNetwork forward returns a dict {"prob", "S",
    "uncertainty", ...}. Recover the Dirichlet parameters:
    alpha_k = p_k * S (or use out["alpha"] directly when present)."""
    if isinstance(out, dict):
        if "alpha" in out:
            return out["alpha"]
        return out["prob"] * out["S"].reshape(-1, 1)
    return F.softplus(out) + 1.0                      # logits fallback


def evidential_loss(alpha: torch.Tensor, targets: torch.Tensor,
                    epoch: int, anneal_epochs: int) -> torch.Tensor:
    """Bayes-risk cross-entropy for a Dirichlet(alpha) + annealed KL term."""
    S = alpha.sum(dim=1, keepdim=True)
    y = F.one_hot(targets, num_classes=alpha.shape[1]).float()

    # Bayes-risk CE:  sum_k y_k (digamma(S) - digamma(alpha_k))
    ce = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)

    # KL(Dir(alpha_tilde) || Dir(1)) on the misleading evidence
    alpha_tilde = y + (1.0 - y) * alpha
    S_t = alpha_tilde.sum(dim=1, keepdim=True)
    K = alpha.shape[1]
    kl = (torch.lgamma(S_t).squeeze(1) - torch.lgamma(alpha_tilde).sum(dim=1)
          - torch.lgamma(torch.tensor(float(K), device=alpha.device))
          + ((alpha_tilde - 1.0)
             * (torch.digamma(alpha_tilde) - torch.digamma(S_t))).sum(dim=1))
    lam = min(1.0, epoch / max(1, anneal_epochs))
    return (ce + lam * kl).mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--agent-name", type=str, default="expert")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--anneal-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Data ------------------------------------------------------------------
    X = np.load(args.data_dir / "observations.npy")
    y = np.load(args.data_dir / "labels.npy")
    actions = np.load(args.data_dir / "actions.npy")
    num_classes = int(actions.shape[0])
    print(f"data: {X.shape[0]} pairs, obs_dim={X.shape[1]}, "
          f"num_classes={num_classes}")

    # 2. Scaler (fit on TRAIN split only) --------------------------------------
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=args.val_frac, random_state=args.seed, stratify=None)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)

    dl_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr),
                                     torch.from_numpy(y_tr)),
                       batch_size=args.batch_size, shuffle=True)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(X_va),
                                     torch.from_numpy(y_va)),
                       batch_size=args.batch_size)

    # 3. Train -----------------------------------------------------------------
    enn = EvidentialNetwork(input_dim=X.shape[1],
                            num_classes=num_classes).to(device)
    opt = torch.optim.Adam(enn.parameters(), lr=args.lr)
    best_va, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        enn.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            alpha = alpha_from_output(enn(xb))
            loss = evidential_loss(alpha, yb, epoch, args.anneal_epochs)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        enn.eval()
        va_loss, va_hits, n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                alpha = alpha_from_output(enn(xb))
                va_loss += evidential_loss(
                    alpha, yb, epoch, args.anneal_epochs).item() * len(xb)
                va_hits += (alpha.argmax(1) == yb).sum().item()
                n += len(xb)
        va_loss /= n
        if va_loss < best_va:
            best_va = va_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in enn.state_dict().items()}
        print(f"epoch {epoch:3d}  train={tr_loss / len(X_tr):.4f}  "
              f"val={va_loss:.4f}  val_acc={va_hits / n:.3f}")

    # 4. Export artifacts ------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.out_dir / f"enn_{args.agent_name}.pth")
    (args.out_dir / "scaler_params.json").write_text(json.dumps({
        "type": "StandardScaler",
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_features_in": int(scaler.n_features_in_),
    }))
    (args.out_dir / "enn_meta.json").write_text(json.dumps({
        "input_dim": int(X.shape[1]),
        "num_classes": num_classes,
        "n_curated_actions": num_classes,
        "environment": "l2rpn_icaps_2021_small",
        "grid2op_version": "1.9.8",
        "agent": args.agent_name,
        "class_mapping": {int(k): int(k) for k in range(num_classes)},
    }))
    print(f"\n[ok] wrote enn_{args.agent_name}.pth, scaler_params.json, "
          f"enn_meta.json in {args.out_dir}/")

    # 5. Percentile calibration (automatic) -----------------------------------
    # The scaler is created HERE, at training time; the calibration reference
    # is built right after, on (a subset of) the scaled training states.
    from recommendation_uncertainty import build_calibration, save_calibration
    enn.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    enn.eval()
    n_cal = min(len(X), 5000)
    idx = np.random.RandomState(args.seed).choice(len(X), n_cal, replace=False)
    X_cal = scaler.transform(X[idx]).astype(np.float32)
    total_ref, action_ref = build_calibration(enn.cpu(), X_cal)
    calib_path = args.out_dir / "enn_pctile_calib.npz"
    save_calibration(calib_path, total_ref, action_ref)
    print(f"[ok] wrote {calib_path} (calibration on {n_cal} training states)")
    print("Done -- run_example.py will pick these artifacts up "
          "automatically.")


if __name__ == "__main__":
    main()
