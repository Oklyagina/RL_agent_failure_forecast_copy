"""Tune ENN parameters on an existing rollout bundle.

This script is intentionally separate from ``training/train_enn.py``.  It uses
the same active rollout files and ENN implementation, trains one fresh model per
configuration, ranks candidates by validation loss, and writes isolated tuning
artifacts under ``artifacts/<ENV_NAME>/<agent>/tuning``.

Usage:
    python training/tune_enn.py
    python training/tune_enn.py --device cuda
    python training/tune_enn.py --epochs 100
    python training/tune_enn.py --smoke-test --epochs 2
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_config import (  # noqa: E402
    AGENT_NAME,
    ARTIFACTS_DIR,
    ENN_ANNEAL_EPOCHS,
    ENN_EPOCHS,
    ENN_LR,
    ENN_VAL_FRAC,
    ENV_NAME,
    SEED,
)
from src.enn_models import EvidentialNetwork  # noqa: E402
from training.train_enn import alpha_from_output, evidential_loss  # noqa: E402


@dataclass(frozen=True)
class CandidateConfig:
    lr: float
    hidden_dim: int
    dropout: float
    batch_size: int


@dataclass
class CandidateResult:
    run_id: int
    lr: float
    hidden_dim: int
    dropout: float
    batch_size: int
    epochs_trained: int
    train_loss: float
    validation_loss: float
    validation_accuracy: float
    runtime_seconds: float
    device: str
    seed: int


@dataclass
class PreparedData:
    input_dim: int
    num_classes: int
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    scaler: StandardScaler


def default_rollout_dir(agent_name: str) -> Path:
    return ARTIFACTS_DIR / ENV_NAME / agent_name / "rollouts"


def default_tuning_dir(agent_name: str) -> Path:
    return ARTIFACTS_DIR / ENV_NAME / agent_name / "tuning"


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower().strip()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    cuda_available = torch.cuda.is_available()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if cuda_available:
            return torch.device("cuda")
        print("[tune_enn] CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if cuda_available else "cpu")


def load_rollout_arrays(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations_path = data_dir / "observations.npy"
    labels_path = data_dir / "labels.npy"
    actions_path = data_dir / "actions.npy"
    missing = [
        str(path) for path in (observations_path, labels_path, actions_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing rollout files: " + ", ".join(missing))

    observations = np.load(observations_path).astype(np.float32)
    labels = np.load(labels_path).astype(np.int64).reshape(-1)
    actions = np.load(actions_path)
    if observations.ndim != 2 or labels.ndim != 1:
        raise ValueError(
            f"Invalid rollout shapes: observations={observations.shape}, labels={labels.shape}"
        )
    if len(observations) == 0 or len(observations) != len(labels):
        raise ValueError(
            f"Rollout row mismatch: observations={len(observations)}, labels={len(labels)}"
        )
    if actions.ndim != 2 or len(actions) < 2:
        raise ValueError(f"actions.npy must be 2-D with at least two rows, got {actions.shape}")
    if int(labels.min()) < 0 or int(labels.max()) >= len(actions):
        raise ValueError(
            f"labels must be in [0, {len(actions) - 1}], got "
            f"[{int(labels.min())}, {int(labels.max())}]"
        )
    return observations, labels, actions


def prepare_data(
    observations: np.ndarray,
    labels: np.ndarray,
    actions: np.ndarray,
    *,
    val_frac: float,
    seed: int,
) -> PreparedData:
    train_x, validation_x, train_y, validation_y = train_test_split(
        observations,
        labels,
        test_size=val_frac,
        random_state=seed,
        stratify=None,
    )
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x).astype(np.float32)
    validation_x = scaler.transform(validation_x).astype(np.float32)
    return PreparedData(
        input_dim=int(observations.shape[1]),
        num_classes=int(actions.shape[0]),
        train_x=train_x,
        train_y=train_y.astype(np.int64),
        validation_x=validation_x,
        validation_y=validation_y.astype(np.int64),
        scaler=scaler,
    )


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=min(int(batch_size), len(x)),
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def evaluate(
    model: EvidentialNetwork,
    loader: DataLoader,
    *,
    device: torch.device,
    epoch: int,
    anneal_epochs: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n_rows = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            alpha = alpha_from_output(model(xb))
            loss = evidential_loss(alpha, yb, epoch, anneal_epochs)
            batch_size = len(xb)
            total_loss += float(loss.item()) * batch_size
            correct += int((alpha.argmax(1) == yb).sum().item())
            n_rows += batch_size
    if n_rows == 0:
        raise RuntimeError("Evaluation DataLoader yielded no rows.")
    return total_loss / n_rows, correct / n_rows


def train_candidate(
    config: CandidateConfig,
    data: PreparedData,
    *,
    run_id: int,
    total_runs: int,
    epochs: int,
    anneal_epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[CandidateResult, dict[str, torch.Tensor]]:
    set_random_seeds(seed)
    train_loader = make_loader(
        data.train_x,
        data.train_y,
        batch_size=config.batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = make_loader(
        data.validation_x,
        data.validation_y,
        batch_size=config.batch_size,
        shuffle=False,
        seed=seed,
    )
    model = EvidentialNetwork(
        input_dim=data.input_dim,
        num_classes=data.num_classes,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    print(
        f"\n[tune_enn] Configuration {run_id}/{total_runs}: "
        f"lr={config.lr:g}, hidden_dim={config.hidden_dim}, "
        f"dropout={config.dropout:g}, batch_size={config.batch_size}"
    )
    t0 = time.time()
    best_validation_loss = float("inf")
    best_state = None
    final_train_loss = float("nan")
    final_validation_accuracy = float("nan")

    progress = tqdm(range(1, epochs + 1), desc=f"run {run_id}/{total_runs}", unit="epoch")
    for epoch in progress:
        model.train()
        train_loss = 0.0
        train_rows = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            alpha = alpha_from_output(model(xb))
            loss = evidential_loss(alpha, yb, epoch, anneal_epochs)
            loss.backward()
            optimizer.step()
            batch_size = len(xb)
            train_loss += float(loss.item()) * batch_size
            train_rows += batch_size
        if train_rows == 0:
            raise RuntimeError("Training DataLoader yielded no rows.")

        final_train_loss = train_loss / train_rows
        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            device=device,
            epoch=epoch,
            anneal_epochs=anneal_epochs,
        )
        final_validation_accuracy = validation_accuracy
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        progress.set_postfix({
            "train": f"{final_train_loss:.4f}",
            "val": f"{validation_loss:.4f}",
            "val_acc": f"{validation_accuracy:.3f}",
            "best": f"{best_validation_loss:.4f}",
        })

    if best_state is None:
        raise RuntimeError(f"Configuration {run_id} did not produce a checkpoint.")
    result = CandidateResult(
        run_id=run_id,
        lr=config.lr,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        batch_size=config.batch_size,
        epochs_trained=epochs,
        train_loss=final_train_loss,
        validation_loss=best_validation_loss,
        validation_accuracy=final_validation_accuracy,
        runtime_seconds=time.time() - t0,
        device=str(device),
        seed=seed,
    )
    return result, best_state


def build_search_space(
    *,
    lr_values: Iterable[float],
    hidden_dim_values: Iterable[int],
    dropout_values: Iterable[float],
    batch_size_values: Iterable[int],
) -> list[CandidateConfig]:
    return [
        CandidateConfig(lr=lr, hidden_dim=hidden_dim, dropout=dropout, batch_size=batch_size)
        for lr, hidden_dim, dropout, batch_size in itertools.product(
            lr_values,
            hidden_dim_values,
            dropout_values,
            batch_size_values,
        )
    ]


def smoke_search_space() -> list[CandidateConfig]:
    return [
        CandidateConfig(lr=ENN_LR, hidden_dim=16, dropout=0.0, batch_size=16),
        CandidateConfig(lr=ENN_LR, hidden_dim=32, dropout=0.05, batch_size=16),
    ]


def write_results(
    results: list[CandidateResult],
    *,
    output_dir: Path,
    best_checkpoint_name: str,
    best_state: dict[str, torch.Tensor],
) -> tuple[Path, Path, Path, Path]:
    if not results:
        raise ValueError("No tuning results to write.")
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results, key=lambda row: row.validation_loss)
    best = ranked[0]
    csv_path = output_dir / "enn_tuning_results.csv"
    json_path = output_dir / "enn_tuning_results.json"
    best_params_path = output_dir / "best_enn_parameters.csv"
    best_path = output_dir / best_checkpoint_name

    fieldnames = list(asdict(best).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in ranked:
            writer.writerow(asdict(result))

    payload = {
        "selection_metric": "validation_loss",
        "selection_mode": "min",
        "best_run_id": best.run_id,
        "best_validation_loss": best.validation_loss,
        "best_parameters": {
            "lr": best.lr,
            "hidden_dim": best.hidden_dim,
            "dropout": best.dropout,
            "batch_size": best.batch_size,
        },
        "runs": [asdict(result) for result in ranked],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with best_params_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["env_key", "parameter", "value"])
        writer.writeheader()
        writer.writerows([
            {"env_key": "ENN_LR", "parameter": "lr", "value": f"{best.lr:g}"},
            {
                "env_key": "ENN_HIDDEN_DIM",
                "parameter": "hidden_dim",
                "value": str(best.hidden_dim),
            },
            {
                "env_key": "ENN_DROPOUT",
                "parameter": "dropout",
                "value": f"{best.dropout:g}",
            },
            {
                "env_key": "ENN_BATCH_SIZE",
                "parameter": "batch_size",
                "value": str(best.batch_size),
            },
        ])

    torch.save(best_state, best_path)
    return csv_path, json_path, best_params_path, best_path


def print_ranking(results: list[CandidateResult]) -> None:
    ranked = sorted(results, key=lambda row: row.validation_loss)
    print("\n[tune_enn] Final ranking by validation_loss")
    print("run | lr | hidden_dim | dropout | batch_size | val_loss | val_acc | train_loss | runtime_s")
    for result in ranked:
        print(
            f"{result.run_id:3d} | {result.lr:g} | {result.hidden_dim:10d} | "
            f"{result.dropout:g} | {result.batch_size:10d} | "
            f"{result.validation_loss:.6f} | {result.validation_accuracy:.4f} | "
            f"{result.train_loss:.6f} | {result.runtime_seconds:.1f}"
        )
    best = ranked[0]
    print("\nBEST PARAMETERS")
    print(
        f"lr={best.lr:g}, hidden_dim={best.hidden_dim}, "
        f"dropout={best.dropout:g}, batch_size={best.batch_size}"
    )
    print(f"best_validation_loss={best.validation_loss:.6f} (run {best.run_id})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", default=AGENT_NAME)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=ENN_EPOCHS)
    parser.add_argument("--anneal-epochs", type=int, default=ENN_ANNEAL_EPOCHS)
    parser.add_argument("--val-frac", type=float, default=ENN_VAL_FRAC)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--lr-grid", default="1e-3,3e-3,5e-3")
    parser.add_argument("--hidden-dim-grid", default="128,256,512")
    parser.add_argument("--dropout-grid", default="0.05,0.15,0.3")
    parser.add_argument("--batch-size-grid", default="512,1024")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run two tiny configurations for pipeline validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}")
    if not 0.0 < args.val_frac < 1.0:
        raise ValueError(f"val-frac must be in (0, 1), got {args.val_frac}")

    data_dir = args.data_dir or default_rollout_dir(args.agent_name)
    output_dir = args.out_dir or default_tuning_dir(args.agent_name)
    device = resolve_device(args.device)

    observations, labels, actions = load_rollout_arrays(data_dir)
    data = prepare_data(
        observations,
        labels,
        actions,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    search_space = smoke_search_space() if args.smoke_test else build_search_space(
        lr_values=parse_float_list(args.lr_grid),
        hidden_dim_values=parse_int_list(args.hidden_dim_grid),
        dropout_values=parse_float_list(args.dropout_grid),
        batch_size_values=parse_int_list(args.batch_size_grid),
    )
    if not search_space:
        raise ValueError("Search space is empty.")

    print(
        f"[tune_enn] data={data_dir} rows={len(observations)} "
        f"train={len(data.train_y)} validation={len(data.validation_y)} "
        f"input_dim={data.input_dim} num_classes={data.num_classes}"
    )
    print(f"[tune_enn] device={device} epochs={args.epochs} seed={args.seed}")
    print(f"[tune_enn] overall configurations: {len(search_space)}")

    results = []
    overall_best_state = None
    overall_best_loss = float("inf")
    for index, config in enumerate(search_space, start=1):
        result, best_state = train_candidate(
            config,
            data,
            run_id=index,
            total_runs=len(search_space),
            epochs=args.epochs,
            anneal_epochs=args.anneal_epochs,
            seed=args.seed,
            device=device,
        )
        results.append(result)
        if result.validation_loss < overall_best_loss:
            overall_best_loss = result.validation_loss
            overall_best_state = best_state
        current_best = min(results, key=lambda row: row.validation_loss)
        print(
            f"[tune_enn] best so far: run {current_best.run_id} "
            f"validation_loss={current_best.validation_loss:.6f}"
        )

    if overall_best_state is None:
        raise RuntimeError("No best checkpoint state was produced.")
    csv_path, json_path, best_params_path, best_path = write_results(
        results,
        output_dir=output_dir,
        best_checkpoint_name=f"best_enn_{args.agent_name}.pth",
        best_state=overall_best_state,
    )
    print_ranking(results)
    print(f"\n[tune_enn] wrote {csv_path}")
    print(f"[tune_enn] wrote {json_path}")
    print(f"[tune_enn] wrote {best_params_path}")
    print(f"[tune_enn] wrote {best_path}")
    print("[tune_enn] production ENN artifacts were not modified.")


if __name__ == "__main__":
    main()
