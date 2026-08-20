"""Training loop, loss functions, and the experiment CLI.

    python -m src.train --features clip
    python -m src.train --features resnet50
    python -m src.train --features clip resnet50          # late fusion
    python -m src.train --features clip --loss mse+rank   # add ranking term
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .metrics import compute_metrics, format_metrics
from .model import build_model


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor,
                          margin: float = 0.0) -> torch.Tensor:
    """Margin ranking loss over all pairs within the batch.

    MSE optimises absolute error while the metrics we report measure ordering,
    so the training objective and the evaluation criterion are not the same
    quantity.  This term closes that gap by penalising inverted pairs directly:
    for every (i, j) it asks only that the model rank them the way the
    annotators did.

    It also pushes back against AADB's peaked label distribution.  Predicting
    near the mean everywhere is an MSE-friendly strategy that carries no
    ranking information; under a pairwise term that strategy is penalised for
    every pair it fails to separate.

    Pairs with equal targets carry no ordering information and are masked out.
    """
    diff_pred = pred - pred.t()          # (B, B)
    diff_true = target - target.t()

    sign = torch.sign(diff_true)
    mask = sign != 0                     # drop ties and the diagonal

    if not mask.any():
        return pred.sum() * 0.0          # keeps the graph connected

    losses = torch.clamp(margin - sign * diff_pred, min=0.0)
    return losses[mask].mean()


def make_loss(name: str, rank_weight: float = 1.0):
    """Return a callable(pred, target) -> scalar loss."""
    mse = nn.MSELoss()
    if name == "mse":
        return mse
    if name == "rank":
        return lambda p, t: pairwise_ranking_loss(p, t)
    if name == "mse+rank":
        return lambda p, t: mse(p, t) + rank_weight * pairwise_ranking_loss(p, t)
    raise KeyError(f"unknown loss {name!r}; choose mse | rank | mse+rank")


# --------------------------------------------------------------------------
# Data plumbing
# --------------------------------------------------------------------------

def load_features(names: list[str], split: str, feature_dir: str
                  ) -> tuple[list[np.ndarray], np.ndarray]:
    """Load one or more feature matrices plus the shared score vector.

    Every backbone writes rows in the same order, so features from different
    backbones line up index by index.  The assertion below is cheap insurance
    against a stale or partial cache.
    """
    feats = [np.load(os.path.join(feature_dir, f"{n}_{split}.npy")) for n in names]
    scores = np.load(os.path.join(feature_dir, f"scores_{split}.npy"))

    for n, f in zip(names, feats):
        if len(f) != len(scores):
            raise ValueError(
                f"{n}_{split}.npy has {len(f)} rows but scores_{split}.npy has "
                f"{len(scores)} -- caches are out of sync, re-extract"
            )
    return feats, scores


def to_tensors(feats: list[np.ndarray], scores: np.ndarray
               ) -> tuple[list[torch.Tensor], torch.Tensor]:
    X = [torch.tensor(f, dtype=torch.float32) for f in feats]
    # unsqueeze(1): targets must be (N, 1) to match the model output.  Left as
    # (N,) they would broadcast against predictions into an (N, N) loss.
    y = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)
    return X, y


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

@torch.no_grad()
def predict(model: nn.Module, views: list[torch.Tensor], device: str,
            batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(views[0]), batch_size):
        chunk = [v[i:i + batch_size].to(device) for v in views]
        out.append(model(*chunk).cpu())
    return torch.cat(out).numpy().ravel()


def train(train_views: list[torch.Tensor], y_train: torch.Tensor,
          val_views: list[torch.Tensor], y_val: torch.Tensor,
          epochs: int = 50, batch_size: int = 128, loss_name: str = "mse",
          rank_weight: float = 1.0, seed: int = 0, device: str = "cuda",
          verbose: bool = True, **model_kwargs):
    """Fit a head, keeping the weights that scored best on validation SRCC.

    Early stopping is not an optimisation here but a correctness requirement:
    this model peaks around epoch 10 and then declines while training loss keeps
    falling.  Selection is made against validation only -- the test set is
    touched once, after training, by the caller.  Choosing a checkpoint by test
    performance would leak the test set into model selection and inflate every
    number that follows.
    """
    dims = [v.shape[1] for v in train_views]
    model, optimizer = build_model(dims, seed=seed, device=device, **model_kwargs)
    criterion = make_loss(loss_name, rank_weight)

    loader = DataLoader(
        TensorDataset(*train_views, y_train),
        batch_size=batch_size,
        shuffle=True,   # correct here, unlike extraction: batches should vary
    )

    best = {"srcc": -1.0, "epoch": 0, "state": None}
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        for *views, targets in loader:
            views = [v.to(device) for v in views]
            targets = targets.to(device)
            optimizer.zero_grad()
            criterion(model(*views), targets).backward()
            optimizer.step()

        val_srcc = compute_metrics(
            predict(model, val_views, device), y_val.numpy()
        )["SRCC"]
        history.append(val_srcc)

        if val_srcc > best["srcc"]:
            best.update(
                srcc=val_srcc,
                epoch=epoch,
                # clone(): without it the dict holds references that later
                # epochs overwrite in place, restoring the last weights instead
                # of the best ones.
                state={k: v.clone() for k, v in model.state_dict().items()},
            )

    # NaN compares False against everything, so a run whose validation SRCC is
    # never finite leaves best["state"] unset.  Without this it would surface as
    # an opaque TypeError inside load_state_dict.
    if best["state"] is None:
        raise RuntimeError(
            "no epoch produced a finite validation SRCC -- predictions were "
            "probably constant or diverged; check the learning rate and that "
            "the cached features are not all zeros"
        )

    model.load_state_dict(best["state"])
    if verbose:
        print(f"[train] best epoch {best['epoch']}/{epochs} "
              f"| val SRCC {best['srcc']:.4f}")
    return model, {"best_epoch": best["epoch"],
                   "val_srcc": best["srcc"],
                   "history": history}


def run_experiment(feature_names: list[str], feature_dir: str,
                   val_fraction: float = 0.1, seed: int = 0,
                   device: str | None = None, save: str | None = None,
                   **train_kwargs) -> dict:
    """End-to-end: load cached features, fit, evaluate once on test.

    `save` writes a checkpoint that src.score can load to rate new images.
    """
    # Deliberately not pick_device(): the head is small enough that on Apple
    # MPS the per-batch host-device transfers cost more than the compute they
    # save, and CPU is the faster choice.  Feature extraction is where the
    # accelerator matters, and that step selects one on its own.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tr_feats, tr_scores = load_features(feature_names, "train", feature_dir)
    te_feats, te_scores = load_features(feature_names, "test", feature_dir)

    Xtr_all, ytr_all = to_tensors(tr_feats, tr_scores)
    Xte, yte = to_tensors(te_feats, te_scores)

    # One shared split index keeps every view consistent with the labels.
    idx_tr, idx_val = train_test_split(
        np.arange(len(ytr_all)), test_size=val_fraction, random_state=seed
    )
    Xtr = [v[idx_tr] for v in Xtr_all]
    Xval = [v[idx_val] for v in Xtr_all]
    ytr, yval = ytr_all[idx_tr], ytr_all[idx_val]

    model, info = train(Xtr, ytr, Xval, yval, seed=seed, device=device,
                        **train_kwargs)

    test_pred = predict(model, Xte, device)
    test_metrics = compute_metrics(test_pred, yte.numpy())
    print(f"[test ] {format_metrics(test_metrics)}")

    if save:
        # The test predictions travel with the weights so that a score for a new
        # image can be placed against a known distribution.  A raw 0.58 means
        # little on its own -- what a viewer wants to know is where it lands
        # relative to other photographs, and that needs a reference sample.
        torch.save({
            "state_dict": model.state_dict(),
            "features": feature_names,
            "dims": [int(v.shape[1]) for v in Xtr],
            "model_kwargs": {k: v for k, v in train_kwargs.items()
                             if k in ("hidden", "dropout")},
            "reference": torch.tensor(test_pred, dtype=torch.float32),
        }, save)
        print(f"[train] saved model to {save}")

    return {"features": feature_names,
            "dims": [v.shape[1] for v in Xtr],
            **{k: v for k, v in info.items() if k != "history"},
            **test_metrics}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", nargs="+", default=["clip"],
                   help="backbone names; several enables late fusion")
    p.add_argument("--feature-dir", default="features")
    p.add_argument("--loss", default="mse", choices=["mse", "rank", "mse+rank"])
    p.add_argument("--rank-weight", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results", default=None,
                   help="append the result row to this JSON file")
    p.add_argument("--save", default=None,
                   help="write the trained head here for use with src.score")
    args = p.parse_args()

    result = run_experiment(
        args.features, args.feature_dir,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        loss_name=args.loss, rank_weight=args.rank_weight,
        hidden=args.hidden, dropout=args.dropout, lr=args.lr,
        save=args.save,
    )
    result["loss"] = args.loss
    result["seed"] = args.seed

    if args.results:
        rows = []
        if os.path.exists(args.results):
            with open(args.results) as f:
                rows = json.load(f)
        rows.append(result)
        with open(args.results, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[train] appended to {args.results}")


if __name__ == "__main__":
    main()
