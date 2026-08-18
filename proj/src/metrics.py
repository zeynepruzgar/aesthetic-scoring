"""Evaluation metrics for continuous aesthetic scoring.

Aesthetic quality is relative: whether a photo scores 0.62 or 0.71 matters far
less than whether it ranks above a photo humans liked less.  Rank correlations
capture exactly that and are invariant to any monotonic rescaling of the
predictions, so a model whose outputs are uniformly shifted or compressed is not
penalised for it.

Spearman (SRCC) correlates the rank positions; Kendall (KRCC) counts concordant
versus discordant pairs and is the stricter of the two -- expect it to read
several points below SRCC on the same predictions.

MSE is reported for completeness but is not what models are selected on.  With
AADB's peaked score distribution, predicting near the mean for everything yields
a decent MSE and a near-zero correlation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Return SRCC, KRCC and MSE for one set of predictions."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape}")

    return {
        "SRCC": float(spearmanr(pred, true).correlation),
        "KRCC": float(kendalltau(pred, true).correlation),
        "MSE": float(np.mean((pred - true) ** 2)),
    }


def format_metrics(m: dict[str, float]) -> str:
    return f"SRCC {m['SRCC']:.4f} | KRCC {m['KRCC']:.4f} | MSE {m['MSE']:.4f}"
