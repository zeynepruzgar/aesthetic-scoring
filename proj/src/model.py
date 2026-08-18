"""Regression heads placed on top of frozen backbone features.

Both constructors build a fresh model *and* a fresh optimizer on every call.
That is deliberate rather than incidental: reusing a partially trained model
across runs -- trivially easy when re-executing a single notebook cell -- means
the validation set has already been fitted, and validation scores stop measuring
generalisation.  Building both inside a function makes that mistake structurally
impossible.  The optimizer must be rebuilt too, since AdamW carries per-parameter
state that would otherwise leak between runs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Two-layer MLP mapping a feature vector to a single score.

    ReLU between the linear layers is what makes the model more than linear:
    without a nonlinearity, stacked Linear layers collapse mathematically into
    one.  Dropout randomly disables units during training so the head cannot
    lean too heavily on any small group of feature dimensions.
    """

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionHead(nn.Module):
    """Late fusion of two feature views.

    Each view is projected to a common width before concatenation rather than
    being concatenated raw.  Raw concatenation would let the wider view dominate
    purely by dimensionality -- ResNet-50 contributes 2048 dimensions against
    CLIP's 512 -- and the two also differ in scale, since CLIP embeddings are
    L2-normalised while pooled ResNet activations are not.  Projecting first puts
    both views on comparable footing and lets the model weigh them on merit.
    """

    def __init__(self, dims: list[int], proj: int = 256,
                 hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(d, proj), nn.ReLU()) for d in dims
        ])
        self.head = nn.Sequential(
            nn.Linear(proj * len(dims), hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, *views: torch.Tensor) -> torch.Tensor:
        projected = [p(v) for p, v in zip(self.projections, views)]
        return self.head(torch.cat(projected, dim=1))


def build_model(dims: list[int], seed: int = 0, device: str = "cuda",
                lr: float = 1e-3, weight_decay: float = 1e-4,
                **kwargs) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Construct a head sized for `dims` plus a matching optimizer.

    A single entry in `dims` gives an MLPHead; several give a FusionHead.
    """
    torch.manual_seed(seed)
    model = (MLPHead(dims[0], **kwargs) if len(dims) == 1
             else FusionHead(dims, **kwargs)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    return model, optimizer
