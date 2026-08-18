"""Frozen backbones used as feature extractors.

Each backbone exposes the same interface: a torchvision-style preprocessing
transform and an `encode` callable mapping a batch of images to a feature
matrix.  Nothing here is trained -- the backbones are used purely to turn
images into vectors, and all learning happens in the regression head.

Two failure modes are worth naming because neither raises an error:

1.  Preprocessing constants.  CLIP and ImageNet models use different channel
    statistics.  Feeding CLIP ImageNet-normalised tensors (or vice versa)
    produces plausible-looking features that are quietly worse.  Each backbone
    therefore owns its own transform.

2.  QuickGELU.  The original OpenAI CLIP weights were trained with the
    QuickGELU activation.  Recent open_clip versions default the plain
    'ViT-B-32' name to standard GELU, which runs those weights under an
    activation they never saw.  It emits a UserWarning and degrades results, so
    the architecture name here is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Backbone:
    name: str
    dim: int
    transform: Callable
    encode: Callable[[torch.Tensor], torch.Tensor]


def load_clip(variant: str = "ViT-B-32-quickgelu",
              pretrained: str = "openai",
              device: str = "cuda") -> Backbone:
    """CLIP image encoder with L2-normalised output embeddings."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        variant, pretrained=pretrained
    )
    model = model.to(device).eval()

    dim = model.visual.output_dim

    def encode(batch: torch.Tensor) -> torch.Tensor:
        feats = model.encode_image(batch)
        # CLIP embeddings are designed to be used on the unit sphere.
        return feats / feats.norm(dim=-1, keepdim=True)

    return Backbone(f"clip_{variant}_{pretrained}", dim, preprocess, encode)


def load_resnet50(device: str = "cuda") -> Backbone:
    """ImageNet-pretrained ResNet-50 truncated to its 2048-d pooled features."""
    import torchvision
    from torchvision import transforms

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)

    # Replace the 1000-class classifier with a pass-through so forward() returns
    # the pooled convolutional features rather than class logits.
    model.fc = nn.Identity()
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return Backbone("resnet50_imagenet", 2048, transform, model)


BACKBONES = {
    "clip": load_clip,
    "clip-large": lambda device="cuda": load_clip("ViT-L-14", "openai", device),
    "resnet50": load_resnet50,
}


def get_backbone(name: str, device: str = "cuda") -> Backbone:
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    return BACKBONES[name](device=device)
