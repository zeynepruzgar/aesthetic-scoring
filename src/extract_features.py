"""Extract frozen backbone features for every AADB image and cache them to disk.

Run once per backbone; every downstream experiment then operates on the cached
arrays and takes seconds instead of minutes.

    python -m src.extract_features --backbone clip     --images data/datasetImages_warp256
    python -m src.extract_features --backbone resnet50 --images data/datasetImages_warp256

Outputs, written to --out:

    {backbone}_train.npy   (n_train, dim)   float32
    {backbone}_test.npy    (n_test,  dim)   float32
    scores_train.npy       (n_train,)       float32
    scores_test.npy        (n_test,)        float32

Row order is identical across every file, which is what makes features from
different backbones concatenable later.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .backbones import get_backbone, pick_device
from .data import describe, load_aadb


class ImageFolder(Dataset):
    """Minimal image dataset: applies a transform and returns the tensor."""

    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        # convert('RGB') guards against the greyscale images in AADB, which
        # would otherwise arrive with one channel and break the batch.
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)


@torch.no_grad()
def extract(paths: list[str], backbone, device: str,
            batch_size: int = 64, num_workers: int = 2) -> np.ndarray:
    """Encode every image, preserving input order.

    shuffle=False is load-bearing: the returned rows are matched to the score
    array positionally.  Shuffling here would pair each image's features with
    another image's label, and nothing downstream would complain.
    """
    loader = DataLoader(
        ImageFolder(paths, backbone.transform),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    chunks = []
    done = 0
    for i, batch in enumerate(loader, 1):
        chunks.append(backbone.encode(batch.to(device)).float().cpu())
        done += len(batch)   # not i * batch_size: the final batch is short
        if i % 20 == 0:
            print(f"  {done:6d} / {len(paths)}", flush=True)
    return torch.cat(chunks).numpy().astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="clip",
                   help="clip | clip-large | resnet50")
    p.add_argument("--images", required=True, help="directory of AADB images")
    p.add_argument("--labels", default="data/AADBinfo.mat")
    p.add_argument("--out", default="features", help="output directory")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None,
                   help="cuda | mps | cpu (default: best available)")
    p.add_argument("--overwrite", action="store_true",
                   help="re-extract even if cached files exist")
    args = p.parse_args()

    device = args.device or pick_device()
    os.makedirs(args.out, exist_ok=True)

    targets = {s: os.path.join(args.out, f"{args.backbone}_{s}.npy")
               for s in ("train", "test")}
    if not args.overwrite and all(os.path.exists(p) for p in targets.values()):
        print(f"[extract] {args.backbone}: cached, nothing to do "
              f"(use --overwrite to force)")
        return

    splits = load_aadb(args.images, args.labels)
    describe(splits)

    print(f"[extract] loading backbone {args.backbone!r} on {device}")
    backbone = get_backbone(args.backbone, device=device)
    print(f"[extract] output dimension: {backbone.dim}")

    for name, split in splits.items():
        print(f"[extract] {name}: {len(split)} images")
        feats = extract(split.paths, backbone, device,
                        args.batch_size, args.num_workers)
        np.save(targets[name], feats)

        score_path = os.path.join(args.out, f"scores_{name}.npy")
        np.save(score_path, split.scores)
        print(f"[extract] wrote {targets[name]} {feats.shape}")

    print("[extract] done")


if __name__ == "__main__":
    main()
