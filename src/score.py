"""Score new photographs with a trained head.

    python -m src.score --model model.pt --images photo.jpg
    python -m src.score --model model.pt --images album/ --top 10

Unlike the rest of the pipeline this path is not batched over a cached feature
matrix: an image arrives, goes through the same frozen backbone the head was
trained on, and comes out as one number.  The backbone must match the one named
in the checkpoint, which is why the checkpoint records it rather than leaving it
to the caller to remember.

Scores are reported two ways.  The raw value is what the head predicts on AADB's
0-1 scale, but on its own it is hard to read -- the label distribution is peaked
and almost everything lands between 0.4 and 0.7.  The percentile places the
image against the model's predictions on the held-out test set, which is the
comparison a person actually wants: not "is this a 0.58" but "is this better
than most photographs".
"""

from __future__ import annotations

import argparse
import os

import torch
from PIL import Image

from .backbones import get_backbone, pick_device
from .model import build_model

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_images(paths: list[str]) -> list[str]:
    """Expand directories into their image files; keep plain files as given."""
    out = []
    for path in paths:
        if os.path.isdir(path):
            out.extend(
                os.path.join(path, f) for f in sorted(os.listdir(path))
                if f.lower().endswith(EXTENSIONS)
            )
        else:
            out.append(path)
    if not out:
        raise SystemExit("no images found")
    return out


def load_head(model_path: str, device: str):
    """Rebuild the trained head from a checkpoint written by src.train --save."""
    # Loaded onto CPU and moved by load_state_dict, so that the reference
    # distribution stays on CPU where the percentile is computed.  Comparing a
    # CPU prediction against an MPS reference returns the same value for every
    # image instead of raising -- silently wrong, which is worse than a crash.
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model, _ = build_model(ckpt["dims"], device=device, **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["features"], ckpt["reference"]


@torch.no_grad()
def score_images(paths: list[str], model_path: str, device: str | None = None,
                 batch_size: int = 32) -> list[tuple[str, float, float]]:
    """Return (path, score, percentile) for every image, in input order."""
    device = device or pick_device()
    head, feature_names, reference = load_head(model_path, device)
    backbones = [get_backbone(n, device=device) for n in feature_names]

    results = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in chunk]

        # One view per backbone, each with its own preprocessing -- the same
        # separation the extraction path keeps, for the same reason.
        views = [
            b.encode(torch.stack([b.transform(im) for im in images]).to(device)).float()
            for b in backbones
        ]
        preds = head(*views).cpu().ravel()

        for path, pred in zip(chunk, preds):
            pct = float((reference < pred).float().mean()) * 100
            results.append((path, float(pred), pct))
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="model.pt", help="checkpoint from src.train --save")
    p.add_argument("--images", nargs="+", required=True,
                   help="image files, or directories to scan")
    p.add_argument("--top", type=int, default=None,
                   help="show only the N highest-scoring images")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="cuda | mps | cpu")
    args = p.parse_args()

    paths = collect_images(args.images)
    print(f"[score] {len(paths)} image(s)")
    results = score_images(paths, args.model, args.device, args.batch_size)

    # Ranking is the point of the model, so show them ranked.
    results.sort(key=lambda r: -r[1])
    if args.top:
        results = results[:args.top]

    width = max(len(os.path.basename(p)) for p, _, _ in results)
    for path, score, pct in results:
        bar = "#" * round(pct / 5)
        print(f"{os.path.basename(path):{width}s}  {score:.3f}  "
              f"top {100 - pct:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
