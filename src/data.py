"""Loading AADB labels and pairing them with image files on disk.

The AADB train/test split and ground-truth scores ship inside the authors'
GitHub repository as a MATLAB file (AADBinfo.mat, 175 KB).  Only that one file
is needed, so it is fetched directly rather than by cloning the repository,
which also carries 16 MB of Caffe and MATLAB demo code from 2016.  The image
archive is distributed separately via Google Drive.

Kong et al., "Photo Aesthetics Ranking Network with Attributes and Content
Adaptation", ECCV 2016.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

LABEL_URL = ("https://raw.githubusercontent.com/aimerykong/"
             "deepImageAestheticsAnalysis/master/AADBinfo.mat")


@dataclass
class Split:
    """One data split: image paths and their aesthetic scores, aligned by index."""

    paths: list[str]
    scores: np.ndarray

    def __len__(self) -> int:
        return len(self.paths)


def load_labels(mat_path: str) -> dict[str, tuple[list[str], np.ndarray]]:
    """Read AADBinfo.mat into {split: (filenames, scores)}.

    The .mat layout nests each filename inside a (1, 1) array, hence the [0].
    """
    mat = sio.loadmat(mat_path)
    # ravel() rather than [0] on the score arrays: MATLAB writes them as a 2-D
    # matrix and whether that is (1, N) or (N, 1) is not something to guess.
    # Guessing wrong yields a length-1 array, and build_split's zip would then
    # silently truncate the whole split to a single sample.
    return {
        "train": ([x[0] for x in mat["trainNameList"][0]],
                  mat["trainScore"].ravel()),
        "test": ([x[0] for x in mat["testNameList"][0]],
                 mat["testScore"].ravel()),
    }


def build_split(names: list[str], scores: np.ndarray, image_dir: str) -> Split:
    """Keep only labelled images that are actually present on disk.

    Names and scores are traversed together so their alignment survives
    filtering.  Filtering the two sequences independently would shift every
    subsequent pairing by one -- an error that raises nothing and destroys
    training silently.
    """
    if len(names) != len(scores):
        raise ValueError(
            f"{len(names)} filenames but {len(scores)} scores -- the label file "
            f"was parsed into misaligned arrays"
        )

    on_disk = set(os.listdir(image_dir))  # set -> O(1) membership tests
    paths, kept = [], []
    for name, score in zip(names, scores):
        if name in on_disk:
            paths.append(os.path.join(image_dir, name))
            kept.append(float(score))
    return Split(paths=paths, scores=np.asarray(kept, dtype=np.float32))


def load_aadb(image_dir: str, mat_path: str) -> dict[str, Split]:
    """Load both splits, with sanity checks that would otherwise fail late."""
    labels = load_labels(mat_path)
    splits = {k: build_split(n, s, image_dir) for k, (n, s) in labels.items()}

    train_names = set(os.path.basename(p) for p in splits["train"].paths)
    test_names = set(os.path.basename(p) for p in splits["test"].paths)
    overlap = train_names & test_names
    if overlap:
        raise ValueError(f"train/test overlap: {len(overlap)} images appear in both")

    for name, split in splits.items():
        missing = len(labels[name][0]) - len(split)
        if missing:
            print(f"[data] {name}: {missing} labelled images not found on disk")

    return splits


def describe(splits: dict[str, Split]) -> None:
    """Print split sizes and label statistics."""
    for name, s in splits.items():
        sc = s.scores
        print(
            f"[data] {name:5s} n={len(s):5d}  "
            f"range=[{sc.min():.2f}, {sc.max():.2f}]  "
            f"mean={sc.mean():.3f}  std={sc.std():.3f}"
        )
