# Aesthetic Image Scoring

Predicting continuous aesthetic quality scores for photographs by combining
frozen CLIP embeddings with CNN visual features, evaluated on perceptual ranking
quality.

## Approach

Both backbones are used purely as feature extractors and are never fine-tuned.
CLIP was pretrained on 400M image-text pairs and ResNet-50 on ImageNet, so their
representations already encode a great deal about content, composition and
style. Only a small regression head is trained, which makes each experiment take
seconds rather than minutes and keeps the whole study runnable on a free Colab
GPU.

Features are extracted once and cached to `.npy`; every experiment afterwards
operates on those arrays.

## Dataset

AADB (Aesthetics and Attributes Database), Kong et al., ECCV 2016. 10,000 Flickr
photographs rated by multiple annotators, official split 8,458 train / 1,000
test, continuous scores in [0, 1].

Labels ship inside the authors' repository:

```bash
git clone --depth 1 https://github.com/aimerykong/deepImageAestheticsAnalysis.git
```

Images must be downloaded manually from the authors'
[Google Drive folder](https://drive.google.com/drive/folders/0BxeylfSgpk1MOVduWGxyVlJFUHM?resourcekey=0-qecf-sZVexPbF6XLU4Gq_g)
— take `datasetImages_warp256.zip` (130 MB, images resized to 256×256; CLIP
resizes to 224 anyway, so the full-resolution archive is unnecessary). The links
predate the current Drive sharing format and cannot be resolved by `gdown`.

The archive holds 9,958 files but only 9,458 carry public labels; the remaining
500 are the paper's validation split, whose scores were never released. They are
ignored.

Images are Creative Commons licensed from Flickr and distributed for research
use only.

## Setup

```bash
pip install -r requirements.txt
unzip datasetImages_warp256.zip -d data/
```

## Usage

Extract features once per backbone:

```bash
python -m src.extract_features --backbone clip     --images data/datasetImages_warp256
python -m src.extract_features --backbone resnet50 --images data/datasetImages_warp256
```

Train and evaluate:

```bash
python -m src.train --features clip                        # CLIP only
python -m src.train --features resnet50                    # CNN only
python -m src.train --features clip resnet50               # late fusion
python -m src.train --features clip --loss mse+rank        # with ranking loss
python -m src.train --features clip resnet50 --loss mse+rank --results results.json
```

`notebooks/experiments.ipynb` runs the same experiments interactively and builds
the ablation table.

## Method notes

**Validation protocol.** 10% of the training split is held out for validation.
Early stopping and every other selection decision are made against it; the test
set is evaluated exactly once per configuration. Selecting a checkpoint by test
performance would leak the test set into model selection.

**Fusion.** Each view is projected to a common width before concatenation.
Concatenating raw would let ResNet-50's 2048 dimensions dominate CLIP's 512 by
dimensionality alone, and the two also differ in scale since CLIP embeddings are
L2-normalised while pooled ResNet activations are not.

**Ranking loss.** MSE optimises absolute error while the reported metrics measure
ordering. A pairwise margin term targets that mismatch directly and also counters
AADB's peaked label distribution, where predicting near the mean yields an
acceptable MSE and no ranking information.

**Metrics.** Spearman and Kendall rank correlations are invariant to monotonic
rescaling of predictions, which is the right property for a task where relative
judgement matters more than absolute calibration. Kendall is the stricter of the
two and reads lower on the same predictions.

## Results

| Features | Loss | SRCC | KRCC | MSE |
|---|---|---|---|---|
| CLIP ViT-B/32 | MSE | 0.7671 | 0.5888 | 0.0170 |

For reference, the original AADB paper reports SRCC 0.678 on this test set. The
comparison is not architecture against architecture: CLIP brings large-scale
pretraining that was not available in 2016, and the gap should be read as what
that pretraining buys rather than as a better head design.

## Reference

Kong, S., Shen, X., Lin, Z., Mech, R., Fowlkes, C. *Photo Aesthetics Ranking
Network with Attributes and Content Adaptation.* ECCV 2016.
