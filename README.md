# Aesthetic Image Scoring

Predicting continuous aesthetic quality scores for photographs by combining
frozen CLIP embeddings with CNN visual features, evaluated on perceptual ranking
quality.

## Approach

Both backbones are used purely as feature extractors and are never fine-tuned.
CLIP was pretrained on 400M image-text pairs and ResNet-50 on ImageNet, so their
representations already encode a great deal about content, composition and
style. Only a small regression head is trained, which makes each experiment take
seconds rather than minutes and keeps the whole study runnable on a laptop.

Features are extracted once and cached to `.npy`; every experiment afterwards
operates on those arrays.

## Dataset

AADB (Aesthetics and Attributes Database), Kong et al., ECCV 2016. 10,000 Flickr
photographs rated by multiple annotators, official split 8,458 train / 1,000
test, continuous scores in [0, 1].

Labels live in a single 175 KB MATLAB file, `AADBinfo.mat`, inside the authors'
repository; Setup below fetches just that file. Cloning the repository instead
would drag in 16 MB of Caffe and MATLAB demo code from 2016 that nothing here
uses.

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
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data
curl -sLo data/AADBinfo.mat \
  https://raw.githubusercontent.com/aimerykong/deepImageAestheticsAnalysis/master/AADBinfo.mat
unzip datasetImages_warp256.zip -d data/
```

`data/` then holds `AADBinfo.mat` and `datasetImages_warp256/`, and is the only
directory the code reads from.

Python 3.12 rather than whatever `python3` points at: PyTorch wheels lag the
newest interpreter release by several months.

No GPU is required. Extraction picks the best available device -- CUDA, then
Apple MPS, then CPU -- and the regression head trains on cached features in
seconds on any machine. Extraction takes about two minutes per backbone on
Apple silicon and closer to twenty on CPU.

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

### Scoring new photographs

Training with `--save` keeps the head, which `src.score` then applies to any
image -- a file, several files, or a directory to rank:

```bash
python -m src.train --features clip --save model.pt
python -m src.score --model model.pt --images photo.jpg
python -m src.score --model model.pt --images ~/Pictures/album --top 10
```

```
sunset.jpg     0.867  top   1.4%  ####################
portrait.jpg   0.612  top  32.8%  #############
snapshot.jpg   0.247  top  97.8%
```

The raw score is on AADB's 0-1 scale. It is hard to read alone, because the
label distribution is peaked and most photographs land between 0.4 and 0.7, so
each image is also placed against the model's predictions on the held-out test
set. Those predictions are stored in the checkpoint for exactly this purpose:
the useful question is not whether an image is a 0.58 but whether it beats most
photographs.

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

Ten seeds per configuration. A seed fixes the train/validation split, so every
configuration sees the same ten splits and the columns can be compared row
against row. Reporting a single seed here would be misleading: the spread
across seeds is about half a point of SRCC, which is the same size as the
differences being argued about.

| Features | Loss | SRCC (mean ± sd) | vs CLIP | p |
|---|---|---|---|---|
| CLIP ViT-B/32 | mse | **0.7645 ± 0.0034** | — | — |
| CLIP ViT-B/32 | mse+rank | 0.7613 ± 0.0063 | −0.0033 | 0.17 |
| CLIP + ResNet-50 | mse | 0.7575 ± 0.0057 | −0.0070 | 0.008 |
| CLIP + ResNet-50 | mse+rank | 0.7516 ± 0.0051 | −0.0130 | <0.001 |
| ResNet-50 | mse | 0.6139 ± 0.0080 | −0.1507 | <0.001 |

`p` is a paired t-test over the ten shared seeds; a Wilcoxon signed-rank test
agrees on every row.

For reference, the original AADB paper reports SRCC 0.678 on this test set. The
comparison is not architecture against architecture: CLIP brings large-scale
pretraining that was not available in 2016, and the gap should be read as what
that pretraining buys rather than as a better head design.

**Fusion hurts, slightly but consistently.** Adding ResNet-50 costs 0.7 SRCC,
and CLIP wins on 8 of the 10 shared seeds. ResNet-50 alone reaches 0.614, so its
features are not uninformative -- they are largely redundant with what CLIP
already encodes, and the extra width buys overfitting instead of signal. The
symptom is visible in the training curves: fused models peak within the first
two epochs and decline from there, against roughly epoch 10 for CLIP alone.

**The ranking loss does nothing measurable.** The 0.3-point deficit does not
separate from seed noise (p = 0.17, CLIP ahead on 6 of 10 seeds). The premise
was that optimising MSE while measuring rank correlation leaves ordering on the
table; at this scale it evidently does not, and a 512-dimensional CLIP embedding
is separable enough that MSE already recovers most of the available ordering.
The two effects do stack -- fusion plus ranking loss is the worst of the four
CLIP configurations, and that gap is unambiguous.

The honest summary is that the simplest configuration wins and the two ideas
this study set out to test were not worth their complexity.

## Reference

Kong, S., Shen, X., Lin, Z., Mech, R., Fowlkes, C. *Photo Aesthetics Ranking
Network with Attributes and Content Adaptation.* ECCV 2016.
