---
date: '2026-11-06T00:00:00-05:00'
draft: false
title: 'The 98% Was an Illusion: Five Ways to Classify a Polymer Curve'
description: "My earlier CNN looked 98% accurate because the same material appeared in both training and test. Under material-grouped cross-validation on a cleaned 599-curve dataset, five very different models all land near 93–95%."
author: 'Jorgen Bergstrom'
tags: ['ML', 'CNN', 'Gradient Boosting', 'Autoencoders', 'Polymer Mechanics', 'Python']
categories: ['Polymer Mechanics']
---

## Introduction

In an [earlier article](/posts/elastomer_thermoplastic_classifier/) I trained a small CNN to decide whether an axis-free stress-strain image came from a rubber or a thermoplastic. The headline number was about **98% validation accuracy**. I also added a caveat: because many materials had several curves — different strain rates, loading directions, or cyclic tests — curves from the same material could end up in both the training and validation sets. That kind of leakage makes a model look better than it really is.

This article is the follow-up. I cleaned the dataset, grouped the cross-validation folds by material so that no material ever appears in both training and validation, and compared **five different ways** of turning a stress-strain curve into a prediction:

1. an image CNN on rendered stress-strain plots,
2. a 1D CNN on the raw resampled sequence,
3. a supervised autoencoder on the raw curve,
4. an MLP on 28 hand-crafted polymer-mechanics features,
5. gradient-boosted trees on those same hand-crafted features.

The short version: the image CNN falls from ~98% to **94.8%**, and all five methods land within about **2.4 percentage points** of each other. The biggest lesson is not that one model wins. It is that **the evaluation protocol matters more than the architecture**, and that a well-designed physics feature set can tie a CNN.

## The problem

Rubbers (elastomers) and thermoplastics respond to load in characteristically different ways:

- A **rubber** has a low initial modulus, then a long, relatively flat rubbery plateau, followed by a pronounced upturn as the polymer chains approach their extensibility limit. The curve is often J-shaped or S-shaped.
- A **thermoplastic** typically has a stiffer initial region, a yield point, and then a post-yield plateau or softening as it draws.

The distinction is real, but it is not always clean. Thermoplastic elastomers, copolyesters, and thermoplastic vulcanizates sit between the two categories. Add compression tests, cyclic loading, and relaxation tests, and the classification problem becomes a good test of whether a model is learning *shape* or memorizing *materials*.

## The dataset

The cleaned dataset contains **599 curves**:

- **448 thermoplastics** and **151 rubbers**
- **54 material labels**, which collapse to **35 physical material groups** after merging near-duplicate names
- data drawn from **14 references**
- mostly uniaxial tension and compression, plus a small number of biaxial, planar, triaxial, relaxation, and cyclic tests

The original image set had 607 curves. The cleanup pass:

- fixed stress-sign errors in several PEEK, polycarbonate, Hytrel, and polypropylene files,
- reclassified Copolyester and Santoprene from thermoplastic to rubber, raising the rubber count from about 118 to 151,
- truncated unreasonable strain ranges in LLDPE and Delrin,
- removed a few anomalous files, including one Nylon 66 curve.

The class balance matters for reading the results. A trivial model that always predicts "thermoplastic" would score **74.8%** on this dataset. The models below are clearly doing better than that, but accuracy alone is not the whole story.

## The protocol

For a fair comparison, every method sees the same data and the same folds.

**Normalization.** Stress is scaled per curve to $[-1, 1]$ by its maximum absolute value. Strain is not scaled per curve; it is mapped through one fixed global affine transform (strain divided by 8) so that the relative strain extent of each curve is preserved. The reason is physical: rubber and thermoplastic curves have similar stress and strain ranges, so the class signal lives in the **shape** of the curve, not in its absolute magnitude. Per-curve stress scaling removes magnitude while keeping the shape, including the sign of the stress.

**Grouped cross-validation.** The data is split into five folds using `GroupKFold`, grouped by material. All curves from one material go into the same fold. No material ever appears in both training and validation. This answers the question that actually matters for a new material: *can the model generalize to a polymer it has never seen?*

**Equal treatment.** All five methods use the same folds, the same normalization, and the same rule for reporting accuracy: after training, keep the model state with the best validation loss (or best validation accuracy for the 1D CNN) and report its accuracy. That rule is identical across methods, so the comparison is apples-to-apples. It is also slightly optimistic, because the validation fold is used to pick the epoch; I will come back to that in the limitations.

## Five ways to see a curve

### 1. Image CNN

Each curve is rendered as a 400×400 grayscale image with no axes, ticks, labels, or titles. The x-axis is engineering strain on a fixed common axis, and the y-axis is per-curve normalized stress. The image is resized to 180×180 and passed through three convolutional blocks (32 → 64 → 64 channels), each with a 3×3 convolution, ReLU, batch normalization, and max pooling, followed by a fully connected head and a single thermoplastic logit.

This is the same family of model as the original article, but now evaluated on the cleaned 599-curve set with material-grouped folds.

### 2. 1D CNN on the raw sequence

Instead of rendering an image, this model resamples each curve to 200 points and feeds a `(2, 200)` tensor: one channel for strain and one for normalized stress. The 1D CNN uses the same depth and channel structure as the image CNN, with 1D convolutions over the strain axis. It is the closest numerical analogue of the image model: it learns local shape features — a yield knee, a strain-stiffening upturn — directly from the numbers.

### 3. Supervised autoencoder

The autoencoder compresses the 400-dimensional curve (200 strain points plus 200 normalized stress points) through a bottleneck of **16 latent numbers**, then reconstructs the input. A small classifier head is attached to the bottleneck, and the network is trained with a combined loss:

$$
\mathcal{L} = \mathrm{MSE}(\text{reconstruction}) + 0.3 \cdot \mathrm{BCE}(\text{class prediction}).
$$

The reconstruction term keeps the latent space faithful to the curve shape; the classification term pushes the two classes apart. A small MLP is then trained on the 16 latent features. This is the "learned representation" approach: no hand-engineered features, but a much narrower information channel than the raw sequence.

### 4. Hand-crafted features + MLP

Here the representation is designed by hand, using polymer mechanics. Each curve is reduced to **28 features**, including:

- initial modulus (a Young's modulus proxy),
- peak stress and strain at peak,
- the ratio of mid-curve stress to peak stress, which captures yield behavior,
- curvature at low and high strain, which separates a J-shaped rubber curve from a yield-and-plateau thermoplastic curve,
- secant moduli and stress at fixed strain fractions,
- area under the curve (toughness),
- flags for test type and true-versus-engineering strain.

Those features are standardized and fed to a small MLP. The point is not that the features are perfect; the point is that they encode the physics explicitly, so the model has far less to learn from 599 examples.

### 5. Gradient boosting on the features

The same 28 features are also given to XGBoost. Trees handle features on different scales natively, capture nonlinear interactions such as "high modulus **and** low yield ratio implies rubber," and often outperform small neural networks on tabular data.

For context, I also tested boosting on the raw 400-dimensional sequence. That version is noticeably worse, which turns out to be one of the most instructive results in the whole study.

## Results

All numbers below are validation accuracy on the identical material-grouped five-fold split.

| Fold | Image CNN | XGBoost + features | 1D CNN | Autoencoder | Features + MLP |
|------|-----------|--------------------|--------|-------------|----------------|
| 1 | 0.950 | 1.000 | 1.000 | 0.950 | 0.983 |
| 2 | 0.967 | 0.983 | 0.967 | 0.925 | 0.942 |
| 3 | 0.875 | 0.817 | 0.842 | 0.883 | 0.808 |
| 4 | 0.983 | 1.000 | 0.942 | 0.933 | 1.000 |
| 5 | 0.966 | 0.933 | 0.899 | 0.941 | 0.891 |
| **Mean ± std** | **0.948 ± 0.038** | **0.947 ± 0.070** | **0.930 ± 0.055** | **0.927 ± 0.023** | **0.925 ± 0.069** |

A few things stand out.

**The image CNN and XGBoost on physics features are effectively tied.** The CNN is ahead by a fraction of a percentage point, which is far smaller than the fold-to-fold variation. A convolutional network looking at pixels and a tree ensemble looking at 28 hand-computed numbers extract essentially the same amount of information from this dataset.

**All five methods are within 2.4 points.** The spread from best to worst is smaller than the spread between folds for most methods. Fold 3 is hard for everyone (0.81–0.88), and fold 4 is easy for everyone (0.93–1.00). That pattern is a strong hint that the difficulty is in the materials in each fold, not in the model class.

**The autoencoder has the smallest variance.** Its ±2.3% is noticeably tighter than the others, even though its mean is mid-pack. The 1D CNN and XGBoost are close behind the leaders but vary more across folds.

**The 1D CNN is competitive but not the winner.** It lands at 93.0%, between the two leaders and the two 92.5–92.7% methods. That is a good result for a model that sees only the raw sequence, but it does not beat the image representation.

## The representation effect is bigger than the model effect

The most striking result comes from comparing XGBoost on two different encodings of the same curves:

| Representation | XGBoost | LightGBM |
|----------------|---------|----------|
| Raw 400-dim sequence | 89.0% ± 9.0% | 87.1% ± 12.1% |
| 28 hand-crafted features | 94.3% ± 6.6% | 95.0% ± 6.9% |

Same algorithm, same curves, same folds — a **5 to 8 point** difference depending only on how the curve is represented. Trees split on one feature at a time, so a raw sequence of 200 interleaved `(strain, stress)` points is a poor input: most individual coordinates are weak, noisy, or redundant. Hand-crafted features compress the curve into a handful of meaningful, decorrelated numbers, and the trees suddenly have something to work with.

This is the central lesson of the study. Representation matters as much as — sometimes more than — the choice of model. The image CNN and the 1D CNN work because convolutions can learn useful shape filters from raw structure. The feature MLP and XGBoost work because the physics has already been decoded by hand. Both routes are valid; they just put the intelligence in different places.

## Why the 98% was an illusion

The original ~98% came from a single random 75/25 split. With multiple curves per material, that split scattered the same material's strain rates and test directions across training and validation. The model could recognize a material it had already seen, rather than generalize to a new one.

Under the grouped protocol, the image CNN scores **94.8%**. That is still a good result, but it is not 98%. The gap is the leakage. The exact size of the drop also reflects the dataset cleanup — the cleaned set has different labels and a different class balance — but the protocol is the main reason.

This is a useful general lesson. When a validation split is random over *samples* but the samples are not independent — multiple curves per material, multiple images per patient, multiple frames per video — the validation score measures memorization as much as generalization. Grouping by the right unit is often more important than adding capacity to the model.

## Limitations and what I would do next

This is a small, observational dataset, and the comparison has several caveats.

**Only 35 material groups.** Five folds means roughly seven groups per validation fold. A single unusual material can swing a fold by several points, which is why the per-fold spread is large and why the mean ± std is more informative than any single fold.

**Accuracy hides class-specific behavior.** The dataset is 75% thermoplastic. A model can score in the low 90s while still missing a meaningful fraction of rubbers. The per-curve review shows that the image CNN misclassified about 31 of 599 curves, often compression curves with negative stress or borderline materials such as Arnitel, Hytrel, Santoprene, Copolyester, PTFE, and HDPE/UHMWPE. Reporting per-class recall, a confusion matrix, and calibrated probabilities would be more useful than a single accuracy number.

**The epoch-selection rule is mildly optimistic.** Every method reports the model state that did best on the validation fold. The rule is identical for all five methods, so the ranking is fair, but the absolute numbers are a little higher than a strict nested cross-validation would give.

**The labels are not always crisp.** Copolyester and Santoprene were reclassified during cleanup. Thermoplastic elastomers are, by design, between the two categories. The classifier is learning a useful engineering distinction, not a fundamental law of nature.

**Test type may be a confound.** Most curves are uniaxial, but the mix of tension, compression, cyclic, and relaxation tests is not perfectly balanced across classes. A model can pick up on test-type cues. A stricter experiment would stratify or balance the test types, or evaluate on a held-out test type.

The obvious next steps are per-class metrics and calibration, a larger and more balanced material set, and perhaps a split that holds out entire test types rather than only materials.

## Reproducing the comparison

The full comparison is available in the public GitHub repository [Polymer_Curve_Classifier](https://github.com/Jorgen-Bergstrom/Polymer_Curve_Classifier). It contains the code and documentation only; the experimental stress-strain data is not included. The repository README documents the expected data layout.

```bash
git clone https://github.com/Jorgen-Bergstrom/Polymer_Curve_Classifier.git
cd Polymer_Curve_Classifier
uv sync
uv run python polymer_classifier/run_comparison.py --method all
```

By default the code looks for `data/` at the repository root. Set `POLYMER_DATA_DIR` if your data lives elsewhere.

Individual methods can be run separately:

```bash
uv run python polymer_classifier/run_comparison.py --method cnn
uv run python polymer_classifier/run_comparison.py --method 1dcnn
uv run python polymer_classifier/run_comparison.py --method autoencoder
uv run python polymer_classifier/run_comparison.py --method features
uv run python polymer_classifier/run_comparison.py --method gbdt
```

The per-method README files contain the model details and the standalone results; `comparison/README.md` contains the matched five-way protocol and the table above.

## Takeaway

The headline is not "CNNs are overrated" or "XGBoost wins." It is that **the evaluation protocol is the biggest lever**. Once the same material can no longer appear on both sides of the split, five very different models converge to roughly the same accuracy. The curve-shape signal is apparently easy to extract; the hard parts are cleaning the data, defining an honest split, and deciding what "accuracy" actually means for an imbalanced, observational dataset.

If you have raw curves or images and no domain features, a CNN is a good choice. If you have a handful of physically meaningful descriptors, gradient boosting on those features is a strong, interpretable baseline that can tie a CNN. And if you have neither, the autoencoder is a reasonable middle ground — with the smallest fold-to-fold variance in this study.

The 98% was an illusion. The 94.8% is a more honest number, and a more interesting one.
