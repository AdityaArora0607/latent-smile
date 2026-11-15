# Supervised versus discovered smile directions in a VAE latent space

## Abstract

This report compares two ways of getting controllable smile intensity out of a VAE trained on 64×64 CelebA faces. In approach B, a plain VAE is trained without labels and a smile direction is found afterwards, from validation labels, by a mean difference or a linear probe. In approach A, the latent is split so that one dimension carries a class-conditional smile prior, and an adversary keeps the remaining dimensions free of smile information. Both approaches are edited to the same five absolute levels and are evaluated on the same held-out test split with the same evaluators. The results section is not yet filled in: the experiments have not been run.

## Research question and hypotheses

Does explicitly supervising a smile latent give better controllability and disentanglement than discovering a smile direction in an unsupervised latent space?

Hypotheses, stated before any results are measured:

- H1 (controllability). For a given identity loss, A produces a larger smile-score gain than B across the matched-strength trade-off curve.
- H2 (disentanglement). Under A, smile is less predictable from the content code than under B, once the smile direction is removed from B's latent.
- H3 (label efficiency). B reaches its best directions with far fewer labels than A's supervision budget, so the labelling cost difference may be smaller than it appears.

Each hypothesis is tested only by the metrics named in the evaluation protocol below.

## Data and preprocessing

CelebA aligned images (178×218) are centre-cropped to 148×148 (left 15, top 35) and resized to 64×64 with Lanczos resampling. The prepared dataset holds uint8 RGB images, 40 binary attribute labels, identity labels and the official train / validation / test partition. Pixels are scaled to [-1, 1] for the models. The only training augmentation is random horizontal flip.

VAEs train on the official train split. Validation images are split in half: `val_fit` is used to fit smile directions for B, and `val_select` is used for checkpoint selection. The test split is read only by the evaluation step.

Two external evaluators support the metrics: an attribute evaluator (a small CNN trained on real images plus reconstructions from a separate evaluator-only VAE) and an identity embedder (a pretrained InceptionResNet). Their validation is reported before any metric that depends on them.

## Models

All models share one convolutional backbone. The encoder has four convolutional blocks (32, 64, 128 and 256 channels at 64×64) with SiLU activations and fully connected heads for the posterior mean and log-variance. The decoder mirrors the encoder and ends in a tanh. The latent size is 64 by default. Decoding is Gaussian with a fixed σ. The loss is the per-image summed squared error divided by 2σ², plus β times the KL divergence.

### Approach B: unsupervised VAE with discovered directions

B is a plain VAE trained with no access to labels. After training, a smile direction `v` is found on `val_fit` in three ways:

- mean difference between the posterior means of smiling and non-smiling images;
- an L2-regularised logistic probe, unit-normalised;
- the best single latent coordinate, as a one-hot direction with the sign corrected.

All directions are oriented so that a larger projection means more smiling. Edits set the projection onto `v` to an absolute target, `z := z + (target - z·v) v`, where the five targets are evenly spaced between the 5th and 95th percentiles of train-split projections. Label efficiency is measured by refitting with 100, 1k, 10k and all `val_fit` labels.

### Approach A: supervised latent with class-conditional prior and adversary

The last latent dimension is `z_smile`. The other D-1 dimensions form `z_content`.

- Prior. `z_smile` has prior N(+m, 1) for smiling images and N(−m, 1) otherwise, with m = 2 by default. The KL term for `z_smile` uses the prior matching the image's label. `z_content` has a standard normal prior.
- Adversary. A small MLP predicts the smile label from `z_content` through a gradient-reversal layer. The adversary weight ramps up over the first epochs. Setting the weight to 0 gives A-noadv, which is the ablation that separates the effect of the prior from the effect of the adversary.
- Editing. The five levels set `z_smile` to five values spanning −m to +m.
- Diagnostic. Each epoch logs the mean pixel change when `z_smile` is flipped between +m and −m, to check that the decoder uses the dimension.

## Evaluation protocol

All evaluation is on the test split, by `src.evaluate`, for every run and seed.

- Reconstruction: MSE, MAE and SSIM of the unedited reconstruction.
- Smile control: smile-evaluator score at each of the five levels, the fraction of images whose scores strictly increase, per-image Spearman correlation between level and score, and the level-1-to-level-5 score range.
- Identity preservation: identity-embedder similarity between the edited image and the original, and between the edited image and the unedited reconstruction. Reference points come from CelebA identity labels: the same person smiling and not smiling, and different people.
- Attribute leakage: change in unrelated attributes, reported separately from smile-entailed attributes.
- Matched-strength trade-off: smile-score gain against identity loss and against leakage, swept over about nine edit magnitudes per model. This is the headline comparison.
- Disentanglement probes: how well smile is predicted from `z_content` (for B, from the latent with `v` projected out), using a linear and an MLP probe.
- Label efficiency: B's direction quality at 100, 1k, 10k and all `val_fit` labels.
- Ablations: β-VAE and latent sizes 32 and 128, one seed each.
- Failure cases: chosen by the worst value of each metric, not selected by hand.

Seeds: three each for B, A-noadv and A; one seed for the ablations. Results are reported as mean ± standard deviation over seeds.

## Results

### Reconstruction

_Not yet run._
Source: `results/metrics/summary.md` (per-run values in `results/metrics/<name>_seed<k>.json`).

### Smile control and monotonicity

_Not yet run._
Source: `results/metrics/summary.md` (per-run values in `results/metrics/<name>_seed<k>.json`).

### Identity preservation

_Not yet run._
Source: `results/metrics/summary.md` (per-run values in `results/metrics/<name>_seed<k>.json`).

### Attribute leakage

_Not yet run._
Source: `results/metrics/summary.md` (per-run values in `results/metrics/<name>_seed<k>.json`).

### Matched-strength trade-off

_Not yet run._
Source: `results/figures/tradeoff.png`.

### Disentanglement probes

_Not yet run._
Source: `results/metrics/summary.md` (per-run values in `results/metrics/<name>_seed<k>.json`).

### Label efficiency

_Not yet run._
Source: per-run values in `results/metrics/<name>_seed<k>.json`.

### β and latent-size ablations

_Not yet run._
Source: `results/metrics/summary.csv`.

### Failure cases

_Not yet run._
Source: `results/smile_edits/` and `results/reconstructions/`.

## Deviations from the original specification

Each amendment was a deliberate decision made before results were measured.

| # | Amendment | Reason |
|---|---|---|
| 1 | B sets the projection onto `v` to absolute targets (5th–95th percentile of train projections, five evenly spaced levels). A sets `z_smile` to five values spanning the two class-prior means. | The specification had A absolute and B relative (`z + αv`), which confounds the comparison for already-smiling inputs. |
| 2 | The headline result is a trade-off curve: smile-score gain against identity loss and against leakage, sweeping edit magnitude (about nine points per model). | A per-level comparison is confounded by how far each model moves. |
| 3 | Identity and leakage are measured against both the original and the unedited reconstruction. Reference points from CelebA identity labels: same person smiling vs not, and different people. | This separates the effect of the edit from VAE blur and gives the numbers a scale. |
| 4 | The attribute evaluator is trained on real images plus reconstructions from a separate evaluator-only VAE. Its accuracy on held-out reconstructions is reported before any metric is used. The identity embedder's same/different-identity AUC on reconstructions is reported too. | Evaluators trained on sharp images may fail on blurry 64×64 decodes. |
| 5 | Leakage is scored on unrelated attributes only. Smile-entailed attributes (High_Cheekbones, Mouth_Slightly_Open, Narrow_Eyes, Rosy_Cheeks) are reported separately. | Those attributes are expected to change with a smile. |
| 6 | Label-efficiency experiment for B: directions fitted with 100, 1k, 10k and all `val_fit` labels. | A uses about 160k labels; B may need far fewer. |
| 7 | The likelihood is Gaussian with a fixed σ stated in the config. The loss is per-image summed SSE/(2σ²) + β·KL, and β is defined relative to that. | "β = 1" is undefined without a stated likelihood. |
| 8 | Monotonicity is reported as the fraction strictly increasing, plus per-image Spearman(level, score) and the level-1-to-5 score range. | A strict all-five check is brittle under evaluator noise. |
| 9 | Splits: VAEs train on the official train split; B's directions and model selection use validation (split in half into fit and select); the test split is used only by `evaluate.py`. | Clean separation of fitting, selection and testing. |
| 10 | Seeds: three each for B, A-noadv and A; one seed for β-VAE and the latent-size ablations. | Compute budget. |
| 11 | Dropped: PCA/UMAP plots and human ratings. LPIPS stays optional. | Low value for the question. |
| 12 | The project layout lives at the repository root, not in a `vae-smile-editor/` subdirectory. | The repository already existed. |

## Limitations

- Smiling is a binary label. Intermediate levels are interpolations between the class-prior means (A) or the percentile targets (B), not separately supervised intensities.
- At 64×64 the VAE outputs are blurry, and smile cues occupy a few pixels. Blur affects both the evaluators and the edits.
- CelebA is biased in age, sex and presentation, and its smile labels are annotator judgements. Results describe this dataset.
- A direction that predicts smile is not necessarily the direction the decoder uses to render a smile. The decoder-sensitivity diagnostic and evaluator responses are evidence, not proof, of a causal direction.
- The metrics depend on learned evaluators. Their validation bounds how far the metrics can be trusted, and any metric flagged as unreliable is reported as such.
- Three seeds for the main models and one seed for the ablations give limited variance estimates.
- The 128×128 demo models are for qualitative sanity checks only. No metrics are reported for them.

## Reproduction

The environment, data preparation and pipeline commands are in the repository `README.md`. In short:

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
uv pip install --no-deps facenet-pytorch==2.6.0
.venv/bin/python scripts/prepare_celeba.py --download --raw data/raw/celeba --out data/celeba_64 --size 64
scripts/run_all.sh
.venv/bin/python -m pytest
```

`scripts/run_all.sh` runs the evaluator VAE, the evaluator, the three main models for each seed, discovery for each baseline run, the ablations (unless `SKIP_ABLATIONS=1`), and the final evaluation into `results/`. Seeds, subset size, epoch count and paths can be overridden with the `SEEDS`, `SUBSET`, `EPOCHS`, `DATA` and `PY` environment variables.
