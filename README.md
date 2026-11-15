# latent-smile

Controllable smile editing on 64×64 CelebA faces with two VAE approaches, compared under the same protocol.

## Research question

Does explicitly supervising a smile latent give better controllability and disentanglement than discovering a smile direction in an unsupervised latent space?

## Models

All three share the same convolutional backbone and the same 5-level editing interface.

- **B (`type: vae`, `baseline`)**: a plain VAE trained with no labels. After training, a smile direction `v` is fitted on validation labels (mean difference, or a linear probe). Edits set the projection onto `v` to an absolute target.
- **A-noadv (`type: supervised`, `supervised_noadv`)**: the latent is split as `z = [z_content, z_smile]`, where `z_smile` is the last dimension. `z_smile` has a class-conditional prior N(+m, 1) for smiling images and N(−m, 1) otherwise. No adversary.
- **A (`type: supervised`, `supervised`)**: as A-noadv, plus an adversary behind a gradient-reversal layer that tries to predict the smile label from `z_content`, so the content code is pushed to carry no smile information.

Latent-size (32, 128) and β-VAE ablations are provided for the plain and supervised backbones.

## Setup

Python 3.12 with [uv](https://docs.astral.sh/uv/). The commands are from the header of `requirements.txt`:

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
uv pip install --no-deps facenet-pytorch==2.6.0
```

`facenet-pytorch` is installed without its dependencies because it pins an old torch.

The commands below use `.venv/bin/python`. Activating the environment (`source .venv/bin/activate`) works equally well.

Devices are picked automatically: MPS if available, then CUDA, then CPU. Set `LATENT_SMILE_DEVICE` to override.

## Data

### CelebA

Preprocessing takes each aligned 178×218 image, centre-crops it to 148×148, and resizes to S×S with Lanczos. The prepared dataset is written to `data/` (git-ignored).

```bash
.venv/bin/python scripts/prepare_celeba.py --download --raw data/raw/celeba --out data/celeba_64 --size 64
```

The download source (Google Drive via torchvision) is often quota-limited. If the download fails, the script prints manual instructions: place the CelebA files in `data/raw/celeba` and re-run without `--download`.

### Synthetic data (smoke test)

To check the pipeline without CelebA, generate a synthetic dataset in the same format and train on it with `--data-root`:

```bash
.venv/bin/python scripts/make_synthetic.py ...   # see --help for options and output directory
.venv/bin/python -m src.train --config configs/baseline.yaml --data-root <synthetic dir> --subset 200 --epochs 2
```

The synthetic faces are simple drawings, so they check the code paths only. They say nothing about the research question.

### Splits

The official CelebA partition is used. VAEs train on `train`. Validation images are split in half: `val_fit` fits smile directions, and `val_select` selects checkpoints and calibrates levels. `test` is read only by `src.evaluate`.

## Pipeline

Run in this order. `scripts/run_all.sh` runs all of it.

1. **Evaluator VAE**: `python -m src.train --config configs/evaluator_vae.yaml --seed 100`. A plain VAE whose reconstructions are used only to train the evaluators.
2. **Evaluators**: `python -m src.evaluators --data-root data/celeba_64 --recon-run runs/evaluator_vae/seed100 --out runs/evaluator`. Trains the attribute evaluator and validates it on held-out reconstructions (the identity embedder is pretrained and not retrained); writes `runs/evaluator/attr.pt` and `results/metrics/evaluator_validation.json`.
3. **Main models**: `baseline`, `supervised_noadv` and `supervised`, for seeds 0, 1 and 2. Example: `python -m src.train --config configs/baseline.yaml --seed 0`.
4. **Discovery for each B run**: `python -m src.discover_smile --run runs/baseline/seed0 --evaluator runs/evaluator/attr.pt`. Writes `runs/baseline/seed0/directions.pt`, traversal grids and per-coordinate responses.
5. **Ablations** (seed 0 only): `beta_vae`, `baseline_z32`, `baseline_z128`, `supervised_z32`, `supervised_z128`. Discovery is run for the three plain VAEs.
6. **Evaluation**: `python -m src.evaluate --runs runs/baseline/seed0 runs/supervised/seed0 ... --evaluator runs/evaluator/attr.pt --data-root data/celeba_64 --out results`.
7. **App**: `python app.py --supervised runs/supervised/seed0 --discovered runs/baseline/seed0 [--direction probe] [--share]`. Loads checkpoints only; it does not train.

Quick run of the whole pipeline on a subset, with fewer epochs:

```bash
SUBSET=5000 EPOCHS=2 scripts/run_all.sh
```

Other overrides: `SEEDS="0 1 2"`, `SKIP_ABLATIONS=1`, `DATA=...`, `PY=...`. Training commands use `--resume`, so the script can be restarted.

### Demo models at 128×128 (optional)

For the public demo, train one 128×128 model each of A and B with `configs/demo128_supervised.yaml` and `configs/demo128_baseline.yaml` (data root `data/celeba_128`). These are for sanity-checking reconstructions and edit grids only. No metrics are reported for them.

## Methodology

The comparison is designed so that the two approaches are measured on equal terms.

- **Absolute smile levels for both.** B sets the projection onto `v` to absolute targets between the 5th and 95th percentile of train-split projections. A sets `z_smile` to five values spanning its two class-prior means. Neither approach moves an already-smiling face by a relative amount, which would confound the comparison.
- **Matched-strength trade-off curve.** The headline result is a curve of smile-score gain against identity loss and against attribute leakage, across about nine edit magnitudes per model. Per-level comparisons are confounded by how far each model moves, so the curve is used instead.
- **Edits measured against the reconstruction as well as the original.** Identity and leakage are reported relative to the original image and to the unedited reconstruction. The second reference separates the effect of the edit from the blur of the 64×64 VAE. Reference points from CelebA identity labels (same person smiling vs not, and different people) give the numbers a scale.
- **Evaluators validated on reconstructions.** The attribute evaluator is trained on real images plus reconstructions from a separate VAE, and its accuracy on held-out reconstructions is reported before any metric uses it. The identity embedder's same/different-identity AUC on reconstructions is reported too. Metrics that the validation shows to be unreliable are flagged.
- **Smile-entailed attributes separated from leakage.** Leakage is scored on unrelated attributes only (eyeglasses, sex, age, hair colours, hats, bangs, and similar). Attributes that change with a smile (High_Cheekbones, Mouth_Slightly_Open, Narrow_Eyes, Rosy_Cheeks) are reported separately.
- **Label efficiency.** B's directions are fitted with 100, 1k, 10k and all validation labels. A uses about 160k labels, so this checks how much supervision B actually needs.
- **Split discipline.** `train` for VAE training; `val_fit` for B's directions; `val_select` for A's calibration, checkpoint selection and the evaluator checks; `test` only in `src.evaluate`.
- **Fixed likelihood.** The decoder is Gaussian with a fixed σ stated in the config. The loss is the per-image summed squared error divided by 2σ², plus β times the KL. β is therefore defined relative to that likelihood.
- **Monotonicity reported in three ways.** Fraction of images whose scores strictly increase across the five levels, per-image Spearman correlation between level and score, and the level-1-to-level-5 score range. An all-five strict check is brittle under evaluator noise.

## Repository layout

```
configs/          YAML configs (schema in src/config.py)
src/
  data.py         prepared-dataset reader, synthetic data generator
  models/         shared backbone, VAE, SupervisedVAE
  losses.py       VAE and supervised losses
  train.py        training loop, checkpoints, logs and curves
  checkpoint.py   save and load
  inference.py    SmileEditor: five-level edits for both approaches
  discover_smile.py  smile directions for B, calibration, traversals
  evaluators.py   attribute evaluator and identity embedder
  metrics.py      metric definitions
  evaluate.py     test-split evaluation and aggregated tables
  face_align.py   face detection and alignment for the app
scripts/
  prepare_celeba.py  CelebA preprocessing
  make_synthetic.py  synthetic dataset for smoke tests
  run_all.sh         full pipeline
tests/            offline CPU tests on the synthetic dataset
app.py            Gradio demo
reports/report.md research report skeleton
data/             prepared datasets (git-ignored)
runs/             checkpoints and per-run outputs (git-ignored)
results/          evaluation outputs (metrics, figures, image grids)
```

## Tests

```bash
.venv/bin/python -m pytest
```

The tests run offline on CPU, using the synthetic dataset. They need no CelebA, no network access and no pretrained-weight downloads.

## Limitations

- **Binary labels.** Smiling is a yes/no label, so the intermediate levels are interpolations between the two class-prior means (A) or between the percentile targets (B). They are not separately supervised intensities.
- **Low resolution.** 64×64 VAE outputs are blurry. Smile cues are a few pixels, and the blur affects both the evaluators and the edits. Reconstruction baselines are reported to separate this effect.
- **CelebA bias.** CelebA is skewed in age, sex, and presentation, and its smile labels are per-image judgements by annotators. Results describe this dataset and this labelling, not smiles in general.
- **Predictive is not causal.** A direction that predicts the smile label is not necessarily the direction that causes the smile in the decoder. The decoder-sensitivity check and the evaluator responses are evidence about the decoder, not proof.
- **Evaluator dependence.** Metrics use learned evaluators and an identity embedder that was pretrained on other data. Their validation numbers bound how far the metrics can be trusted.
- **Small seed counts.** Three seeds for the main models and one seed for ablations. Variation across seeds is reported, but it is not a large sample.

## Results

Results are produced by `python -m src.evaluate` and written to `results/`:

- `results/metrics/` per-run JSON, `summary.csv` and `summary.md`, and `evaluator_validation.json`
- `results/figures/` including the trade-off figure
- `results/reconstructions/` and `results/smile_edits/` image grids

No results are included in this repository yet. The experiments have not been run. Once they have, the tables and figures will be in `results/`, and `reports/report.md` will be filled in from them.
