# Vectorized Quantum-Behaved Delta PSO for Gradient-Free MLP Training

Reproduction code for the paper:

> S. Chancay and I. Pineda, *"Vectorized Quantum-Behaved Delta PSO for
> Gradient-Free MLP Training"*, IEEE Access.

This folder is a **frozen archive**: it reproduces the paper's experiments as published. The statistical report cited in the paper is
[`src/results/stats_report.md`](./src/results/stats_report.md), and the
per-fold raw results are the JSON files in [`src/results/`](./src/results/).

## Contents

```
vectorized-qdpso-mlp/
├── requirements.txt           # pinned dependencies
└── src/
    ├── main_qpsoO.py          # QDPSOo — vectorized QDPSO (proposed method)
    ├── main_qpso.py           # QDPSO  — original per-particle implementation (timing baseline)
    ├── main_pso.py            # PSO / PSO_bound baselines (torch-pso)
    ├── main_adam.py           # Adam baseline (100 and 1000 epochs)
    ├── analyze_stats.py       # statistical tests → results/stats_report.md
    ├── run_utils.py           # shared CLI, seeding, data loading, JSON dumps
    ├── load_model.py          # load a saved .pth model
    ├── run_benchmark_fast.sh  # full benchmark batch (4 datasets × 5 seeds × 5 optimizers)
    ├── run_mcw.sh             # full MCW batch (ORG + isomap/pca/mds × {42,21,14,7} × 5 seeds)
    ├── run_g_sweep.sh         # sensitivity sweep of g ∈ {0.6 … 1.5}
    ├── run_qpso_slow.sh       # per-particle QDPSO timing runs
    ├── one_swarm/             # QDPSOo/QDPSO optimizers + MLP model
    ├── adam/                  # Adam training loop
    ├── data/                  # dataset loaders (sklearn benchmarks + MCW images)
    ├── metrics/               # metrics + loss/ROC plotting
    └── results/               # per-fold raw results (JSON) + stats_report.md
```

Generated artifacts (`models/`, `output/`, `metrics/graphics/`) are not
tracked here; they are recreated by the runs below.

## 1. Environment setup

Requirements: Linux, [Miniforge/Conda](https://github.com/conda-forge/miniforge)
(or any Python 3.12), optionally an NVIDIA GPU. The code auto-selects
`cuda` when available and falls back to CPU otherwise.

```bash
conda create -n vectorized-qdpso python=3.12 -y
conda activate vectorized-qdpso
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__, '| cuda:', torch.cuda.is_available())"
# 2.5.1+cu124 | cuda: True   (cuda: False is fine — runs on CPU)
```

## 2. Datasets

**Benchmarks** (`circle`, `iris`, `wine`, `breast_cancer`): loaded
automatically through scikit-learn. Nothing to download.

**MCW — Multi-class Weather Dataset** (image classification, 4 classes):
download it from Mendeley Data
([doi:10.17632/4drtyfjtfy.1](https://data.mendeley.com/datasets/4drtyfjtfy/1))
and place the images under `src/data/img/mcw/`, one folder per class:

```
src/data/img/mcw/
├── cloudy/
├── rain/
├── shine/
└── sunrise/
```

The first MCW run extracts the 84 handcrafted features (HSV histogram 4^3 +
Haralick 13 + Hu moments 7) from the ~1,120 images (≈1–2 min) and caches
them in `src/data/img/mcw_features_150x150_b4.npz`; later runs load the
cache instantly.

> **Note:** If the images fail to load or you encounter any errors, check the MCW dataset images and remove them (there are about 2–4 images).

## 3. Quick start (single runs)

All commands run from `src/`:

```bash
cd src
```

QDPSOo (proposed method) on iris — paper configuration
(g = 1.13, 20 particles, 4 folds, 100 epochs), ~20 s on GPU:

```bash
python main_qpsoO.py --dataset iris --seed 0
# INFO - Mean accuracy on test dataset: 0.9583 - std: 0.0546
# INFO - Per-fold results saved at: ./results/iris_QPSOo_4_s0.json
```

Baselines on the same dataset/seed:

```bash
python main_pso.py  --dataset iris --variant PSO_bound --seed 0
python main_adam.py --dataset iris --epochs 100  --seed 0 --tag ep100
python main_adam.py --dataset iris --epochs 1000 --seed 0 --tag ep1000
```

MCW with dimensionality reduction (84 → 7 features, PCA; the paper uses
1000 epochs by default for MCW):

```bash
python main_qpsoO.py --dataset mcw --reduction pca --components 7 --seed 0
```

### CLI options (shared by all `main_*.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `iris` | `circle`, `iris`, `wine`, `breast_cancer`, `mcw` |
| `--epochs` | 100 (benchmarks) / 1000 (mcw) | training iterations |
| `--particles` | 20 | swarm size (`n` particles) |
| `--g` | 1.13 | `contraction–expansion coefficient` (QDPSO/QDPSOo) |
| `--folds` | 4 | K-fold cross-validation folds |
| `--seed` | 0 | algorithm seed (swarm/weight init) |
| `--split-seed` | 100 | train/test split and KFold seed (paper: 100) |
| `--reduction` | – | `isomap`, `pca`, `mds` (MCW only) |
| `--components` | – | reduced dimensionality (MCW only) |
| `--select-by` | `val` | best-fold selection criterion (`test` = legacy behavior) |
| `--tag` | – | suffix to keep output files unique |

Each run writes per-fold results to `results/<dataset>_<optimizer>_<suffix>.json`,
the best model to `models/`, per-epoch logs to `output/`, and loss/ROC figures
to `metrics/graphics/`.

> **Note:** running with the default seeds regenerates the same
> `results/*.json` filenames that ship with this archive (the paper's raw
> results).

## 4. Full reproduction of the paper

```bash
cd src
./run_benchmark_fast.sh   # 4 benchmarks × 5 seeds × {QDPSOo, PSO, PSO_bound, Adam@100, Adam@1000}
./run_mcw.sh              # MCW: ORG(84) + {isomap,pca,mds} × {42,21,14,7} × 5 seeds
./run_g_sweep.sh          # g ∈ {0.6, 0.8, 0.96, 1.13, 1.3, 1.5} sensitivity (R2.6/R3.5)
./run_qpso_slow.sh        # per-particle QDPSO timing comparison (~5 h, 1 seed)
```

Then regenerate the statistical report (mean ± CI95, Friedman, pairwise
Wilcoxon with Holm correction):

```bash
python analyze_stats.py   # → results/stats_report.md
```

## 5. Reproducibility

Seeds are fixed across `random`, `numpy`, and `torch` (`run_utils.set_seeds`).
In a fresh environment built exactly as in step 1 (GPU: NVIDIA GTX 1050 Ti,
CUDA 12.4), single runs of `main_qpsoO.py`, `main_pso.py`, and `main_adam.py`
on iris (seed 0) reproduced the shipped per-fold accuracies in
`results/*.json` **bit-exactly**. CPU-only execution or different
GPU/CUDA/PyTorch versions may produce slightly different (statistically
equivalent) values.

