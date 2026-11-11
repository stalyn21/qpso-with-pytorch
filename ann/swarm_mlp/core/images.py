"""Datasets de imágenes para la CNN del paper 2 (ver MEMORY/.../paper2-cnn-plan.md).

Protocolo (MNIST y Fashion-MNIST):
- **Test = test oficial** (10,000 imágenes), para compararse con la literatura.
- Del train oficial (60,000) salen, estratificados y con `data_seed`:
    * validación: `val_size` imágenes (selección de modelos);
    * fitness: `fitness_samples` imágenes, sobre las que los enjambres evalúan la
      función de costo (0 = todo el resto del train). Es la "train" de la corrida:
      train_acc y la L2 (n_train) se miden sobre ella.
- Píxeles en [0, 1] (división por 255). Con `input_norm="standardize"` además se
  resta la media y se divide por el desvío de cada canal, calculados SÓLO sobre las
  imágenes de fitness: validación y test no aportan nada al ajuste.

MCW (Multiclass Weather) entra antes de la corrida final; necesita fijar resolución
y no tiene test oficial.
"""
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from core.features import DEFAULT_CACHE_DIR

#: Datasets de imágenes y su clase de torchvision.
IMAGE_DATASETS = ("fmnist", "mnist")

#: Normalizaciones de la entrada (ver docstring del módulo).
INPUT_NORMS = ("none", "standardize")

_TORCHVISION = {"fmnist": "FashionMNIST", "mnist": "MNIST"}


def _load_official(name: str, train: bool):
    import torchvision.datasets as tvd

    root = DEFAULT_CACHE_DIR / "torchvision"
    root.mkdir(parents=True, exist_ok=True)
    ds = getattr(tvd, _TORCHVISION[name])(root=str(root), train=train, download=True)
    X = ds.data.numpy().astype(np.float32)[:, None, :, :] / 255.0
    y = ds.targets.numpy().astype(np.int64)
    return X, y


def prepare_image_dataset(
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    data_seed: int = 100,
    val_size: int = 5000,
    fitness_samples: int = 2000,
    input_norm: str = "none",
    norm_eps: float = 1e-6,
) -> Dict[str, Any]:
    """Splits de `name` como tensores (N, C, H, W) en `device` (ver docstring del módulo)."""
    if name not in IMAGE_DATASETS:
        raise ValueError(f"Dataset de imágenes desconocido: {name}. Opciones: {IMAGE_DATASETS}")
    if input_norm not in INPUT_NORMS:
        raise ValueError(f"input_norm desconocido: {input_norm}. Opciones: {INPUT_NORMS}")
    X_tr, y_tr = _load_official(name, train=True)
    X_te, y_te = _load_official(name, train=False)

    if not 0 < val_size < len(y_tr):
        raise ValueError(f"val_size debe estar en (0, {len(y_tr)}); se dio {val_size}")
    idx = np.arange(len(y_tr))
    idx_rest, idx_val = train_test_split(idx, test_size=val_size, stratify=y_tr,
                                         random_state=data_seed)
    if fitness_samples and fitness_samples < len(idx_rest):
        idx_fit, _ = train_test_split(idx_rest, train_size=fitness_samples,
                                      stratify=y_tr[idx_rest], random_state=data_seed + 1)
    else:
        idx_fit = idx_rest

    X_fit = X_tr[np.sort(idx_fit)]
    X_val = X_tr[np.sort(idx_val)]
    if input_norm == "standardize":
        # Estadísticas por canal, sólo de las imágenes de fitness.
        mean = X_fit.mean(axis=(0, 2, 3), keepdims=True, dtype=np.float64)
        std = X_fit.std(axis=(0, 2, 3), keepdims=True, dtype=np.float64)
        scale = 1.0 / np.maximum(std, norm_eps)
        X_fit, X_val, X_te = ((a - mean) * scale for a in (X_fit, X_val, X_te))
        stats = {"input_mean": mean.ravel().tolist(), "input_std": std.ravel().tolist()}
    else:
        stats = {}

    def f(a):
        return torch.as_tensor(a, dtype=dtype, device=device)

    def l(a):
        return torch.as_tensor(a, dtype=torch.long, device=device)

    c, h, w = X_tr.shape[1:]
    return {
        "X_train": f(X_fit), "y_train": l(y_tr[np.sort(idx_fit)]),
        "X_val": f(X_val), "y_val": l(y_tr[np.sort(idx_val)]),
        "X_test": f(X_te), "y_test": l(y_te),
        "n_features": int(c * h * w),
        "n_classes": int(y_tr.max() + 1),
        "in_channels": int(c),
        "image_size": int(h),
        "train_pool": int(len(idx_rest)),
        "input_norm": input_norm,
        **stats,
    }
