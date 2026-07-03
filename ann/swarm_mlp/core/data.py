"""Carga de datasets benchmark + splits (estratificado 70/15/15 y K-fold CV) + tensorizado a device."""
from typing import Dict, Tuple
import numpy as np
import torch
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_circles
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


BENCHMARK_DATASETS = ("iris", "wine", "breast", "circle")


def load_dataset(name: str, data_seed: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    name = name.lower()
    if name == "iris":
        d = load_iris()
        return d.data, d.target
    if name == "wine":
        d = load_wine()
        return d.data, d.target
    if name == "breast":
        d = load_breast_cancer()
        return d.data, d.target
    if name == "circle":
        X, y = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=data_seed)
        return X, y
    raise ValueError(f"Unknown dataset: {name}. Options: {BENCHMARK_DATASETS}")


def split_70_15_15(X: np.ndarray, y: np.ndarray, data_seed: int = 100):
    """Estratificado: 70% train, 15% val, 15% test."""
    X_rem, X_test, y_rem, y_test = train_test_split(
        X, y, test_size=0.15, random_state=data_seed, stratify=y
    )
    # 15/85 ≈ 0.1765 → 15% del total para val, 70% para train
    X_train, X_val, y_train, y_val = train_test_split(
        X_rem, y_rem, test_size=0.1765, random_state=data_seed, stratify=y_rem
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_dataset(
    name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    data_seed: int = 100,
) -> Dict[str, object]:
    """Carga + normaliza + split estratificado + tensoriza + mueve a `device`."""
    X, y = load_dataset(name, data_seed=data_seed)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, y_train, X_val, y_val, X_test, y_test = split_70_15_15(X, y, data_seed=data_seed)

    def to_float(a):
        return torch.as_tensor(a, dtype=dtype, device=device)

    def to_long(a):
        return torch.as_tensor(a, dtype=torch.long, device=device)

    return {
        "X_train": to_float(X_train), "y_train": to_long(y_train),
        "X_val":   to_float(X_val),   "y_val":   to_long(y_val),
        "X_test":  to_float(X_test),  "y_test":  to_long(y_test),
        "n_features": int(X_train.shape[1]),
        "n_classes":  int(np.unique(y).size),
    }


def prepare_dataset_kfold(
    name: str,
    k: int,
    fold_idx: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    data_seed: int = 100,
    val_frac_within_train: float = 0.15,
) -> Dict[str, object]:
    """K-fold CV estratificado para evaluar generalización sin sesgo de un solo split.

    Para fold_idx ∈ [0, k):
      - test = el fold k-ésimo (~ n_total / k muestras)
      - el resto (k-1 folds) → train + val
      - dentro de ese resto se hace un sub-split estratificado para val
        (`val_frac_within_train`, default 15% del 80% no-test ≈ 12% del total)

    Importante (sin data leakage):
      - El MinMaxScaler se ajusta SÓLO con `train_fold` y se aplica a val/test.
      - El split de test usa StratifiedKFold con `data_seed`; el split de val
        usa `train_test_split` con `data_seed + fold_idx + 1` para variar
        ligeramente entre folds.

    Devuelve el mismo dict que `prepare_dataset` para compatibilidad con
    `run_sessions`, más metadatos `fold_idx`, `n_folds`.
    """
    if not (0 <= fold_idx < k):
        raise ValueError(f"fold_idx={fold_idx} fuera de rango [0, {k})")
    if k < 2:
        raise ValueError(f"K-fold requiere k >= 2 (recibido k={k})")

    X, y = load_dataset(name, data_seed=data_seed)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=data_seed)
    splits = list(skf.split(X, y))
    train_val_idx, test_idx = splits[fold_idx]

    X_tv, y_tv = X[train_val_idx], y[train_val_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]

    # Sub-split estratificado dentro del 80% no-test → train (~68%) + val (~12%).
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_frac_within_train,
        random_state=data_seed + fold_idx + 1,
        stratify=y_tv,
    )

    # Fit scaler SOLO con train (sin leakage), aplicar a val/test.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    def to_float(a):
        return torch.as_tensor(a, dtype=dtype, device=device)

    def to_long(a):
        return torch.as_tensor(a, dtype=torch.long, device=device)

    return {
        "X_train": to_float(X_train), "y_train": to_long(y_train),
        "X_val":   to_float(X_val),   "y_val":   to_long(y_val),
        "X_test":  to_float(X_test),  "y_test":  to_long(y_test),
        "n_features": int(X_train.shape[1]),
        "n_classes":  int(np.unique(y).size),
        "fold_idx": int(fold_idx),
        "n_folds":  int(k),
    }
