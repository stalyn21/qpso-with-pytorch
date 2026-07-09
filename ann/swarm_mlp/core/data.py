"""Carga de datasets benchmark + splits (estratificado 70/15/15 y K-fold CV) + tensorizado a device.

Datasets del paper (`PAPER_DATASETS`): iris y breast se usan crudos; las variantes
`mcw-*` y `fmnist-*` parten de 84 features de imagen (ver `core/features.py`) y
aplican reducción de dimensionalidad a 7 componentes POR SPLIT, sin fugas:

    split 70/15/15 (o fold) → MinMax fit(train) → Isomap/PCA(7) fit(train)
    → re-escala MinMax fit(train)

La re-escala final deja todos los datasets en el mismo rango de entrada que
iris/breast. El factor de reducción 84/12 = 7 sigue a paper-1 (mejor factor en MCW).
"""
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_circles
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


BENCHMARK_DATASETS = ("iris", "wine", "breast", "circle")

# Suite estándar del paper 2 (6 variantes; circle/wine quedan fuera del paper).
PAPER_DATASETS = ("iris", "breast", "mcw-isomap", "mcw-pca", "fmnist-isomap", "fmnist-pca")

N_COMPONENTS_REDUCED = 7   # 84 features / 12 (factor de paper-1)


def _parse_reduction(name: str) -> Tuple[str, Optional[str]]:
    """'mcw-isomap' → ('mcw', 'isomap'); 'iris' → ('iris', None)."""
    name = name.lower()
    for method in ("isomap", "pca"):
        suffix = f"-{method}"
        if name.endswith(suffix):
            return name[: -len(suffix)], method
    return name, None


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
    if name == "mcw":
        from core.features import load_mcw_raw
        return load_mcw_raw()
    if name == "fmnist":
        from core.features import load_fmnist_raw
        return load_fmnist_raw()
    raise ValueError(
        f"Unknown dataset: {name}. Options: {BENCHMARK_DATASETS} + {PAPER_DATASETS}"
    )


def _prepared_split_cache(name: str, data_seed: int, tag: str = "split701515"):
    """Ruta del caché del split YA reducido (solo variantes mcw-*/fmnist-*).

    El pipeline completo es determinista dado (name, data_seed): mismo split
    estratificado, mismo fit del reductor. Cachearlo evita re-fitear Isomap
    (~4 min en fmnist) en cada corrida del factorial.
    """
    from core.features import DEFAULT_CACHE_DIR
    return DEFAULT_CACHE_DIR / "prepared" / f"{name}_{tag}_dseed{data_seed}.npz"


def _reduce_after_split(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    method: str,
    n_components: int = N_COMPONENTS_REDUCED,
    data_seed: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pipeline de reducción SIN fugas: todos los fit son SOLO sobre train.

    1. MinMax fit(train) → transform val/test   (estándar del protocolo; el
       StandardScaler por tipo de feature de ann/data/mcw.py NO se usa aquí)
    2. Isomap/PCA(n_components) fit(train) → transform val/test
    3. re-escala MinMax fit(train) → transform val/test (mismo rango de entrada
       que los datasets sin reducción)
    """
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if method == "isomap":
        n_neighbors = min(100, X_train.shape[0] - 1)
        reducer = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    elif method == "pca":
        reducer = PCA(n_components=n_components, random_state=data_seed)
    else:
        raise ValueError(f"Método de reducción desconocido: {method} (use isomap|pca)")

    X_train = reducer.fit_transform(X_train)
    X_val = reducer.transform(X_val)
    X_test = reducer.transform(X_test)

    rescaler = MinMaxScaler()
    X_train = rescaler.fit_transform(X_train)
    X_val = rescaler.transform(X_val)
    X_test = rescaler.transform(X_test)
    return X_train, X_val, X_test


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
    base, method = _parse_reduction(name)
    X, y = load_dataset(base, data_seed=data_seed)

    if method is None:
        # Ruta legacy (iris/wine/breast/circle): MinMax sobre X completo ANTES del
        # split (comportamiento histórico, conservado por reproducibilidad).
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        X_train, y_train, X_val, y_val, X_test, y_test = split_70_15_15(X, y, data_seed=data_seed)
    else:
        # Variantes reducidas (mcw-*/fmnist-*): split PRIMERO, luego scaler +
        # reductor + re-escala con fit SOLO en train (sin fugas).
        # El resultado se cachea por (name, data_seed): el pipeline es determinista
        # y el fit de Isomap es caro (~4 min en fmnist).
        cache_path = _prepared_split_cache(name, data_seed)
        if cache_path.exists():
            d = np.load(cache_path)
            X_train, y_train = d["X_train"], d["y_train"]
            X_val, y_val = d["X_val"], d["y_val"]
            X_test, y_test = d["X_test"], d["y_test"]
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = split_70_15_15(X, y, data_seed=data_seed)
            X_train, X_val, X_test = _reduce_after_split(
                X_train, X_val, X_test, method=method, data_seed=data_seed,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
            )

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

    base, method = _parse_reduction(name)
    X, y = load_dataset(base, data_seed=data_seed)

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

    if method is None:
        # Fit scaler SOLO con train (sin leakage), aplicar a val/test.
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        # Variantes reducidas: scaler + reductor + re-escala, fit SOLO en train.
        # Caché por (name, data_seed, fold) — mismo motivo que en prepare_dataset.
        cache_path = _prepared_split_cache(name, data_seed, tag=f"cv{k}fold{fold_idx}")
        if cache_path.exists():
            d = np.load(cache_path)
            X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
        else:
            X_train, X_val, X_test = _reduce_after_split(
                X_train_raw, X_val_raw, X_test_raw, method=method, data_seed=data_seed,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
            )

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
