"""Datasets de imágenes para la CNN del paper 2 (ver MEMORY/.../paper2-cnn-plan.md).

Dos protocolos, según si el dataset trae test oficial:

MNIST y Fashion-MNIST (`prepare_image_dataset`):
- **Test = test oficial** (10,000 imágenes), para compararse con la literatura.
- Del train oficial (60,000) salen, estratificados y con `data_seed`:
    * validación: `val_size` imágenes (selección de modelos);
    * fitness: `fitness_samples` imágenes, sobre las que los enjambres evalúan la
      función de costo (0 = todo el resto del train). Es la "train" de la corrida:
      train_acc y la L2 (n_train) se miden sobre ella.

MCW, Multi-Class Weather (`prepare_image_dataset_kfold`):
- 1,123 imágenes a color de tamaños variados, 4 clases, sin test oficial. Trae 25
  grupos de archivos idénticos byte a byte (26 copias de más, siempre en la misma
  clase): con `dedupe=True` (default) se queda la primera de cada grupo en orden de
  nombre, 1,097 imágenes. Sin eso una copia puede caer en fitness y otra en test.
- Se redimensionan a `image_size` × `image_size` estirando (sin recortar, como en el
  paper 1) con interpolación por área, que promedia los píxeles al reducir.
- **K-fold CV estratificado, igual que el MLP** (`core.data.prepare_dataset_kfold`):
  test = el fold `fold_idx`; del resto sale la validación (`val_frac_within_train`)
  y lo demás es fitness (submuestreado si `fitness_samples` es menor).

En ambos casos los píxeles quedan en [0, 1] (división por 255). Con
`input_norm="standardize"` además se resta la media y se divide por el desvío de cada
canal, calculados SÓLO sobre las imágenes de fitness: validación y test no aportan
nada al ajuste.
"""
from pathlib import Path
from typing import Any, Dict, Optional

import hashlib

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from core.features import DEFAULT_CACHE_DIR, DEFAULT_MCW_ROOT, MCW_CLASSES

#: Datasets de imágenes que acepta la CNN.
IMAGE_DATASETS = ("fmnist", "mnist", "mcw")

#: Con test oficial: una partición por `data_seed` (no admiten K-fold).
OFFICIAL_TEST_DATASETS = ("fmnist", "mnist")

#: Sin test oficial: K-fold CV estratificado, como el MLP.
CV_IMAGE_DATASETS = ("mcw",)

#: Lado en píxeles al que se redimensiona cada dataset de tamaño variable si no se pide
#: otro. 32 px deja la CNN de un tamaño parecido a la de Fashion-MNIST (small-flat:
#: 3,700 parámetros contra 5,258) y es ~5 veces más barata que 64 px.
DEFAULT_IMAGE_SIZE = {"mcw": 32}

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


def _load_mcw(image_size: int, dedupe: bool = True, root: Path = DEFAULT_MCW_ROOT,
              cache_dir: Path = DEFAULT_CACHE_DIR):
    """MCW como (X [N, 3, S, S] float32 en [0, 1], y [N] int64), en orden RGB.

    Decodificar las 1,123 imágenes (algunas de 4,752 px) tarda; el resultado se cachea
    en `cache/mcw_img{S}[_dedup].npz` como uint8. Las clases siguen `MCW_CLASSES` y los
    archivos van ordenados por nombre, así el orden (y las particiones) es estable.
    Con `dedupe`, de cada grupo de archivos idénticos (mismo MD5) queda el primero.
    """
    if image_size < 1:
        raise ValueError(f"image_size debe ser >= 1; se dio {image_size}")
    cache_path = Path(cache_dir) / f"mcw_img{image_size}{'_dedup' if dedupe else ''}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        X8, y = d["X"], d["y"]
    else:
        import cv2

        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(
                f"No existe {root}. MCW debe estar en ann/data/img/mcw/ con subcarpetas "
                f"{list(MCW_CLASSES)}.")
        images, labels, seen, n_dup = [], [], set(), 0
        for class_idx, class_name in enumerate(MCW_CLASSES):
            files = sorted(p for p in (root / class_name).iterdir() if p.is_file())
            for path in files:
                if dedupe:
                    digest = hashlib.md5(path.read_bytes()).hexdigest()
                    if digest in seen:
                        n_dup += 1
                        continue
                    seen.add(digest)
                # IMREAD_COLOR lleva a 3 canales las imágenes en gris y quita el alfa.
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"No se pudo leer la imagen {path}")
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
                labels.append(class_idx)
        X8 = np.stack(images).astype(np.uint8)
        y = np.asarray(labels, dtype=np.int64)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X8, y=y)
        print(f"  [mcw] {len(y)} imágenes a {image_size}x{image_size} "
              f"({n_dup} duplicados descartados) → caché {cache_path}")
    return X8.astype(np.float32) / 255.0, y.astype(np.int64)


def _check_input_norm(input_norm: str) -> None:
    if input_norm not in INPUT_NORMS:
        raise ValueError(f"input_norm desconocido: {input_norm}. Opciones: {INPUT_NORMS}")


def _subsample_fitness(idx: np.ndarray, y: np.ndarray, fitness_samples: int,
                       random_state: int) -> np.ndarray:
    """Submuestra estratificada de `idx` para fitness (0 o >= len(idx) = todo)."""
    if fitness_samples and fitness_samples < len(idx):
        idx, _ = train_test_split(idx, train_size=fitness_samples, stratify=y[idx],
                                  random_state=random_state)
    return idx


def _pack(X_fit, y_fit, X_val, y_val, X_te, y_te, *, device, dtype, input_norm,
          norm_eps, n_classes, extra) -> Dict[str, Any]:
    """Normalización de la entrada (ajustada sólo en fitness) y paso a tensores."""
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

    c, h, w = X_fit.shape[1:]
    return {
        "X_train": f(X_fit), "y_train": l(y_fit),
        "X_val": f(X_val), "y_val": l(y_val),
        "X_test": f(X_te), "y_test": l(y_te),
        "n_features": int(c * h * w),
        "n_classes": int(n_classes),
        "in_channels": int(c),
        "image_size": int(h),
        "input_norm": input_norm,
        **extra,
        **stats,
    }


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
    """Splits de un dataset con test oficial como tensores (N, C, H, W) en `device`."""
    if name not in OFFICIAL_TEST_DATASETS:
        if name in CV_IMAGE_DATASETS:
            raise ValueError(f"{name} no tiene test oficial: usar prepare_image_dataset_kfold "
                             "(--cv-folds K en el runner).")
        raise ValueError(f"Dataset de imágenes desconocido: {name}. Opciones: {IMAGE_DATASETS}")
    _check_input_norm(input_norm)
    X_tr, y_tr = _load_official(name, train=True)
    X_te, y_te = _load_official(name, train=False)

    if not 0 < val_size < len(y_tr):
        raise ValueError(f"val_size debe estar en (0, {len(y_tr)}); se dio {val_size}")
    idx = np.arange(len(y_tr))
    idx_rest, idx_val = train_test_split(idx, test_size=val_size, stratify=y_tr,
                                         random_state=data_seed)
    idx_fit = np.sort(_subsample_fitness(idx_rest, y_tr, fitness_samples, data_seed + 1))
    idx_val = np.sort(idx_val)
    return _pack(X_tr[idx_fit], y_tr[idx_fit], X_tr[idx_val], y_tr[idx_val], X_te, y_te,
                 device=device, dtype=dtype, input_norm=input_norm, norm_eps=norm_eps,
                 n_classes=y_tr.max() + 1, extra={"train_pool": int(len(idx_rest))})


def prepare_image_dataset_kfold(
    name: str,
    k: int,
    fold_idx: int,
    device: torch.device,
    dtype: torch.dtype,
    data_seed: int = 100,
    val_frac_within_train: float = 0.15,
    fitness_samples: int = 0,
    input_norm: str = "none",
    norm_eps: float = 1e-6,
    image_size: Optional[int] = None,
    dedupe: bool = True,
) -> Dict[str, Any]:
    """Fold `fold_idx` de K-fold CV estratificado para un dataset sin test oficial.

    Mismo esquema que `core.data.prepare_dataset_kfold` (el del MLP):
      - test = el fold `fold_idx` (StratifiedKFold con `data_seed`, ~N/k imágenes);
      - del resto, `val_frac_within_train` va a validación (semilla
        `data_seed + fold_idx + 1`) y lo demás es el pool de fitness;
      - si `fitness_samples` es menor que el pool, se submuestrea estratificado
        (semilla `data_seed + fold_idx + 2`); 0 = todo el pool.
    `dedupe` descarta los archivos repetidos antes de particionar (ver `_load_mcw`).
    """
    if name not in CV_IMAGE_DATASETS:
        if name in OFFICIAL_TEST_DATASETS:
            raise ValueError(f"{name} usa el test oficial: usar prepare_image_dataset.")
        raise ValueError(f"Dataset de imágenes desconocido: {name}. Opciones: {IMAGE_DATASETS}")
    if k < 2:
        raise ValueError(f"K-fold requiere k >= 2 (recibido k={k})")
    if not 0 <= fold_idx < k:
        raise ValueError(f"fold_idx={fold_idx} fuera de rango [0, {k})")
    if not 0 < val_frac_within_train < 1:
        raise ValueError(f"val_frac_within_train debe estar en (0, 1); se dio {val_frac_within_train}")
    _check_input_norm(input_norm)
    size = image_size if image_size is not None else DEFAULT_IMAGE_SIZE[name]
    X, y = _load_mcw(size, dedupe=dedupe)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=data_seed)
    idx_tv, idx_te = list(skf.split(X, y))[fold_idx]
    idx_pool, idx_val = train_test_split(idx_tv, test_size=val_frac_within_train,
                                         stratify=y[idx_tv], random_state=data_seed + fold_idx + 1)
    idx_fit = np.sort(_subsample_fitness(idx_pool, y, fitness_samples, data_seed + fold_idx + 2))
    idx_val, idx_te = np.sort(idx_val), np.sort(idx_te)
    return _pack(X[idx_fit], y[idx_fit], X[idx_val], y[idx_val], X[idx_te], y[idx_te],
                 device=device, dtype=dtype, input_norm=input_norm, norm_eps=norm_eps,
                 n_classes=len(MCW_CLASSES),
                 extra={"train_pool": int(len(idx_pool)), "fold_idx": fold_idx, "n_folds": k,
                        "dedupe": dedupe})
