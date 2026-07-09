"""Extracción de features de imágenes para los datasets del paper (MCW, Fashion-MNIST).

SOLO extracción cruda (84 features por imagen) + caché .npz. La normalización,
el split y la reducción de dimensionalidad viven en `core/data.py`, siguiendo el
protocolo sin fugas del paper: todo fit (scaler y reductor) SOLO sobre train.

Features por imagen (siempre 84):
- MCW (color, redimensionada a 150×150):
    histograma HSV 4³=64 + Haralick 13 + momentos de Hu 7
- Fashion-MNIST (grises, 28×28 nativa):
    histograma de intensidad 64 bins + Haralick 13 + momentos de Hu 7
    (espejo monocanal del pipeline MCW: el histograma de intensidad reemplaza
    al HSV, que no existe sin color)

La extracción corre una vez por dataset; el resultado crudo (X_84, y) se cachea
en `swarm_mlp/cache/` (gitignored).
"""
from pathlib import Path
from typing import Tuple

import numpy as np

# cv2/mahotas/torchvision se importan lazy dentro de las funciones que los usan:
# el resto del paquete (iris/breast) no debe pagar ese costo ni requerirlos.

MCW_CLASSES = ("cloudy", "rain", "shine", "sunrise")
FMNIST_PER_CLASS = 1000          # submuestra estratificada: 1000 × 10 clases = 10k
FMNIST_SUBSAMPLE_SEED = 100      # seed FIJO — la submuestra es canónica; los
                                 # data_seeds de los splits operan sobre ella

_HERE = Path(__file__).parent.resolve()          # swarm_mlp/core/
DEFAULT_CACHE_DIR = _HERE.parent / "cache"       # swarm_mlp/cache/  (gitignored)
DEFAULT_MCW_ROOT = _HERE.parent.parent / "data" / "img" / "mcw"   # ann/data/img/mcw


# ---------------------------------------------------------------------------
# Extractores por imagen (84 features)
# ---------------------------------------------------------------------------

def _haralick_13(gray: np.ndarray) -> np.ndarray:
    """Textura Haralick (13): media sobre las 4 direcciones de la GLCM."""
    import mahotas
    h = mahotas.features.haralick(gray).mean(axis=0)
    # Imágenes casi constantes pueden producir no-finitos; neutralizarlos.
    return np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)


def _hu_7(gray: np.ndarray) -> np.ndarray:
    """Momentos de Hu (7) sobre la imagen en grises."""
    import cv2
    moments = cv2.moments(gray)
    return cv2.HuMoments(moments).flatten()


def _hsv_histogram_64(bgr: np.ndarray, bins: int = 4) -> np.ndarray:
    """Histograma HSV 3D normalizado (bins³ = 64 para bins=4). Igual que mcw.py."""
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        [bins, bins, bins],
                        [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _gray_histogram_64(gray: np.ndarray, bins: int = 64) -> np.ndarray:
    """Histograma de intensidad normalizado (64 bins) — análogo monocanal del HSV-64."""
    import cv2
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_mcw_features(bgr: np.ndarray, img_size: Tuple[int, int] = (150, 150)) -> np.ndarray:
    """84 features de una imagen BGR de MCW (misma receta que ann/data/mcw.py)."""
    import cv2
    bgr = cv2.resize(bgr, img_size)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return np.hstack([_hsv_histogram_64(bgr), _haralick_13(gray), _hu_7(gray)])


def extract_fmnist_features(gray: np.ndarray) -> np.ndarray:
    """84 features de una imagen 28×28 en grises (uint8) de Fashion-MNIST."""
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    return np.hstack([_gray_histogram_64(gray), _haralick_13(gray), _hu_7(gray)])


# ---------------------------------------------------------------------------
# Loaders crudos con caché .npz
# ---------------------------------------------------------------------------

def _load_npz_cache(cache_path: Path):
    if cache_path.exists():
        d = np.load(cache_path)
        return d["X"], d["y"]
    return None


def _save_npz_cache(cache_path: Path, X: np.ndarray, y: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y)


def load_mcw_raw(
    root: Path = DEFAULT_MCW_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """MCW crudo: (X ~[1123, 84] float64, y [1123] int64). Cachea en mcw_raw84.npz."""
    cache_path = Path(cache_dir) / "mcw_raw84.npz"
    cached = _load_npz_cache(cache_path)
    if cached is not None:
        return cached

    import cv2
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"No existe {root}. El dataset MCW (imágenes) debe estar en ann/data/img/mcw/"
            f" con subcarpetas {list(MCW_CLASSES)}."
        )

    features, labels = [], []
    for class_idx, class_name in enumerate(MCW_CLASSES):
        class_path = root / class_name
        image_files = sorted(class_path.glob("*"))
        if verbose:
            print(f"  [mcw] {class_name}: {len(image_files)} imágenes")
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            features.append(extract_mcw_features(img))
            labels.append(class_idx)

    if not features:
        raise ValueError(f"No se pudo cargar ninguna imagen válida desde {root}")

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    _save_npz_cache(cache_path, X, y)
    if verbose:
        print(f"  [mcw] features crudos: {X.shape} → caché {cache_path}")
    return X, y


def load_fmnist_raw(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    per_class: int = FMNIST_PER_CLASS,
    subsample_seed: int = FMNIST_SUBSAMPLE_SEED,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fashion-MNIST crudo: (X [10000, 84], y [10000]). Cachea en fmnist_raw84.npz.

    Submuestra estratificada CANÓNICA del train split (60k): `per_class` imágenes
    por clase con seed fijo (default 100). Los 5 data_seeds de los splits del
    protocolo operan siempre sobre esta misma submuestra, igual que en los demás
    datasets (donde la "población" también es fija).
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"fmnist_raw84_{per_class}pc_seed{subsample_seed}.npz"
    cached = _load_npz_cache(cache_path)
    if cached is not None:
        return cached

    from torchvision.datasets import FashionMNIST

    tv_root = cache_dir / "torchvision"
    tv_root.mkdir(parents=True, exist_ok=True)
    ds = FashionMNIST(root=str(tv_root), train=True, download=True)
    images = ds.data.numpy()          # [60000, 28, 28] uint8
    labels = ds.targets.numpy()       # [60000] int64

    rng = np.random.default_rng(subsample_seed)
    idx_sel = []
    for c in range(10):
        idx_c = np.flatnonzero(labels == c)
        idx_sel.append(rng.choice(idx_c, size=per_class, replace=False))
    idx_sel = np.sort(np.concatenate(idx_sel))

    if verbose:
        print(f"  [fmnist] submuestra estratificada: {len(idx_sel)} imágenes "
              f"({per_class}/clase, seed={subsample_seed}); extrayendo 84 features…")

    features = [extract_fmnist_features(images[i]) for i in idx_sel]

    X = np.asarray(features, dtype=np.float64)
    y = labels[idx_sel].astype(np.int64)
    _save_npz_cache(cache_path, X, y)
    if verbose:
        print(f"  [fmnist] features crudos: {X.shape} → caché {cache_path}")
    return X, y
