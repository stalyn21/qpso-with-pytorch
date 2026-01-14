"""
MCW (Multi-Class Weather) Dataset Loader

Este modulo proporciona funciones para cargar y preprocesar el dataset
de imagenes de clima (MCW) para clasificacion con redes neuronales.

Features extraidas:
- Histogram HSV: 64 features (bins^3 donde bins=4)
- Haralick texture: 13 features
- Hu moments: 7 features
- Total: 84 features

Clases:
- cloudy (0)
- rain (1)
- shine (2)
- sunrise (3)

Metodos de reduccion disponibles:
- isomap: Isometric Mapping (preserva distancias geodesicas)
- pca: Principal Component Analysis (lineal, rapido)
- mds: Multidimensional Scaling (no soporta transform)
"""

import cv2
import numpy as np
import mahotas
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import Isomap, MDS
from sklearn.decomposition import PCA


# =============================================================================
# CONSTANTES
# =============================================================================

MCW_CLASSES = ['cloudy', 'rain', 'shine', 'sunrise']
REDUCTION_METHODS = ['isomap', 'pca', 'mds']


# =============================================================================
# DATACLASS PARA RESULTADO
# =============================================================================

@dataclass
class MCWDataResult:
    """Resultado de la carga del dataset MCW."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    class_names: List[str]
    n_features: int
    n_classes: int
    reduction_method: Optional[str]
    n_components: Optional[int]
    original_features: int

    def __repr__(self) -> str:
        return (
            f"MCWDataResult(\n"
            f"  train_samples={len(self.y_train)},\n"
            f"  val_samples={len(self.y_val)},\n"
            f"  test_samples={len(self.y_test)},\n"
            f"  n_features={self.n_features},\n"
            f"  n_classes={self.n_classes},\n"
            f"  reduction={self.reduction_method},\n"
            f"  original_features={self.original_features}\n"
            f")"
        )


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================

class MCWDataset:
    """
    Cargador y procesador del dataset MCW (Multi-Class Weather).

    Extrae features de imagenes usando:
    - Histograma de color HSV
    - Caracteristicas de textura Haralick
    - Momentos de Hu

    Args:
        root_path: Ruta al directorio con imagenes
        img_size: Tamaño de redimension de imagenes (alto, ancho)
        bins: Numero de bins para histograma (default: 4, genera bins^3=64 features)

    Example:
        >>> dataset = MCWDataset('./data/img/mcw')
        >>> result = dataset.load(train_size=0.7, val_size=0.1, test_size=0.2)
        >>> print(result.n_features)
        84
    """

    def __init__(
        self,
        root_path: str = './data/img/mcw',
        img_size: Tuple[int, int] = (150, 150),
        bins: int = 4
    ):
        self.root_path = Path(root_path)
        self.img_size = img_size
        self.bins = bins
        self.classes = MCW_CLASSES

        # Tamaños de features
        self.hist_size = bins ** 3  # 64 para bins=4
        self.haralick_size = 13
        self.hu_size = 7
        self.total_features = self.hist_size + self.haralick_size + self.hu_size

        # Scalers para normalizacion por tipo de feature
        self.scaler_hist = StandardScaler()
        self.scaler_haralick = StandardScaler()
        self.scaler_hu = StandardScaler()

        # Reductor de dimensionalidad
        self.reducer = None

        self._validate_structure()

    def _validate_structure(self) -> None:
        """Valida que exista la estructura de directorios correcta."""
        if not self.root_path.exists():
            raise ValueError(f"El directorio {self.root_path} no existe")

        missing = []
        for class_name in self.classes:
            class_path = self.root_path / class_name
            if not class_path.exists():
                missing.append(class_name)

        if missing:
            raise ValueError(
                f"Directorios faltantes: {missing}\n"
                f"Estructura esperada:\n"
                f"{self.root_path}/\n" +
                "\n".join(f"├── {c}/" for c in self.classes)
            )

    def _extract_hu_moments(self, image: np.ndarray) -> np.ndarray:
        """Extrae momentos de Hu (7 features)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).flatten()
        return hu

    def _extract_haralick(self, image: np.ndarray) -> np.ndarray:
        """Extrae caracteristicas de textura Haralick (13 features)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        return haralick

    def _extract_histogram(self, image: np.ndarray) -> np.ndarray:
        """Extrae histograma de color HSV (bins^3 features)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [self.bins, self.bins, self.bins],
            [0, 256, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extrae y combina todas las features de una imagen."""
        image = cv2.resize(image, self.img_size)

        hist = self._extract_histogram(image)
        haralick = self._extract_haralick(image)
        hu = self._extract_hu_moments(image)

        return np.hstack([hist, haralick, hu])

    def _load_images(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Carga todas las imagenes y extrae features."""
        features = []
        labels = []

        for class_idx, class_name in enumerate(self.classes):
            class_path = self.root_path / class_name
            image_files = list(class_path.glob('*'))

            if not image_files:
                print(f"Advertencia: No hay imagenes en {class_path}")
                continue

            if verbose:
                print(f"Procesando {class_name}: {len(image_files)} imagenes")

            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    feat = self._extract_features(img)
                    features.append(feat)
                    labels.append(class_idx)

                except Exception as e:
                    if verbose:
                        print(f"Error en {img_path.name}: {e}")

        if not features:
            raise ValueError("No se pudieron cargar imagenes validas")

        return np.array(features), np.array(labels)

    def _normalize_features(
        self,
        X: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Normaliza features por tipo (histogram, haralick, hu).

        Args:
            X: Features a normalizar
            fit: Si True, ajusta los scalers; si False, solo transforma

        Returns:
            Features normalizadas
        """
        # Separar por tipo
        X_hist = X[:, :self.hist_size]
        X_haralick = X[:, self.hist_size:self.hist_size + self.haralick_size]
        X_hu = X[:, -self.hu_size:]

        if fit:
            X_hist_norm = self.scaler_hist.fit_transform(X_hist)
            X_haralick_norm = self.scaler_haralick.fit_transform(X_haralick)
            X_hu_norm = self.scaler_hu.fit_transform(X_hu)
        else:
            X_hist_norm = self.scaler_hist.transform(X_hist)
            X_haralick_norm = self.scaler_haralick.transform(X_haralick)
            X_hu_norm = self.scaler_hu.transform(X_hu)

        return np.hstack([X_hist_norm, X_haralick_norm, X_hu_norm])

    def _apply_reduction(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        method: str,
        n_components: int,
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aplica reduccion de dimensionalidad.

        Args:
            X_train: Features de entrenamiento
            X_val: Features de validacion
            X_test: Features de test
            method: Metodo de reduccion ('isomap', 'pca', 'mds')
            n_components: Numero de componentes objetivo
            random_state: Semilla aleatoria

        Returns:
            Tuple con features reducidas (train, val, test)
        """
        method = method.lower()

        if method not in REDUCTION_METHODS:
            raise ValueError(
                f"Metodo '{method}' no soportado. "
                f"Use: {REDUCTION_METHODS}"
            )

        if method == 'isomap':
            n_neighbors = min(100, X_train.shape[0] - 1)
            self.reducer = Isomap(
                n_components=n_components,
                n_neighbors=n_neighbors
            )
            X_train_red = self.reducer.fit_transform(X_train)
            X_val_red = self.reducer.transform(X_val)
            X_test_red = self.reducer.transform(X_test)

        elif method == 'pca':
            self.reducer = PCA(n_components=n_components)
            X_train_red = self.reducer.fit_transform(X_train)
            X_val_red = self.reducer.transform(X_val)
            X_test_red = self.reducer.transform(X_test)

        elif method == 'mds':
            # MDS no soporta transform, se aplica a cada conjunto
            self.reducer = MDS(
                n_components=n_components,
                random_state=random_state,
                normalized_stress='auto'
            )
            X_train_red = self.reducer.fit_transform(X_train)

            reducer_val = MDS(
                n_components=n_components,
                random_state=random_state,
                normalized_stress='auto'
            )
            X_val_red = reducer_val.fit_transform(X_val)

            reducer_test = MDS(
                n_components=n_components,
                random_state=random_state,
                normalized_stress='auto'
            )
            X_test_red = reducer_test.fit_transform(X_test)

        return X_train_red, X_val_red, X_test_red

    def load(
        self,
        train_size: float = 0.70,
        val_size: float = 0.10,
        test_size: float = 0.20,
        reduction_method: Optional[str] = None,
        n_components: Optional[int] = None,
        random_state: int = 42,
        verbose: bool = True
    ) -> MCWDataResult:
        """
        Carga y preprocesa el dataset MCW.

        Args:
            train_size: Proporcion de entrenamiento (default: 0.70)
            val_size: Proporcion de validacion (default: 0.10)
            test_size: Proporcion de test (default: 0.20)
            reduction_method: Metodo de reduccion ('isomap', 'pca', 'mds')
            n_components: Componentes objetivo (default: n_features/6)
            random_state: Semilla aleatoria
            verbose: Mostrar progreso

        Returns:
            MCWDataResult con los datos procesados
        """
        if verbose:
            print("="*60)
            print(" CARGANDO DATASET MCW")
            print("="*60)

        # 1. Cargar imagenes y extraer features
        if verbose:
            print("\n1. Extrayendo features de imagenes...")

        X, y = self._load_images(verbose=verbose)

        if verbose:
            print(f"\n   Total imagenes: {len(y)}")
            print(f"   Features originales: {X.shape[1]}")
            print(f"   - Histogram HSV: {self.hist_size}")
            print(f"   - Haralick: {self.haralick_size}")
            print(f"   - Hu moments: {self.hu_size}")

        # 2. Split train/test
        if verbose:
            print("\n2. Dividiendo dataset...")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # Split train/val
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp
        )

        if verbose:
            print(f"   Train: {len(y_train)} ({train_size*100:.0f}%)")
            print(f"   Val: {len(y_val)} ({val_size*100:.0f}%)")
            print(f"   Test: {len(y_test)} ({test_size*100:.0f}%)")

        # 3. Normalizar por tipo de feature
        if verbose:
            print("\n3. Normalizando features por tipo...")

        X_train_norm = self._normalize_features(X_train, fit=True)
        X_val_norm = self._normalize_features(X_val, fit=False)
        X_test_norm = self._normalize_features(X_test, fit=False)

        original_features = X_train_norm.shape[1]

        # 4. Reduccion de dimensionalidad (opcional)
        if reduction_method is not None:
            if n_components is None:
                n_components = original_features // 6

            if verbose:
                print(f"\n4. Aplicando reduccion de dimensionalidad...")
                print(f"   Metodo: {reduction_method}")
                print(f"   Componentes: {n_components}")

            X_train_final, X_val_final, X_test_final = self._apply_reduction(
                X_train_norm, X_val_norm, X_test_norm,
                method=reduction_method,
                n_components=n_components,
                random_state=random_state
            )

            if verbose:
                print(f"   Shape final: {X_train_final.shape[1]} features")
        else:
            X_train_final = X_train_norm
            X_val_final = X_val_norm
            X_test_final = X_test_norm

        # 5. Estadisticas finales
        if verbose:
            print("\n5. Distribucion de clases:")
            for i, name in enumerate(self.classes):
                train_count = np.sum(y_train == i)
                val_count = np.sum(y_val == i)
                test_count = np.sum(y_test == i)
                print(f"   {name}: train={train_count}, val={val_count}, test={test_count}")
            print("="*60)

        return MCWDataResult(
            X_train=X_train_final,
            X_val=X_val_final,
            X_test=X_test_final,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            class_names=self.classes,
            n_features=X_train_final.shape[1],
            n_classes=len(self.classes),
            reduction_method=reduction_method,
            n_components=n_components,
            original_features=original_features
        )


# =============================================================================
# FUNCION AUXILIAR
# =============================================================================

def load_mcw(
    root_path: str = './data/img/mcw',
    train_size: float = 0.70,
    val_size: float = 0.10,
    test_size: float = 0.20,
    reduction_method: Optional[str] = None,
    n_components: Optional[int] = None,
    random_state: int = 42,
    img_size: Tuple[int, int] = (150, 150),
    bins: int = 4,
    verbose: bool = True
) -> MCWDataResult:
    """
    Funcion auxiliar para cargar el dataset MCW de forma directa.

    Args:
        root_path: Ruta al directorio de imagenes
        train_size: Proporcion de entrenamiento (default: 0.70)
        val_size: Proporcion de validacion (default: 0.10)
        test_size: Proporcion de test (default: 0.20)
        reduction_method: Metodo de reduccion ('isomap', 'pca', 'mds')
        n_components: Componentes objetivo (default: n_features/6)
        random_state: Semilla aleatoria
        img_size: Tamaño de imagenes
        bins: Bins para histograma
        verbose: Mostrar progreso

    Returns:
        MCWDataResult con los datos procesados

    Example:
        >>> data = load_mcw('./data/img/mcw', reduction_method='pca', n_components=20)
        >>> print(f"Features: {data.n_features}, Classes: {data.n_classes}")
    """
    dataset = MCWDataset(
        root_path=root_path,
        img_size=img_size,
        bins=bins
    )

    return dataset.load(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        reduction_method=reduction_method,
        n_components=n_components,
        random_state=random_state,
        verbose=verbose
    )
