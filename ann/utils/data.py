"""
Utilidades para Procesamiento de Datos

Funciones para cargar, preprocesar y manejar datasets
para entrenamiento de redes neuronales con QPSO.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union, List
from torch.utils.data import DataLoader, TensorDataset


def load_dataset(
    name: str,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga un dataset predefinido.

    Args:
        name: Nombre del dataset ('iris', 'wine', 'breast_cancer', 'digits')
        **kwargs: Argumentos adicionales

    Returns:
        Tupla (X_train, X_test, y_train, y_test)

    Example:
        >>> X_train, X_test, y_train, y_test = load_dataset('iris')
    """
    from sklearn import datasets
    from sklearn.model_selection import train_test_split as sklearn_split
    from sklearn.preprocessing import StandardScaler

    dataset_loaders = {
        'iris': datasets.load_iris,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer,
        'digits': datasets.load_digits,
    }

    if name not in dataset_loaders:
        raise ValueError(f"Dataset no soportado: {name}. "
                        f"Opciones: {list(dataset_loaders.keys())}")

    # Cargar dataset
    data = dataset_loaders[name]()
    X, y = data.data, data.target

    # Normalizar
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    test_size = kwargs.get('test_size', 0.2)
    random_state = kwargs.get('random_state', 42)

    X_train, X_test, y_train, y_test = sklearn_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_test_split(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide datos en train y test.

    Args:
        X: Features
        y: Labels
        test_size: Proporcion de test
        random_state: Semilla aleatoria
        stratify: Mantener proporcion de clases

    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split as sklearn_split

    # Convertir a numpy si es tensor
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    stratify_param = y if stratify else None

    return sklearn_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )


def normalize_data(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    method: str = 'standard'
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Normaliza los datos.

    Args:
        X_train: Datos de entrenamiento
        X_test: Datos de test (opcional)
        method: Metodo de normalizacion ('standard', 'minmax', 'robust')

    Returns:
        Datos normalizados
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    scalers = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler
    }

    if method not in scalers:
        raise ValueError(f"Metodo no soportado: {method}")

    scaler = scalers[method]()
    X_train_normalized = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_normalized = scaler.transform(X_test)
        return X_train_normalized, X_test_normalized

    return X_train_normalized


def create_dataloaders(
    X_train: Union[np.ndarray, torch.Tensor],
    y_train: Union[np.ndarray, torch.Tensor],
    X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
    y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
    batch_size: int = 32,
    shuffle_train: bool = True,
    device: str = 'cpu'
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Crea DataLoaders de PyTorch.

    Args:
        X_train: Features de entrenamiento
        y_train: Labels de entrenamiento
        X_val: Features de validacion (opcional)
        y_val: Labels de validacion (opcional)
        batch_size: Tamano de batch
        shuffle_train: Mezclar datos de entrenamiento
        device: Dispositivo

    Returns:
        Tupla (train_loader, val_loader)
    """
    # Convertir a tensores
    if isinstance(X_train, np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if isinstance(y_train, np.ndarray):
        y_train = torch.tensor(y_train, dtype=torch.long)

    # Mover a dispositivo
    device = torch.device(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # Crear dataset y loader de train
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train
    )

    # Crear loader de validacion si existe
    val_loader = None
    if X_val is not None and y_val is not None:
        if isinstance(X_val, np.ndarray):
            X_val = torch.tensor(X_val, dtype=torch.float32)
        if isinstance(y_val, np.ndarray):
            y_val = torch.tensor(y_val, dtype=torch.long)

        X_val = X_val.to(device)
        y_val = y_val.to(device)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    return train_loader, val_loader


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 3,
    n_informative: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera datos sinteticos para clasificacion.

    Args:
        n_samples: Numero de muestras
        n_features: Numero de features
        n_classes: Numero de clases
        n_informative: Numero de features informativos
        random_state: Semilla aleatoria

    Returns:
        Tupla (X, y)
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative - 2,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state
    )

    return X, y
