"""
Modulo de carga de datos para QPSO Neural Network Training.

Este modulo proporciona funciones para cargar y preprocesar
diferentes datasets para el entrenamiento de redes neuronales.
"""

from .mcw import (
    MCWDataset,
    load_mcw,
    MCW_CLASSES,
    REDUCTION_METHODS
)

__all__ = [
    "MCWDataset",
    "load_mcw",
    "MCW_CLASSES",
    "REDUCTION_METHODS"
]
