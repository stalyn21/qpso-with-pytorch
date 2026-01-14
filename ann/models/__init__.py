"""
Modelos de Redes Neuronales Compatibles con QPSO

Este modulo proporciona implementaciones de redes neuronales
que pueden ser optimizadas con QPSO.
"""

from .ann import QPSOCompatibleANN, create_scaled_architecture

__all__ = ["QPSOCompatibleANN", "create_scaled_architecture"]
