"""
QPSO Neural Network Training Package

Este paquete proporciona una implementacion modular para entrenar
redes neuronales utilizando QPSO (Quantum Particle Swarm Optimization)
como optimizador en lugar de backpropagation tradicional.

Modulos:
    - models: Modelos de redes neuronales compatibles con QPSO
    - optimizers: Optimizadores QPSO para redes neuronales
    - trainers: Logica de entrenamiento y validacion
    - utils: Utilidades para datos y metricas

Ejemplo de uso:
    from models import QPSOCompatibleANN
    from optimizers import QPSONNOptimizer
    from trainers import Trainer
"""

__version__ = "2.0.0"
__author__ = "Stalyn Javier Chancay Moreira"
__copyright__ = "Copyright (c) 2024 Stalyn Javier Chancay Moreira"
__license__ = "MIT"

from .models import QPSOCompatibleANN
from .optimizers import QPSONNOptimizer
from .trainers import Trainer, TrainingConfig

__all__ = [
    "QPSOCompatibleANN",
    "QPSONNOptimizer",
    "Trainer",
    "TrainingConfig",
]
