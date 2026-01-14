"""
Optimizadores QPSO para Redes Neuronales

Este modulo proporciona optimizadores basados en QPSO
para entrenar redes neuronales sin backpropagation.

Optimizadores disponibles:
    - QPSONNOptimizer: Optimizador basado en QPSO
    - QDPSONNOptimizer: Optimizador basado en QDPSO

Estrategias de entrenamiento:
    - ForwardStrategy: Entrenamiento estandar
    - WeightedStrategy: Forward con pesos por capa
    - LayerwiseStrategy: Entrenamiento capa por capa
"""

from .qpso_nn import QPSONNOptimizer, QDPSONNOptimizer, NNOptimizationConfig
from .training_strategies import (
    create_training_strategy,
    StrategyConfig,
    TrainingStrategy,
    TrainingResult,
    ForwardStrategy,
    WeightedStrategy,
    LayerwiseStrategy,
)

__all__ = [
    # Optimizadores
    "QPSONNOptimizer",
    "QDPSONNOptimizer",
    "NNOptimizationConfig",
    # Estrategias
    "create_training_strategy",
    "StrategyConfig",
    "TrainingStrategy",
    "TrainingResult",
    "ForwardStrategy",
    "WeightedStrategy",
    "LayerwiseStrategy",
]
