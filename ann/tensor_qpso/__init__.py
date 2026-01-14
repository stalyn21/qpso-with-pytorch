"""
Tensor QPSO - Optimizadores QPSO basados en tensores.

Este modulo contiene implementaciones optimizadas de QPSO y QDPSO
usando operaciones tensoriales para mejor rendimiento en GPU.
"""

from .qpso_tensor_optimized import (
    QPSOTensorOptimized,
    QDPSOTensorOptimized,
    OptimizationResult,
    CallbackEvent,
    BoundaryStrategy,
    get_device
)

__all__ = [
    'QPSOTensorOptimized',
    'QDPSOTensorOptimized',
    'OptimizationResult',
    'CallbackEvent',
    'BoundaryStrategy',
    'get_device'
]
