"""
Utilidades para QPSO Neural Network Training

Este modulo proporciona funciones utilitarias para
procesamiento de datos, metricas y visualizacion.
"""

from .data import (
    load_dataset,
    train_test_split,
    normalize_data,
    create_dataloaders
)
from .metrics import (
    calculate_accuracy,
    calculate_confusion_matrix,
    calculate_classification_report,
    MulticlassMetrics,
    plot_confusion_matrix,
    plot_training_history,
    plot_loss_curves,
    plot_accuracy_curves,
    plot_fold_comparison,
    plot_complete_training_summary,
    plot_cv_summary
)

__all__ = [
    # Data utilities
    "load_dataset",
    "train_test_split",
    "normalize_data",
    "create_dataloaders",
    # Metrics
    "calculate_accuracy",
    "calculate_confusion_matrix",
    "calculate_classification_report",
    "MulticlassMetrics",
    # Plotting
    "plot_confusion_matrix",
    "plot_training_history",
    "plot_loss_curves",
    "plot_accuracy_curves",
    "plot_fold_comparison",
    "plot_complete_training_summary",
    "plot_cv_summary",
]
