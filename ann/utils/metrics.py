"""
Metricas de Evaluacion

Funciones para calcular metricas de clasificacion
para evaluar modelos entrenados con QPSO.
"""

import numpy as np
import torch
from typing import Dict, List, Union, Optional
from dataclasses import dataclass


def calculate_accuracy(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Calcula la accuracy de clasificacion.

    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones

    Returns:
        Accuracy como float
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return np.mean(y_true == y_pred)


def calculate_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> np.ndarray:
    """
    Calcula la matriz de confusion.

    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones

    Returns:
        Matriz de confusion como numpy array
    """
    from sklearn.metrics import confusion_matrix

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return confusion_matrix(y_true, y_pred)


def calculate_classification_report(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    target_names: Optional[List[str]] = None,
    output_dict: bool = True
) -> Union[str, Dict]:
    """
    Genera reporte de clasificacion.

    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones
        target_names: Nombres de las clases
        output_dict: Si True, retorna diccionario

    Returns:
        Reporte como string o diccionario
    """
    from sklearn.metrics import classification_report

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=output_dict
    )


@dataclass
class ClassMetrics:
    """Metricas por clase."""
    precision: float
    recall: float
    f1_score: float
    support: int


class MulticlassMetrics:
    """
    Calculador de metricas para clasificacion multiclase.

    Proporciona metricas detalladas incluyendo:
    - Accuracy
    - Precision, Recall, F1 por clase
    - Macro y Weighted averages
    - Matriz de confusion

    Example:
        >>> metrics = MulticlassMetrics()
        >>> results = metrics.calculate_all_metrics(y_true, y_pred)
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Inicializa el calculador de metricas.

        Args:
            class_names: Nombres de las clases (opcional)
        """
        self.class_names = class_names

    def calculate_all_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor]
    ) -> Dict:
        """
        Calcula todas las metricas de clasificacion.

        Args:
            y_true: Labels verdaderos
            y_pred: Predicciones

        Returns:
            Diccionario con todas las metricas
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            cohen_kappa_score,
            matthews_corrcoef
        )

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        # Metricas basicas
        accuracy = accuracy_score(y_true, y_pred)

        # Metricas por clase
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Metricas adicionales
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        # Matriz de confusion
        conf_matrix = calculate_confusion_matrix(y_true, y_pred)

        # Reporte por clase
        class_report = calculate_classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        return {
            'accuracy': accuracy,
            'precision': {
                'macro': precision_macro,
                'weighted': precision_weighted
            },
            'recall': {
                'macro': recall_macro,
                'weighted': recall_weighted
            },
            'f1_score': {
                'macro': f1_macro,
                'weighted': f1_weighted
            },
            'cohen_kappa': kappa,
            'matthews_corrcoef': mcc,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }

    def print_summary(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """
        Imprime un resumen de las metricas.

        Args:
            y_true: Labels verdaderos
            y_pred: Predicciones
        """
        metrics = self.calculate_all_metrics(y_true, y_pred)

        print("=" * 50)
        print("METRICAS DE CLASIFICACION")
        print("=" * 50)
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Precision (macro):  {metrics['precision']['macro']:.4f}")
        print(f"Recall (macro):     {metrics['recall']['macro']:.4f}")
        print(f"F1-Score (macro):   {metrics['f1_score']['macro']:.4f}")
        print(f"Cohen's Kappa:      {metrics['cohen_kappa']:.4f}")
        print(f"MCC:                {metrics['matthews_corrcoef']:.4f}")
        print("-" * 50)
        print("Matriz de Confusion:")
        print(metrics['confusion_matrix'])
        print("=" * 50)


def plot_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6)
) -> None:
    """
    Grafica la matriz de confusion.

    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones
        class_names: Nombres de las clases
        save_path: Ruta para guardar la figura
        figsize: Tamano de la figura
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib y seaborn requeridos para graficas")
        return

    conf_matrix = calculate_confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Prediccion')
    plt.ylabel('Real')
    plt.title('Matriz de Confusion')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figura guardada en: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    history: Dict[str, List],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4)
) -> None:
    """
    Grafica el historial de entrenamiento.

    Args:
        history: Diccionario con historial (train_loss, val_loss, etc.)
        save_path: Ruta para guardar la figura
        figsize: Tamano de la figura
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib requerido para graficas")
        return

    n_plots = 0
    if 'train_loss' in history:
        n_plots += 1
    if 'train_acc' in history:
        n_plots += 1

    if n_plots == 0:
        print("No hay datos en el historial")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot loss
    if 'train_loss' in history:
        axes[plot_idx].plot(history['train_loss'], label='Train')
        if 'val_loss' in history and len(history['val_loss']) > 0:
            axes[plot_idx].plot(history['val_loss'], label='Validation')
        axes[plot_idx].set_xlabel('Iteracion')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Curva de Loss')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

    # Plot accuracy
    if 'train_acc' in history:
        axes[plot_idx].plot(history['train_acc'], label='Train')
        if 'val_acc' in history and len(history['val_acc']) > 0:
            axes[plot_idx].plot(history['val_acc'], label='Validation')
        axes[plot_idx].set_xlabel('Iteracion')
        axes[plot_idx].set_ylabel('Accuracy')
        axes[plot_idx].set_title('Curva de Accuracy')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figura guardada en: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_loss_curves(
    history: Dict[str, List],
    title: str = "Curvas de Loss",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Grafica las curvas de loss (train y validation).

    Args:
        history: Diccionario con train_loss y val_loss
        title: Titulo de la grafica
        save_path: Ruta para guardar la figura
        figsize: Tamano de la figura
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib requerido para graficas")
        return

    if 'train_loss' not in history or len(history['train_loss']) == 0:
        print("No hay datos de loss en el historial")
        return

    plt.figure(figsize=figsize)

    iterations = range(1, len(history['train_loss']) + 1)
    plt.plot(iterations, history['train_loss'], 'b-', label='Train Loss', linewidth=2)

    if 'val_loss' in history and len(history['val_loss']) > 0:
        plt.plot(iterations, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Iteracion', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Marcar mejor punto
    min_loss = min(history['train_loss'])
    min_iter = history['train_loss'].index(min_loss) + 1
    plt.axvline(x=min_iter, color='g', linestyle='--', alpha=0.5, label=f'Mejor iter: {min_iter}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_accuracy_curves(
    history: Dict[str, List],
    title: str = "Curvas de Accuracy",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Grafica las curvas de accuracy (train y validation).

    Args:
        history: Diccionario con train_acc y val_acc
        title: Titulo de la grafica
        save_path: Ruta para guardar la figura
        figsize: Tamano de la figura
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib requerido para graficas")
        return

    if 'train_acc' not in history or len(history['train_acc']) == 0:
        print("No hay datos de accuracy en el historial")
        return

    plt.figure(figsize=figsize)

    iterations = range(1, len(history['train_acc']) + 1)
    plt.plot(iterations, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)

    if 'val_acc' in history and len(history['val_acc']) > 0:
        plt.plot(iterations, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)

    plt.xlabel('Iteracion', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    # Marcar mejor punto
    max_acc = max(history['train_acc'])
    max_iter = history['train_acc'].index(max_acc) + 1
    plt.axvline(x=max_iter, color='g', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_fold_comparison(
    fold_results: List[Dict],
    metric: str = 'val_acc',
    title: str = "Comparacion por Fold",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Grafica comparacion de metricas por fold en cross-validation.

    Args:
        fold_results: Lista de diccionarios con resultados por fold
        metric: Metrica a graficar ('val_acc', 'train_acc', 'val_loss', 'train_loss')
        title: Titulo de la grafica
        save_path: Ruta para guardar la figura
        figsize: Tamano de la figura
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib y numpy requeridos para graficas")
        return

    if not fold_results:
        print("No hay resultados de folds")
        return

    plt.figure(figsize=figsize)

    folds = [r['fold'] for r in fold_results]
    values = [r[metric] for r in fold_results]
    mean_val = np.mean(values)
    std_val = np.std(values)

    # Colores por tipo de metrica
    color = 'steelblue' if 'acc' in metric else 'coral'

    bars = plt.bar(folds, values, color=color, alpha=0.7, edgecolor='black')
    plt.axhline(y=mean_val, color='red', linestyle='--', linewidth=2,
                label=f'Media: {mean_val:.4f} +/- {std_val:.4f}')

    plt.xlabel('Fold', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, axis='y', alpha=0.3)

    # Etiquetas en barras
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_complete_training_summary(
    history: Dict[str, List],
    train_metrics: Dict[str, float],
    val_metrics: Optional[Dict[str, float]] = None,
    test_metrics: Optional[Dict[str, float]] = None,
    title: str = "Resumen de Entrenamiento",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10)
) -> None:
    """
    Genera un resumen visual completo del entrenamiento.

    Incluye:
    - Curvas de loss (train/val)
    - Curvas de accuracy (train/val)
    - Barras comparativas de metricas finales (train/val/test)

    Args:
        history: Diccionario con historial de entrenamiento
        train_metrics: Metricas finales de train {loss, accuracy}
        val_metrics: Metricas finales de validacion
        test_metrics: Metricas finales de test
        title: Titulo principal
        save_path: Ruta para guardar la figura
        figsize: Tamano de la figura
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib y numpy requeridos para graficas")
        return

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. Curva de Loss
    ax1 = axes[0, 0]
    if 'train_loss' in history and len(history['train_loss']) > 0:
        iters = range(1, len(history['train_loss']) + 1)
        ax1.plot(iters, history['train_loss'], 'b-', label='Train', linewidth=2)
        if 'val_loss' in history and len(history['val_loss']) > 0:
            ax1.plot(iters, history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax1.set_xlabel('Iteracion')
        ax1.set_ylabel('Loss')
        ax1.set_title('Curvas de Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Sin datos de loss', ha='center', va='center')
        ax1.set_title('Curvas de Loss')

    # 2. Curva de Accuracy
    ax2 = axes[0, 1]
    if 'train_acc' in history and len(history['train_acc']) > 0:
        iters = range(1, len(history['train_acc']) + 1)
        ax2.plot(iters, history['train_acc'], 'b-', label='Train', linewidth=2)
        if 'val_acc' in history and len(history['val_acc']) > 0:
            ax2.plot(iters, history['val_acc'], 'r-', label='Validation', linewidth=2)
        ax2.set_xlabel('Iteracion')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Curvas de Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
    else:
        ax2.text(0.5, 0.5, 'Sin datos de accuracy', ha='center', va='center')
        ax2.set_title('Curvas de Accuracy')

    # 3. Barras de Loss final
    ax3 = axes[1, 0]
    labels = ['Train']
    losses = [train_metrics.get('loss', 0)]
    colors = ['steelblue']

    if val_metrics:
        labels.append('Validation')
        losses.append(val_metrics.get('loss', 0))
        colors.append('coral')

    if test_metrics:
        labels.append('Test')
        losses.append(test_metrics.get('loss', 0))
        colors.append('seagreen')

    bars = ax3.bar(labels, losses, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Loss')
    ax3.set_title('Loss Final por Conjunto')
    ax3.grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars, losses):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # 4. Barras de Accuracy final
    ax4 = axes[1, 1]
    labels = ['Train']
    accs = [train_metrics.get('accuracy', 0)]
    colors = ['steelblue']

    if val_metrics:
        labels.append('Validation')
        accs.append(val_metrics.get('accuracy', 0))
        colors.append('coral')

    if test_metrics:
        labels.append('Test')
        accs.append(test_metrics.get('accuracy', 0))
        colors.append('seagreen')

    bars = ax4.bar(labels, accs, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy Final por Conjunto')
    ax4.grid(True, axis='y', alpha=0.3)
    ax4.set_ylim(0, 1.1)

    for bar, val in zip(bars, accs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_cv_summary(
    fold_results: List[Dict],
    title: str = "Resumen Cross-Validation",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6)
) -> None:
    """
    Genera resumen visual de cross-validation.

    Args:
        fold_results: Lista de resultados por fold
        title: Titulo de la grafica
        save_path: Ruta para guardar
        figsize: Tamano de la figura
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib y numpy requeridos para graficas")
        return

    if not fold_results:
        print("No hay resultados de folds")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    folds = [r['fold'] for r in fold_results]
    n_folds = len(folds)
    x = np.arange(n_folds)
    width = 0.35

    # 1. Accuracy por fold
    ax1 = axes[0]
    train_accs = [r['train_acc'] for r in fold_results]
    val_accs = [r['val_acc'] for r in fold_results]

    bars1 = ax1.bar(x - width/2, train_accs, width, label='Train', color='steelblue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, val_accs, width, label='Validation', color='coral', alpha=0.7)

    ax1.axhline(y=np.mean(val_accs), color='red', linestyle='--',
                label=f'Val Media: {np.mean(val_accs):.4f}')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy por Fold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Fold {f}' for f in folds])
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, axis='y', alpha=0.3)

    # 2. Loss por fold
    ax2 = axes[1]
    train_losses = [r['train_loss'] for r in fold_results]
    val_losses = [r['val_loss'] for r in fold_results]

    bars3 = ax2.bar(x - width/2, train_losses, width, label='Train', color='steelblue', alpha=0.7)
    bars4 = ax2.bar(x + width/2, val_losses, width, label='Validation', color='coral', alpha=0.7)

    ax2.axhline(y=np.mean(val_losses), color='red', linestyle='--',
                label=f'Val Media: {np.mean(val_losses):.4f}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss por Fold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Fold {f}' for f in folds])
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    else:
        plt.show()

    plt.close()
