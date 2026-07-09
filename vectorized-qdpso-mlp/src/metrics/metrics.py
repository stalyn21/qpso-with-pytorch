import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import logging

class MulticlassMetrics:
    def __init__(self):
        # Configurar LaTeX para todas las visualizaciones
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}"
        })

    def confusion_matrix(self, y_true, y_pred, labels=None):
        """
        Calcula la matriz de confusión para clasificación multiclase.

        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            labels: Lista de etiquetas de clase

        Returns:
            numpy.ndarray: Matriz de confusión
        """
        return confusion_matrix(y_true, y_pred, labels=labels)

    def accuracy(self, y_true, y_pred):
        """
        Calcula la exactitud global para clasificación multiclase.

        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo

        Returns:
            float: Score de exactitud
        """
        return accuracy_score(y_true, y_pred)

    def precision(self, y_true, y_pred, average='macro'):
        """
        Calcula la precisión para clasificación multiclase.

        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            average: Método de promedio ('macro', 'micro', 'weighted', None)

        Returns:
            float o array: Scores de precisión
        """
        return precision_score(y_true, y_pred, average=average)

    def recall(self, y_true, y_pred, average='macro'):
        """
        Calcula el recall para clasificación multiclase.

        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            average: Método de promedio ('macro', 'micro', 'weighted', None)

        Returns:
            float o array: Scores de recall
        """
        return recall_score(y_true, y_pred, average=average)

    def f1(self, y_true, y_pred, average='macro'):
        """
        Calcula el F1-score para clasificación multiclase.

        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            average: Método de promedio ('macro', 'micro', 'weighted', None)

        Returns:
            float o array: F1-scores
        """
        return f1_score(y_true, y_pred, average=average)

    def calculate_all_metrics(self, y_true, y_pred):
        """
        Calcula todas las métricas básicas para clasificación multiclase.

        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo

        Returns:
            dict: Diccionario con todas las métricas
        """
        metrics = {
            'accuracy': self.accuracy(y_true, y_pred),
            'precision_macro': self.precision(y_true, y_pred, 'macro'),
            'precision_weighted': self.precision(y_true, y_pred, 'weighted'),
            'recall_macro': self.recall(y_true, y_pred, 'macro'),
            'recall_weighted': self.recall(y_true, y_pred, 'weighted'),
            'f1_macro': self.f1(y_true, y_pred, 'macro'),
            'f1_weighted': self.f1(y_true, y_pred, 'weighted'),
            'confusion_matrix': self.confusion_matrix(y_true, y_pred)
        }
        return metrics

    def plot_multiclass_roc(self, classifier, X, y, n_classes, dataset_name, optimizer, f_reduction='org'):
        """
        Calcula y visualiza curvas ROC para clasificación binaria y multiclase con información detallada.
        """
        plt.figure(figsize=(12, 8))

        # Definir colores y estilos
        colors = ['#FF69B4', '#4169E1', '#32CD32', '#FF7F50', '#9370DB']
        line_styles = ['-', '--', '-.', ':', '-']
        legend_elements = []

        # Para clasificación binaria
        if n_classes == 2:
            y_score = classifier.predict_proba(X)

            # Para clase 0
            fpr_0, tpr_0, _ = roc_curve(y, y_score[:, 0], pos_label=0)
            roc_auc_0 = auc(fpr_0, tpr_0)

            # Para clase 1
            fpr_1, tpr_1, _ = roc_curve(y, y_score[:, 1], pos_label=1)
            roc_auc_1 = auc(fpr_1, tpr_1)

            # Plotear curvas ROC individuales para ambas clases
            plt.plot(fpr_0, tpr_0,
                    color=colors[0],
                    linestyle=line_styles[0],
                    linewidth=2)

            plt.plot(fpr_1, tpr_1,
                    color=colors[1],
                    linestyle=line_styles[1],
                    linewidth=2)

            # Calcular micro y macro promedios
            roc_auc_micro = (roc_auc_0 + roc_auc_1) / 2
            roc_auc_macro = roc_auc_micro  # En el caso binario, micro y macro son iguales

            # Añadir micro promedio
            plt.plot(fpr_1, tpr_1,  # Usamos cualquiera de las curvas ya que son complementarias
                    color='deeppink', linestyle=':', linewidth=3)
            
            # Añadir macro promedio
            plt.plot(fpr_1, tpr_1,  # Usamos cualquiera de las curvas ya que son complementarias
                    color='navy', linestyle=':', linewidth=3)
            
            # Añadir leyendas
            legend_elements.append(
                plt.Line2D([0], [0], color='deeppink', linestyle=':', linewidth=3,
                        label=f'Micro-average ROC (AUC = {roc_auc_micro:.3f})')
            )
            legend_elements.append(
                plt.Line2D([0], [0], color='navy', linestyle=':', linewidth=3,
                        label=f'Macro-average ROC (AUC = {roc_auc_macro:.3f})')
            )
            legend_elements.append(
                plt.Line2D([0], [0], color='none',
                        label=f'Number of classes: {n_classes}')
            )
            legend_elements.append(
                plt.Line2D([0], [0], color=colors[0], linestyle=line_styles[0], linewidth=2,
                        label=f'Class 0 (AUC = {roc_auc_0:.3f})')
            )
            legend_elements.append(
                plt.Line2D([0], [0], color=colors[1], linestyle=line_styles[1], linewidth=2,
                        label=f'Class 1 (AUC = {roc_auc_1:.3f})')
            )

            roc_auc_dict = {
                0: roc_auc_0,
                1: roc_auc_1,
                'micro': roc_auc_micro,
                'macro': roc_auc_macro
            }

        else:  # Para clasificación multiclase
            y_bin = label_binarize(y, classes=range(n_classes))
            y_score = classifier.predict_proba(X)

            # Calcular ROC y AUC para cada clase
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i],
                        color=colors[i % len(colors)],
                        linestyle=line_styles[i % len(line_styles)],
                        linewidth=2)

            # Calcular y plotear micro-promedio
            fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            plt.plot(fpr["micro"], tpr["micro"],
                    color='deeppink', linestyle=':', linewidth=3)

            # Calcular y plotear macro-promedio
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            plt.plot(fpr["macro"], tpr["macro"],
                    color='navy', linestyle=':', linewidth=3)

            # Añadir leyendas
            legend_elements.append(
                plt.Line2D([0], [0], color='deeppink', linestyle=':', linewidth=3,
                        label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})')
            )
            legend_elements.append(
                plt.Line2D([0], [0], color='navy', linestyle=':', linewidth=3,
                        label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.3f})')
            )
            legend_elements.append(
                plt.Line2D([0], [0], color='none',
                        label=f'Number of classes: {n_classes}')
            )
            for i in range(n_classes):
                legend_elements.append(
                    plt.Line2D([0], [0], color=colors[i % len(colors)],
                            linestyle=line_styles[i % len(line_styles)], linewidth=2,
                            label=f'Class {i} (AUC = {roc_auc[i]:.3f})')
                )

            roc_auc_dict = roc_auc

        # Línea diagonal de referencia
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        legend_elements.append(
            plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.3,
                    label='Random (AUC = 0.5)')
        )

        """
        if n_classes == 2:
            legend_elements.append(
                plt.Line2D([0], [0], color='none',
                        label=f'Micro-AUC: {roc_auc_micro:.3f}') 
            )
            legend_elements.append(
                plt.Line2D([0], [0], color='none',
                        label=f'Macro-AUC: {roc_auc_macro:.3f}')  
            )
        else:
            legend_elements.append(
                plt.Line2D([0], [0], color='none',
                        label=f'Micro-AUC: {roc_auc["micro"]:.3f}')
            )
            legend_elements.append(
                plt.Line2D([0], [0], color='none',
                        label=f'Macro-AUC: {roc_auc["macro"]:.3f}')
            )
        """

        # Configuración del gráfico
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        # plt.title(f'ROC Curves Analysis - {dataset_name} Dataset\n{optimizer} Optimizer',
                # fontsize=14, pad=20)

        # Configurar leyenda mejorada
        plt.legend(handles=legend_elements,
                loc="lower right",
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=0.9,
                fontsize=10)

        plt.grid(True, linestyle='--', alpha=0.3)

        # Ajustar el diseño y guardar
        plt.tight_layout()
        plt.savefig(f'./metrics/graphics/roc/{optimizer}/{dataset_name}_roc_curves_{optimizer}_{f_reduction}.png',
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor='white',
                    edgecolor='none')
        plt.close()

        return roc_auc_dict

    def log_metrics(self, metrics, dataset_name, fold=None):
        """
        Registra las métricas calculadas usando logging.

        Args:
            metrics: Diccionario con las métricas
            dataset_name: Nombre del dataset
            fold: Número de fold (opcional)
        """
        fold_str = f" - Fold {fold}" if fold is not None else ""
        logging.info(f"\nMétricas para {dataset_name}{fold_str}:")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
        logging.info(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        logging.info(f"Recall (macro): {metrics['recall_macro']:.4f}")
        logging.info(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
        logging.info(f"F1 (macro): {metrics['f1_macro']:.4f}")
        logging.info(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
        logging.info("Confusion Matrix:")
        logging.info("\n" + str(metrics['confusion_matrix']))