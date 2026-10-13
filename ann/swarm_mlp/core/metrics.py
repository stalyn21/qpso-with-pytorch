"""Métricas de evaluación y criterio de selección de modelos."""
import time
from typing import Dict, Tuple

import torch

from core.budget import METER
from core.loss import cross_entropy_from_logits

#: Nombre del criterio, guardado en run.json para distinguir corridas.
SELECTION_CRITERION = "val_acc>val_ce"


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true == y_pred).sum().item() / y_true.numel()


def mse_int(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """MSE entre etiquetas enteras (consistente con el código original).

    Sólo reporte. NO se usa para seleccionar: trata las clases como ordinales
    (confundir la clase 0 con la 9 pesa 81 veces más que con la 1), lo que no
    tiene sentido con clases nominales.
    """
    return ((y_true.float() - y_pred.float()) ** 2).mean().item()


def acc_and_ce(model, flat_params: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    """(accuracy, cross-entropy media) de `flat_params` sobre (X, y), con un solo forward."""
    with torch.no_grad():
        logits = model.forward(X, flat_params)
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        ce = cross_entropy_from_logits(logits, y, dim_classes=-1).mean().item()
    return acc, ce


def selection_key(model, flat_params: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor) -> Tuple[float, float]:
    """Criterio ÚNICO de selección de modelos, en las técnicas y entre sesiones.

    Devuelve `(1 - val_acc, val_ce)`; las claves se comparan como tuplas y gana
    la MENOR: primero más accuracy de validación y, con la misma accuracy, menos
    cross-entropy de validación. La CE es continua, así que deshace los empates
    de accuracy (frecuentes con pocas muestras de val) sin recurrir al train.
    """
    t0 = time.perf_counter()
    acc, ce = acc_and_ce(model, flat_params, X_val, y_val)   # los .item() sincronizan la GPU
    METER.record_selection(acc, ce, time.perf_counter() - t0)
    return (1.0 - acc, ce)


def classification_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int) -> Dict[str, float]:
    """Accuracy balanceada y F1 macro, ambas promediadas sobre las `n_classes` clases.

    - `balanced_acc`: media del recall por clase (no premia acertar sólo la mayoritaria).
    - `macro_f1`: media del F1 por clase; una clase nunca predicha aporta F1 = 0.
    """
    idx = y_true * n_classes + y_pred
    cm = torch.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes).double()
    tp = cm.diag()
    support, predicted = cm.sum(dim=1), cm.sum(dim=0)
    recall = tp / support.clamp(min=1)
    f1 = 2 * tp / (support + predicted).clamp(min=1)
    present = support > 0   # clases sin muestras en este split no entran al promedio
    return {
        "balanced_acc": float(recall[present].mean().item()),
        "macro_f1": float(f1[present].mean().item()),
    }


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int):
    """Matriz de confusión (n_classes, n_classes). Entrega lista de listas serializable."""
    cm = torch.zeros((n_classes, n_classes), dtype=torch.long, device=y_true.device)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm.cpu().tolist()
