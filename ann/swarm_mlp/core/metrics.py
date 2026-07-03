"""Métricas de evaluación."""
import torch


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true == y_pred).sum().item() / y_true.numel()


def mse_int(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """MSE entre etiquetas enteras (consistente con el código original)."""
    return ((y_true.float() - y_pred.float()) ** 2).mean().item()


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int):
    """Matriz de confusión (n_classes, n_classes). Entrega lista de listas serializable."""
    cm = torch.zeros((n_classes, n_classes), dtype=torch.long, device=y_true.device)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm.cpu().tolist()
