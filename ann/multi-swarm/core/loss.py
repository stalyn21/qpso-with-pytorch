"""Pérdidas vectorizadas para entrenamiento de QDPSO en GPU."""
import torch


def cross_entropy_from_logits(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    dim_classes: int = -1,
) -> torch.Tensor:
    """Cross-entropy numéricamente estable desde logits (usa logsumexp).

    logits: (..., C)
    y_true: (...)  long  (índices de clase)
    Retorna: (...) con -log P(y_true | x) por muestra.
    """
    log_Z = torch.logsumexp(logits, dim=dim_classes)
    true_logits = torch.gather(
        logits, dim_classes, y_true.unsqueeze(dim_classes)
    ).squeeze(dim_classes)
    return log_Z - true_logits


def l2_penalty(
    params: torch.Tensor,
    lambda_l2: float,
    d_total: int,
    n_train: int,
) -> torch.Tensor:
    """L2 normalizado: (lambda / 2) * ||params||² / (d_total * n_train).

    Justificación de la normalización (convención Bishop / Goodfellow / MAP bayesiano):

    - Por `d_total` (tamaño total de la red): hace que `lambda` sea invariante al
      tamaño de la red. El mismo `lambda` regulariza con la misma intensidad por
      peso en redes pequeñas y grandes. Sin esta normalización, una red grande
      (ej. breast con d_total~3000) recibe un L2 efectivo ~30× más fuerte que
      una red pequeña (ej. iris con d_total~100) bajo el mismo `lambda`.

    - Por `n_train` (tamaño del dataset de entrenamiento): la cross-entropy se
      promedia sobre n_train, así que el L2 debe normalizarse de la misma forma
      para mantener un balance constante entre término de datos y prior. Es la
      convención del MAP bayesiano: si interpretamos el likelihood como Gaussian
      con varianza fija, `lambda` corresponde al inverso de la varianza del
      prior gaussiano sobre los pesos.

    Uso en BCD: `params` puede ser sólo el slice de una capa (en
    `build_layer_cost_fn`), pero `d_total` y `n_train` son los totales del
    modelo y dataset. La matemática del BCD lo permite porque las otras capas
    son constantes y aportan una contribución constante al L2 que no afecta
    el gradiente local.

    Si params es (P, d) retorna (P,); si (d,) retorna escalar.
    """
    if lambda_l2 <= 0:
        if params.dim() == 1:
            return torch.zeros((), dtype=params.dtype, device=params.device)
        return torch.zeros(params.shape[0], dtype=params.dtype, device=params.device)
    norm_factor = float(d_total * n_train)
    if params.dim() == 1:
        return (lambda_l2 / 2.0) * (params ** 2).sum() / norm_factor
    return (lambda_l2 / 2.0) * (params ** 2).sum(dim=-1) / norm_factor
