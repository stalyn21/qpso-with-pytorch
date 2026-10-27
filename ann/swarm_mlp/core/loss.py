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


#: Formas de la penalización L2 (flag `--l2-form` del runner).
#:   "mean"   (lambda / 2n) · media(θ²) sobre TODOS los parámetros (pesos y sesgos).
#:            Es la forma histórica del repo: (lambda/2)·||θ||² / (d_total · n_train).
#:   "sample" (lambda / 2n) · Σ w² sobre los PESOS, sin los sesgos. Es la weight decay
#:            estándar normalizada por el tamaño de la muestra (Ng, CS229; Nielsen,
#:            "Neural Networks and Deep Learning", cap. 3; Goodfellow et al., § 7.1.1).
#:            Los sesgos no se penalizan porque no interactúan multiplicativamente con
#:            la entrada: penalizarlos no reduce la varianza y sí puede sub-ajustar.
L2_FORMS = ("mean", "sample")


def l2_penalty(
    params: torch.Tensor,
    lambda_l2: float,
    d_total: int,
    n_train: int,
    form: str = "mean",
    weight_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Penalización L2. `form` elige la normalización (ver `L2_FORMS`).

    - form="mean":   (lambda / 2) * ||params||² / (d_total * n_train)
    - form="sample": (lambda / 2) * Σ (params[weight_mask])² / n_train
      `weight_mask` (bool, misma longitud que la última dimensión de `params`) marca
      qué posiciones son pesos; si es None se penalizan todas.

    En "sample" la intensidad por peso no depende del tamaño de la red: una red más
    grande recibe más penalización total, como en la weight decay usual. En "mean"
    la penalización total queda invariante al tamaño de la red. Con el mismo lambda
    "sample" es ~d_total veces más fuerte, así que los lambdas NO son comparables.

    Justificación histórica de la forma "mean":

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
    if form not in L2_FORMS:
        raise ValueError(f"form debe ser uno de {L2_FORMS}, se dio {form!r}.")
    if lambda_l2 <= 0:
        if params.dim() == 1:
            return torch.zeros((), dtype=params.dtype, device=params.device)
        return torch.zeros(params.shape[0], dtype=params.dtype, device=params.device)
    sq = params ** 2
    if form == "sample":
        if weight_mask is not None:
            sq = sq * weight_mask.to(dtype=params.dtype, device=params.device)
        norm_factor = float(n_train)
    else:
        norm_factor = float(d_total * n_train)
    return (lambda_l2 / 2.0) * sq.sum(dim=-1) / norm_factor
