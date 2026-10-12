"""Factory compartida del optimizador swarm (eje 3 del marco: QPSOo vs QDPSOo).

Las cinco técnicas (T1, T2, T2-jacobi, T3, T4) construyen su optimizador por aquí.
Los algoritmos vienen del paquete `qpso-torch` (>= 0.1.0); este módulo sólo agrega
el eje del paper:

    - **QDPSOo** (`qpso_torch.QDPSOTensor`): factor de contracción-expansión ESTÁTICO `g`.
    - **QPSOo**  (`qpso_torch.QPSOTensor`):  factor CE CALCULADO, `alpha` con decaimiento
      lineal `alpha(t) = alpha_max - (alpha_max - alpha_min) * t / T`.

Pregunta de investigación: ¿vale la pena calcular el factor CE, o basta uno estático
bien tuneado? Si QDPSOo resulta muy sensible a `g`, QPSOo gana en robustez aunque
empate en accuracy.

Ambas clases derivan de `qpso_torch.QPSOBaseTensor`, así que exponen la misma API
pública que usan las técnicas (`optimize`, `step`, `reuse_with`, `release`,
`callbacks`, `gbest`, ...). El intercambio es transparente.

⚠️ **`iters` / `max_iters` son semánticos bajo QPSOo.** Con QDPSOo (`g` estático)
nadie los mira; con QPSOo son el `t` y el `T` del decaimiento de alpha. Implicaciones
por técnica:

  - T1  → optimizador nuevo por (round, capa) + `optimize()`: decay 0→max_iter por capa.
  - T2  → optimizador persistente, un `step()` por iteración: decay 0→max_iter en la sesión.
  - T3  → ídem T2.
  - T4  → un optimizador + `optimize()`: decay 0→max_iter.
  - T2-jacobi → `reuse_with()` resetea el contador, así que alpha decae EN SIERRA:
          un decay completo por round. Es intencional (decisión 2026-07-20): cada
          round estrena cost function, o sea un subproblema nuevo, y re-explorar con
          alpha alto al empezarlo es coherente. Bajo QDPSOo no hay diferencia.

`max_iters` es un presupuesto EXACTO en qpso-torch (el `tensor_qpso` anterior corría
`maxIters + 1` iteraciones en `optimize()`).
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from qpso_torch import QDPSOTensor, QPSOBaseTensor, QPSOTensor
from qpso_torch.cost import CostFunction

#: Identificadores aceptados por `--optimizer`. La nomenclatura ("QPSOo"/"QDPSOo")
#: es la del paper-1, mantenida por consistencia.
OPTIMIZERS = ("qdpsoo", "qpsoo")

#: Tipo de retorno de la factory. Las técnicas que guardan optimizadores persistentes
#: (T2 async, T2-jacobi) anotan con esto en vez de con una clase concreta.
SwarmOptimizer = QPSOBaseTensor

#: Default del paper para el factor CE estático de QDPSOo.
#: ⚠️ El default histórico del runner era 0.96; los resultados preliminares 1-seed
#: se corrieron con ese valor y NO son válidos para el paper. Candidato a re-tuneo:
#: el análisis de sensibilidad barre g ∈ {1.13, 1.20, 1.30, 1.40, 1.50}.
DEFAULT_G = 1.13

#: Default de QPSOo: decaimiento lineal (max, min) del factor CE calculado.
DEFAULT_ALPHA: Tuple[float, float] = (1.0, 0.5)


def build_swarm_optimizer(
    cfg: Dict[str, Any],
    *,
    cf,
    dim: int,
    bounds: List[Tuple[float, float]],
    seed: int,
    dtype: torch.dtype,
) -> SwarmOptimizer:
    """Construye el optimizador swarm que indique `cfg["optimizer"]`.

    Los parámetros que varían por técnica y por capa (`cf`, `dim`, `bounds`, `seed`)
    van explícitos; el resto sale de `cfg`. `dtype` viene del MLP, no de cfg, porque
    es el dtype ya resuelto de los tensores.

    Todas las técnicas usan `track_history=False` y `minimize=True`; ninguna necesita
    lo contrario, así que quedan fijos aquí en vez de repetirse cinco veces.
    `vectorized=True` también: las cost functions de `core/mlp.py` evalúan el swarm
    completo en una llamada, y declararlo evita la detección (ambigua si dim == P).
    """
    name = str(cfg.get("optimizer", "qdpsoo")).lower()

    common = dict(
        cost_fn=cf,
        n_particles=cfg["n_particles"],
        dim=dim,
        bounds=bounds,
        max_iters=cfg["max_iter"],
        device=cfg["device"],
        dtype=dtype,
        seed=seed,
        boundary_strategy=cfg["boundary_strategy"],
        tol=cfg["tol"],
        patience=cfg["patience"],
        track_history=False,
        minimize=True,
        vectorized=True,
    )

    if name == "qdpsoo":
        return QDPSOTensor(g=float(cfg.get("g", DEFAULT_G)), **common)

    if name == "qpsoo":
        return QPSOTensor(alpha=resolve_alpha(cfg.get("alpha")), **common)

    raise ValueError(
        f"Optimizador no reconocido: {name!r}. Use uno de {OPTIMIZERS}."
    )


def swap_cost_fn(opt: SwarmOptimizer, cf: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """Cambia la cost function de `opt` SIN tocar nada más (T2 async y T3).

    Esas técnicas inyectan una cost function fresca en cada paso (el contexto de las
    otras capas cambió) y necesitan conservar pbest y el contador de iteraciones —
    `reuse_with()` resetea ambos, así que no sirve aquí.

    qpso-torch 0.1.0 no tiene API pública para esto: es el ÚNICO punto del paquete
    que escribe un atributo privado del optimizador (`_cost`). Cuando la librería
    exponga un equivalente, reemplazar el cuerpo de esta función.

    ⚠️ Los `pbest_values` NO se re-evalúan contra la nueva cost function (mismo
    comportamiento que antes de la migración): quedan medidos contra contextos viejos.
    """
    opt._cost = CostFunction(
        cf,
        dtype=opt.dtype,
        device=opt.device,
        minimize=opt.minimize,
        vectorized=True,
    )


def resolve_alpha(
    alpha: Optional[Union[float, List[float], Tuple[float, ...]]],
) -> Union[float, Tuple[float, float]]:
    """Normaliza `alpha` a lo que espera `QPSOTensor`.

    Acepta `None` (default del paper), un escalar (alpha constante, sin decaimiento)
    o un par `(max, min)` — que es como llega desde argparse (`nargs=2` → lista).
    """
    if alpha is None:
        return DEFAULT_ALPHA
    if isinstance(alpha, (int, float)):
        return float(alpha)
    values = tuple(float(v) for v in alpha)
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return values
    raise ValueError(
        f"alpha debe ser un escalar o un par (max, min); recibido: {alpha!r}"
    )


def optimizer_tag(cfg: Dict[str, Any]) -> str:
    """Etiqueta corta del optimizador para los nombres de directorio de output.

    Va SIEMPRE en el tag (no sólo cuando difiere del default) para que las 16 celdas
    del factorial tengan paths inequívocos y no se sobrescriban entre sí.
    """
    return str(cfg.get("optimizer", "qdpsoo")).lower()
