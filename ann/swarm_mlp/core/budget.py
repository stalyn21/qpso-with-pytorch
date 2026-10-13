"""Medidor del costo de una sesión: evaluaciones de la función de costo y traza.

El número de iteraciones NO es una medida de costo comparable entre técnicas:
T2 y T3 evalúan el enjambre dos veces por paso (re-evaluación de pbest), T1 corre
`max_iter` por capa, T4 mueve un solo enjambre. Lo que sí es comparable, y no
depende del hardware, es cuántas veces se evaluó la función de costo.

Unidad: **una evaluación = una partícula evaluada sobre todo el conjunto de
fitness** (un forward completo de la red). Evaluar un enjambre de P partículas
suma P.

Hay un único medidor por proceso (`METER`). `core.sessions.run_sessions` lo
reinicia al empezar cada sesión; `swarm.optimizer_factory` envuelve toda cost
function que entrega a un optimizador, y `core.metrics.selection_key` registra
cada evaluación de validación. Las técnicas no necesitan saber que existe.
"""
import threading
import time
from typing import Any, Callable, Dict, List

import torch


class RunMeter:
    """Contadores de una sesión. Seguro entre hilos (T2 evalúa desde varios)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.start()

    def start(self) -> None:
        with self._lock:
            self.fitness_evals = 0       # partículas evaluadas (ver unidad arriba)
            self.fitness_calls = 0       # llamadas a una cost function (enjambres evaluados)
            self.selection_evals = 0     # modelos evaluados en validación para seleccionar
            self.selection_time = 0.0    # segundos gastados en esas evaluaciones
            # Un punto por evaluación de validación:
            # [fitness_evals acumuladas, segundos desde start(), val_acc, val_ce]
            self.trace: List[List[float]] = []
            self._t0 = time.perf_counter()

    def count_fitness(self, n_particles: int) -> None:
        with self._lock:
            self.fitness_evals += int(n_particles)
            self.fitness_calls += 1

    def record_selection(self, val_acc: float, val_ce: float, seconds: float) -> None:
        with self._lock:
            self.selection_evals += 1
            self.selection_time += seconds
            self.trace.append([
                self.fitness_evals, time.perf_counter() - self._t0, float(val_acc), float(val_ce),
            ])

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "fitness_evals": self.fitness_evals,
                "fitness_calls": self.fitness_calls,
                "selection_evals": self.selection_evals,
                "selection_time": self.selection_time,
                "trace": [list(p) for p in self.trace],
            }


#: Medidor del proceso.
METER = RunMeter()

#: Columnas de cada punto de `trace`, para quien lea el run.json.
TRACE_COLUMNS = ("fitness_evals", "seconds", "val_acc", "val_ce")


def counted(cost_fn: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Envuelve `cost_fn` para que cada llamada sume sus partículas a `METER`."""

    def wrapper(positions: torch.Tensor) -> torch.Tensor:
        METER.count_fitness(positions.shape[0] if positions.dim() > 1 else 1)
        return cost_fn(positions)

    return wrapper


def evals_per_session(
    technique: str,
    max_iter: int,
    *,
    n_particles: int,
    n_layers: int,
    warm_particles: int,
    n_rounds: int = 1,
    reevaluate_pbest: bool = True,
) -> int:
    """Evaluaciones de fitness que gasta UNA sesión de `technique` con `max_iter`.

    Supone que ningún optimizador corta antes por paciencia (T1, T2-jacobi y T4
    pueden hacerlo; entonces gastan menos). P = partículas, K = partículas de
    warm-start, I = iteraciones, L = capas (enjambres), R = rondas:

        t4          P + K + I·P                      un enjambre
        t1          R · L · (P + K + I·P)            un enjambre nuevo por (ronda, capa)
        t2, t3      L · (P + K + c·I·P)              c = 2 con re-evaluación de pbest, 1 sin ella
        t2-jacobi   L · (P + K + (R-1)·2P + R·I·P)   2P por ronda extra: reuse + pbest re-medidos

    El `P + K` es la evaluación inicial del enjambre más la del cluster de warm-start.
    """
    P, I, L, R = int(n_particles), int(max_iter), int(n_layers), int(n_rounds)
    K = max(1, min(int(warm_particles), P))
    if technique == "t4":
        return P + K + I * P
    if technique == "t1":
        return R * L * (P + K + I * P)
    if technique in ("t2", "t3"):
        return L * (P + K + (2 if reevaluate_pbest else 1) * I * P)
    if technique == "t2-jacobi":
        return L * (P + K + (R - 1) * 2 * P + R * I * P)
    raise ValueError(f"Técnica no reconocida: {technique!r}")


def iters_for_budget(technique: str, eval_budget: int, **kwargs: Any) -> int:
    """Mayor `max_iter` cuya sesión no supera `eval_budget` evaluaciones de fitness.

    Es lo que permite comparar técnicas a IGUAL costo: se fija el presupuesto y cada
    técnica recibe las iteraciones que le alcanzan. `kwargs` son los de
    `evals_per_session`. El costo es lineal en `max_iter`, así que se despeja.
    """
    fixed = evals_per_session(technique, 0, **kwargs)
    per_iter = evals_per_session(technique, 1, **kwargs) - fixed
    iters = (int(eval_budget) - fixed) // per_iter
    if iters < 1:
        raise ValueError(
            f"El presupuesto de {eval_budget} evaluaciones no alcanza para una iteración "
            f"de {technique} (mínimo {fixed + per_iter})."
        )
    return int(iters)
