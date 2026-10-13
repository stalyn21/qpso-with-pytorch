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
