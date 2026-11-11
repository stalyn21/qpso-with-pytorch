"""Límites de búsqueda de los enjambres, por capa.

Modos (`cfg["bounds_mode"]`, CLI `--bounds-mode`):
- ``fixed`` (default, comportamiento previo): todos los parámetros en
  ``cfg["bounds"]`` = (low, high), igual para todas las capas.
- ``fanin``: cada capa k busca en ``±bounds_scale / sqrt(fan_in_k)``, el rango
  de la inicialización uniforme de He (scale = √6) o de PyTorch (scale = 1).
  Pesos y sesgos de la capa comparten el rango. Evita que las capas con muchas
  entradas (p. ej. la segunda conv: 3·3·16 = 144) pasen casi toda la búsqueda
  en pesos que saturan o hacen explotar las activaciones.

fan_in = n_in en una densa (MLP o cabeza de la CNN) y n_in·k·k en una conv.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

BOUNDS_MODES = ("fixed", "fanin")


def layer_fan_in(model, k: int) -> int:
    spec = model.layer_specs[k]
    ksize = getattr(spec, "ksize", 0) or 1
    return int(spec.n_in * ksize * ksize)


def layer_range(model, cfg: Dict[str, Any], k: int) -> Tuple[float, float]:
    """(low, high) de la capa k según el modo de cfg."""
    mode = cfg.get("bounds_mode", "fixed")
    if mode == "fixed":
        low, high = cfg["bounds"]
        return float(low), float(high)
    if mode == "fanin":
        b = float(cfg["bounds_scale"]) / math.sqrt(layer_fan_in(model, k))
        return -b, b
    raise ValueError(f"bounds_mode desconocido: {mode!r}. Opciones: {BOUNDS_MODES}")


def dim_bounds(model, cfg: Dict[str, Any], k: Optional[int] = None) -> List[Tuple[float, float]]:
    """Lista (low, high) por dimensión: de la capa k, o del vector completo si k es None."""
    layers = range(model.num_layers) if k is None else [k]
    out: List[Tuple[float, float]] = []
    for j in layers:
        out.extend([layer_range(model, cfg, j)] * model.layer_specs[j].size)
    return out
