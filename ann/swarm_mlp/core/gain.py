"""Ganancia por capa: W_eff = s_k · W_k (`--layer-gain layer`).

Cada capa con ganancia guarda una coordenada más al final de su segmento:
`[W, b, u]`, con `s = exp(u)`. El enjambre de la capa busca en d + 1 dimensiones.

- Qué escala: sólo los pesos (W_eff = s·W); los sesgos no. Coincide con la L2
  `sample`, que penaliza sólo pesos, y con weight normalization (Salimans y Kingma,
  2016), que usa una ganancia por neurona; ésta es una por capa.
- Parametrización logarítmica: con u uniforme, encoger ×4 cuesta lo mismo que
  agrandar ×4. `u ∈ [ln s_min, ln s_max]` (`--gain-range`).
- Arranque: u = 0 (s = 1), la red sin ganancia.
- L2 sobre los pesos efectivos: s²·Σw². Es lo que le da agarre a la L2: una sola
  coordenada encoge la capa entera. Sin L2, el enjambre tiende a subir s, porque
  minimiza la CE de train (logits más grandes = más confianza).

Los modelos (`core/mlp.py`, `core/cnn.py`) usan estas funciones para armar sus
máscaras y para el índice de la ganancia de cada coordenada.
"""
import math
from typing import Sequence, Tuple

import torch

#: Modos de `--layer-gain`.
GAIN_MODES = ("none", "layer")


def log_gain_range(gain_range: Sequence[float]) -> Tuple[float, float]:
    """(ln s_min, ln s_max): la caja de la coordenada u."""
    s_min, s_max = float(gain_range[0]), float(gain_range[1])
    return math.log(s_min), math.log(s_max)


def validate_gain_range(gain_range: Sequence[float]) -> None:
    s_min, s_max = float(gain_range[0]), float(gain_range[1])
    if not 0 < s_min <= 1 <= s_max or s_min == s_max:
        raise ValueError(f"gain_range debe cumplir 0 < s_min <= 1 <= s_max y s_min < s_max "
                         f"(se dio {s_min}, {s_max}): el arranque es s = 1.")


def segment_masks(n_w: int, n_b: int, gain: bool, device) -> Tuple[torch.Tensor, ...]:
    """Máscaras bool (pesos, sesgos, ganancia) de un segmento `[W, b, (u)]`."""
    size = n_w + n_b + int(gain)
    w = torch.zeros(size, dtype=torch.bool, device=device)
    b = torch.zeros(size, dtype=torch.bool, device=device)
    g = torch.zeros(size, dtype=torch.bool, device=device)
    w[:n_w] = True
    b[n_w : n_w + n_b] = True
    if gain:
        g[-1] = True
    return w, b, g


def segment_gain_index(n_w: int, n_b: int, gain: bool, offset: int, device) -> torch.Tensor:
    """Por coordenada del segmento: índice (en el vector, con `offset`) de la u que la
    escala en la L2, o -1. Sólo los pesos de una capa con ganancia tienen índice."""
    size = n_w + n_b + int(gain)
    idx = torch.full((size,), -1, dtype=torch.long, device=device)
    if gain:
        idx[:n_w] = offset + size - 1
    return idx
