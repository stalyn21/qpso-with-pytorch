"""Warm-start cluster compartido por las cinco técnicas.

Reemplaza las primeras K partículas del swarm con perturbaciones alrededor de un
vector semilla. Mantiene el resto random para preservar exploración.

Sólo usa la API pública de `qpso_torch.QPSOBaseTensor` (las vistas `positions`,
`pbest` y `pbest_values` son vivas: se escriben in-place), así que funciona igual
con QDPSOo y con QPSOo.
"""
import torch

from swarm.optimizer_factory import SwarmOptimizer

#: Desplazamiento para derivar la semilla del ruido a partir de la del optimizador.
#: Evita reusar el mismo flujo que generó las posiciones iniciales del swarm.
_NOISE_SEED_OFFSET = 0x9E3779B1


def warm_start_cluster(
    opt: SwarmOptimizer,
    seed_vec: torch.Tensor,
    k_seeded: int,
    noise_scale: float,
) -> None:
    """Reemplaza las primeras K partículas con perturbaciones alrededor de `seed_vec`.

    - Partícula 0: seed exacto.
    - Partículas 1..K-1: seed + ruido gaussiano de escala `noise_scale * (high - low)`.
    - Partículas K..P-1: quedan como estaban (random init de `__init__`).

    Re-evalúa pbest_values de las K warm-started con el cf actual del opt y actualiza
    gbest. Operación in-place sobre los buffers del optimizador.

    El ruido sale de un generador propio derivado de `opt.seed`: no consume ni depende
    del RNG global de torch (qpso-torch tampoco lo toca), así que el cluster es
    reproducible por optimizador sin importar el orden ni el thread en que se cree.
    """
    if k_seeded < 1:
        return

    positions = opt.positions
    lower, upper = opt.lower, opt.upper

    seed_safe = seed_vec.detach().to(dtype=positions.dtype, device=positions.device)
    seed_safe = torch.clamp(seed_safe, min=lower, max=upper)

    range_size = upper - lower

    with torch.no_grad():
        # Partícula 0: seed exacto
        positions[0] = seed_safe

        # Partículas 1..K-1: seed + ruido pequeño
        if k_seeded > 1:
            generator = torch.Generator(device="cpu").manual_seed(
                opt.seed + _NOISE_SEED_OFFSET
            )
            noise = torch.randn(
                k_seeded - 1, opt.dim, generator=generator, dtype=positions.dtype,
            ).to(positions.device) * noise_scale * range_size
            perturbed = seed_safe.unsqueeze(0) + noise
            positions[1:k_seeded] = torch.clamp(perturbed, min=lower, max=upper)

        # Re-evaluar pbest_values de las K warm-started con el cf actual.
        new_vals = opt.cost_fn(positions[:k_seeded])
        opt.pbest[:k_seeded] = positions[:k_seeded].clone()
        opt.pbest_values[:k_seeded] = new_vals

        opt.update_gbest()
