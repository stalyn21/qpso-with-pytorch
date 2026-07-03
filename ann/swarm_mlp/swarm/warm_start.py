"""Warm-start cluster compartido entre T3 y T4.

Reemplaza las primeras K partículas de un QDPSO con perturbaciones alrededor de un
vector semilla. Mantiene el resto random para preservar exploración.
"""
import torch

from tensor_qpso import QDPSOTensorOptimized


def warm_start_cluster(
    opt: QDPSOTensorOptimized,
    seed_vec: torch.Tensor,
    k_seeded: int,
    noise_scale: float,
) -> None:
    """Reemplaza las primeras K partículas con perturbaciones alrededor de `seed_vec`.

    - Partícula 0: seed exacto.
    - Partículas 1..K-1: seed + ruido gaussiano de escala `noise_scale * (high - low)`.
    - Partículas K..P-1: quedan como estaban (random init de `__init__`).

    Re-evalúa pbest_values de las K warm-started con el cf actual del opt y actualiza
    gbest. Operación in-place sobre los buffers internos del QDPSO.
    """
    if k_seeded < 1:
        return

    seed_safe = seed_vec.detach().to(
        dtype=opt._positions.dtype, device=opt._positions.device
    )
    seed_safe = torch.clamp(seed_safe, min=opt._lower, max=opt._upper)

    range_size = opt._upper - opt._lower

    with torch.no_grad():
        # Partícula 0: seed exacto
        opt._positions[0] = seed_safe

        # Partículas 1..K-1: seed + ruido pequeño
        if k_seeded > 1:
            noise = torch.randn(
                k_seeded - 1, opt._dim,
                dtype=opt._positions.dtype, device=opt._positions.device,
            ) * noise_scale * range_size
            perturbed = seed_safe.unsqueeze(0) + noise
            opt._positions[1:k_seeded] = torch.clamp(
                perturbed, min=opt._lower, max=opt._upper
            )

        # Re-evaluar pbest_values de las K warm-started con el cf actual.
        new_vals = opt._cf(opt._positions[:k_seeded])
        opt._pbest[:k_seeded] = opt._positions[:k_seeded].clone()
        opt._pbest_values[:k_seeded] = new_vals

        opt.update_gbest(opt._minimize)
