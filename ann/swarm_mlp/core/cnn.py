"""CNN pequeña con la MISMA interfaz que `core.mlp.MLP`.

Las técnicas (T1..T4) y el protocolo de sesiones sólo usan del modelo:
`num_layers`, `layer_specs[k].{offset,size,n_in,n_out}`, `d_total`, `dtype`,
`forward`, `predict`, `build_layer_cost_fn` y `build_full_cost_fn`. Esta clase
expone exactamente eso, así que corre con ellas sin cambiar una línea: un bloque
entrenable (conv o densa) = una "capa" = un swarm.

Arquitectura (pensada para que el espacio de búsqueda sea tratable por swarms):

    [conv 3x3 → ReLU → maxpool 2] × len(channels)  →  cabeza  →  densa → logits

`head="gap"` promedia cada canal (global average pooling): la capa densa queda
diminuta. `head="flatten"` aplana el mapa final: más parámetros, más capacidad.

Evaluación vectorizada sobre partículas. Hay tres casos por capa convolucional:
  - entrada compartida, P bancos de filtros  → una conv con los P bancos apilados
  - entrada por partícula, pesos compartidos → P se pliega dentro del batch
  - entrada por partícula, P bancos          → conv agrupada (groups=P)
Las capas ANTERIORES a la que se entrena se calculan una sola vez (prefijo
congelado), igual que en el MLP.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .loss import cross_entropy_from_logits, l2_penalty


@dataclass
class CNNLayerSpec:
    idx: int
    kind: str        # 'conv' | 'fc'
    n_in: int        # canales de entrada (conv) o features (fc)
    n_out: int       # canales de salida (conv) o clases (fc)
    ksize: int       # tamaño del kernel (conv); 0 en fc
    offset: int      # offset en el vector plano de parámetros
    size: int        # pesos + bias


class TinyCNN:
    """CNN configurable con cost functions vectorizadas por capa y completa."""

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 28,
        channels: Sequence[int] = (8, 16),
        n_classes: int = 10,
        ksize: int = 3,
        head: str = "gap",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        max_elems: float = 1.2e8,
    ):
        if head not in ("gap", "flatten"):
            raise ValueError(f"head debe ser 'gap' o 'flatten', recibido {head!r}")
        self.device = device
        self.dtype = dtype
        self.head = head
        self.ksize = ksize
        self.n_classes = n_classes
        self.image_size = image_size
        # Presupuesto de elementos del tensor de activaciones más grande por chunk
        # (acota la memoria de GPU al evaluar P partículas × B imágenes).
        self.max_elems = float(max_elems)

        specs: List[CNNLayerSpec] = []
        offset = 0
        c_in, side = in_channels, image_size
        self._out_sides: List[int] = []
        for i, c_out in enumerate(channels):
            size = c_out * c_in * ksize * ksize + c_out
            specs.append(CNNLayerSpec(i, "conv", c_in, c_out, ksize, offset, size))
            offset += size
            side = (side - ksize + 1) // 2          # conv 'valid' + maxpool 2
            if side < 1:
                raise ValueError("La imagen es demasiado pequeña para esta arquitectura")
            self._out_sides.append(side)
            c_in = c_out

        n_feat = c_in if head == "gap" else c_in * side * side
        size = n_feat * n_classes + n_classes
        specs.append(CNNLayerSpec(len(specs), "fc", n_feat, n_classes, 0, offset, size))
        offset += size

        self.layer_specs = specs
        self.num_layers = len(specs)
        self.d_total = offset
        self.layer_sizes = [in_channels] + list(channels) + [n_classes]

    # ---------- utilidades de indexación ---------- #

    def layer_dim(self, k: int) -> int:
        return self.layer_specs[k].size

    def _split(self, seg: torch.Tensor, spec: CNNLayerSpec):
        """seg: (..., size) → (W, b). W: (..., c_out, c_in, k, k) o (..., n_in, n_out)."""
        lead = seg.shape[:-1]
        if spec.kind == "conv":
            n_w = spec.n_out * spec.n_in * spec.ksize * spec.ksize
            W = seg[..., :n_w].reshape(*lead, spec.n_out, spec.n_in, spec.ksize, spec.ksize)
        else:
            n_w = spec.n_in * spec.n_out
            W = seg[..., :n_w].reshape(*lead, spec.n_in, spec.n_out)
        return W, seg[..., n_w:]

    def _layer_from_flat(self, flat: torch.Tensor, k: int):
        spec = self.layer_specs[k]
        return self._split(flat[spec.offset : spec.offset + spec.size], spec)

    def _apply_head(self, h: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, F)."""
        return h.mean(dim=(2, 3)) if self.head == "gap" else h.flatten(1)

    # ---------- forward estándar (un solo vector de parámetros) ---------- #

    def _prefix(self, X: torch.Tensor, flat: torch.Tensor, k: int) -> torch.Tensor:
        """Activaciones que ENTRAN a la capa k, con las capas < k tomadas de `flat`."""
        h = X
        for j in range(k):
            W, b = self._layer_from_flat(flat, j)
            h = F.max_pool2d(F.relu(F.conv2d(h, W, b)), 2)
        if self.layer_specs[k].kind == "fc":
            h = self._apply_head(h)
        return h

    def forward(self, X: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        """X: (B, C, H, W), flat: (d_total,). Devuelve logits (B, n_classes)."""
        k_fc = self.num_layers - 1
        outs = []
        for chunk in X.split(4096):
            feat = self._prefix(chunk, flat, k_fc)
            W, b = self._layer_from_flat(flat, k_fc)
            outs.append(feat @ W + b)
        return torch.cat(outs)

    def predict(self, X: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        return self.forward(X, flat).argmax(dim=-1)

    # ---------- forward vectorizado sobre partículas ---------- #

    def _run_particles(
        self,
        h: torch.Tensor,
        k: int,
        layers: List[Tuple[torch.Tensor, torch.Tensor, bool]],
        P: int,
    ) -> torch.Tensor:
        """Aplica las capas k..N-1 para P partículas. Devuelve logits (B, P, n_classes).

        h: activaciones que entran a la capa k, COMPARTIDAS por todas las partículas.
        layers[i] = (W, b, batched) para la capa k+i; `batched` indica si W/b traen
        una dimensión inicial de tamaño P.
        """
        per_particle = False   # ¿`h` ya lleva una copia por partícula?
        for i, (W, b, batched) in enumerate(layers):
            spec = self.layer_specs[k + i]
            if spec.kind == "conv":
                B = h.shape[0]
                if not per_particle and batched:
                    # entrada compartida, P bancos: apilar los bancos como canales de salida
                    out = F.conv2d(h, W.reshape(P * spec.n_out, spec.n_in, spec.ksize, spec.ksize),
                                   b.reshape(-1))
                    per_particle = True
                elif per_particle and batched:
                    # entrada por partícula, P bancos: conv agrupada
                    out = F.conv2d(h, W.reshape(P * spec.n_out, spec.n_in, spec.ksize, spec.ksize),
                                   b.reshape(-1), groups=P)
                elif per_particle:
                    # entrada por partícula, pesos compartidos: plegar P dentro del batch
                    hh = h.reshape(B * P, spec.n_in, h.shape[-2], h.shape[-1])
                    out = F.conv2d(hh, W, b)
                    out = out.reshape(B, P * spec.n_out, out.shape[-2], out.shape[-1])
                else:
                    out = F.conv2d(h, W, b)
                h = F.max_pool2d(F.relu(out), 2)
                # Si la siguiente capa es la densa, aplicar la cabeza.
                if self.layer_specs[k + i + 1].kind == "fc":
                    if per_particle:
                        h = h.reshape(B, P, spec.n_out, h.shape[-2], h.shape[-1])
                        h = h.mean(dim=(3, 4)) if self.head == "gap" else h.flatten(2)
                    else:
                        h = self._apply_head(h)
            else:
                if per_particle and batched:
                    return torch.einsum("bpf,pfo->bpo", h, W) + b.unsqueeze(0)
                if per_particle:
                    return h @ W + b
                if batched:
                    return torch.einsum("bf,pfo->bpo", h, W) + b.unsqueeze(0)
                return (h @ W + b).unsqueeze(1).expand(-1, P, -1)
        raise RuntimeError("La arquitectura debe terminar en una capa densa")

    def _chunk_size(self, k: int, P: int) -> int:
        """Imágenes por chunk para que el mayor tensor de activaciones quepa en `max_elems`."""
        biggest = 1.0
        for j in range(k, self.num_layers - 1):
            spec = self.layer_specs[j]
            side_before_pool = self._out_sides[j] * 2 + 1
            biggest = max(biggest, spec.n_out * side_before_pool * side_before_pool)
        return max(1, int(self.max_elems // (P * biggest)))

    def _mean_ce(self, h, y, k, layers, P) -> torch.Tensor:
        """CE media sobre todas las imágenes, por partícula: (P,). Procesa por chunks."""
        total = torch.zeros(P, dtype=self.dtype, device=h.device)
        step = self._chunk_size(k, P)
        for start in range(0, h.shape[0], step):
            logits = self._run_particles(h[start : start + step], k, layers, P)   # (c, P, C)
            y_exp = y[start : start + step].unsqueeze(1).expand(-1, P)
            total += cross_entropy_from_logits(logits, y_exp, dim_classes=-1).sum(dim=0)
        return total / h.shape[0]

    # ---------- cost functions ---------- #

    def build_layer_cost_fn(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        k: int,
        frozen_flat: torch.Tensor,
        lambda_l2: float = 0.0,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """cost_fn(positions: (P, layer_dim(k))) -> (P,) con CE + L2, variando SOLO la capa k."""
        spec_k = self.layer_specs[k]
        d_total, n_train = self.d_total, X.shape[0]

        with torch.no_grad():
            # Prefijo congelado: se calcula una vez por cost function.
            h_before = torch.cat([self._prefix(c, frozen_flat, k) for c in X.split(4096)])
            frozen_post = [self._layer_from_flat(frozen_flat, j) + (False,)
                           for j in range(k + 1, self.num_layers)]

        def cost_fn(positions: torch.Tensor) -> torch.Tensor:
            squeeze = positions.dim() == 1
            if squeeze:
                positions = positions.unsqueeze(0)
            P = positions.shape[0]
            W_k, b_k = self._split(positions, spec_k)
            ce = self._mean_ce(h_before, y, k, [(W_k, b_k, True)] + frozen_post, P)
            out = ce + l2_penalty(positions, lambda_l2, d_total, n_train)
            return out.squeeze(0) if squeeze else out

        return cost_fn

    def build_full_cost_fn(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lambda_l2: float = 0.0,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Cost fn para un swarm único optimizando TODOS los parámetros (T4)."""
        d_total, n_train = self.d_total, X.shape[0]

        def cost_fn(positions: torch.Tensor) -> torch.Tensor:
            squeeze = positions.dim() == 1
            if squeeze:
                positions = positions.unsqueeze(0)
            P = positions.shape[0]
            layers = []
            for spec in self.layer_specs:
                W, b = self._split(positions[:, spec.offset : spec.offset + spec.size], spec)
                layers.append((W, b, True))
            ce = self._mean_ce(X, y, 0, layers, P)
            out = ce + l2_penalty(positions, lambda_l2, d_total, n_train)
            return out.squeeze(0) if squeeze else out

        return cost_fn
