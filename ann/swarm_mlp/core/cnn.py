"""CNN pequeña con la MISMA interfaz que `core.mlp.MLP`.

Las técnicas (T1..T4) y el protocolo de sesiones sólo usan del modelo:
`num_layers`, `layer_specs[k].{offset,size,n_in,n_out}`, `d_total`, `dtype`,
`forward`, `predict`, `build_layer_cost_fn` y `build_full_cost_fn`. Esta clase
expone exactamente eso, así que corre con ellas sin cambiar una línea: un bloque
entrenable (conv o densa) = una "capa" = un swarm.

Arquitectura (pensada para que el espacio de búsqueda sea tratable por swarms):

    [conv 3x3 → (norm) → ReLU → maxpool 2] × len(channels)  →  cabeza  →  densa → logits

`act_norm="layer"` normaliza la salida de cada conv POR IMAGEN sobre (C, H, W) a
media 0 y varianza 1, sin parámetros (LayerNorm sin γ ni β): no agrega dimensiones a
los enjambres y no depende del lote, así que la predicción de una imagen no cambia
con el chunk en que se evalúa. Mantiene la escala de las activaciones aunque los
pesos que propone el enjambre sean malos.

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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .gain import segment_gain_index, segment_masks
from .loss import cross_entropy_from_logits, l2_penalty


#: Tamaños de CNN del paper 2 (decisión 2026-10-29): ancho y profundidad de las
#: convoluciones, siempre con cabeza GAP para que el modelo no cambie con el tamaño
#: de la imagen. La "grande" (16, 32, 64) quedó fuera por costo (GTX 1050 Ti).
CNN_SIZES: Dict[str, Dict] = {
    "small":  {"channels": (8, 16),  "head": "gap"},   # 1,418 params en FMNIST
    "medium": {"channels": (16, 32), "head": "gap"},   # 5,130 params en FMNIST
    # Cabeza aplanada (2026-11-02): GAP tira la información espacial y limita el techo
    # (Adam con 2,000 imágenes de FMNIST: small 0.755, medium 0.797; small-flat 0.825,
    # medium-flat 0.834). La densa crece con el tamaño de la imagen.
    "small-flat":  {"channels": (8, 16),  "head": "flatten"},   # 5,258 params en FMNIST
    "medium-flat": {"channels": (16, 32), "head": "flatten"},   # 12,810 params en FMNIST
}


#: Normalizaciones de activaciones (ver docstring del módulo).
ACT_NORMS = ("none", "layer")


@dataclass
class CNNLayerSpec:
    idx: int
    kind: str        # 'conv' | 'fc'
    n_in: int        # canales de entrada (conv) o features (fc)
    n_out: int       # canales de salida (conv) o clases (fc)
    ksize: int       # tamaño del kernel (conv); 0 en fc
    offset: int      # offset en el vector plano de parámetros
    size: int        # pesos + bias + 1 si tiene ganancia
    gain: bool = False   # última coordenada = u, con W_eff = exp(u)·W (core/gain.py)

    @property
    def n_weights(self) -> int:
        k2 = self.ksize * self.ksize if self.kind == "conv" else 1
        return self.n_out * self.n_in * k2


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
        act_norm: str = "none",
        norm_eps: float = 1e-5,
        layer_gain: bool = False,
    ):
        if head not in ("gap", "flatten"):
            raise ValueError(f"head debe ser 'gap' o 'flatten', recibido {head!r}")
        if act_norm not in ACT_NORMS:
            raise ValueError(f"act_norm debe ser uno de {ACT_NORMS}, recibido {act_norm!r}")
        self.act_norm = act_norm
        self.norm_eps = float(norm_eps)
        self.device = device
        self.dtype = dtype
        self.head = head
        self.ksize = ksize
        self.n_classes = n_classes
        self.image_size = image_size
        # Presupuesto de elementos del tensor de activaciones más grande por chunk
        # (acota la memoria de GPU al evaluar P partículas × B imágenes).
        self.max_elems = float(max_elems)

        self.layer_gain = bool(layer_gain)
        g = int(self.layer_gain)
        specs: List[CNNLayerSpec] = []
        offset = 0
        c_in, side = in_channels, image_size
        self._out_sides: List[int] = []
        for i, c_out in enumerate(channels):
            size = c_out * c_in * ksize * ksize + c_out + g
            specs.append(CNNLayerSpec(i, "conv", c_in, c_out, ksize, offset, size, self.layer_gain))
            offset += size
            side = (side - ksize + 1) // 2          # conv 'valid' + maxpool 2
            if side < 1:
                raise ValueError("La imagen es demasiado pequeña para esta arquitectura")
            self._out_sides.append(side)
            c_in = c_out

        n_feat = c_in if head == "gap" else c_in * side * side
        size = n_feat * n_classes + n_classes + g
        specs.append(CNNLayerSpec(len(specs), "fc", n_feat, n_classes, 0, offset, size,
                                  self.layer_gain))
        offset += size

        self.layer_specs = specs
        self.num_layers = len(specs)
        self.d_total = offset                                  # con las ganancias
        self.n_net = offset - g * len(specs)                   # pesos + sesgos
        self.layer_sizes = [in_channels] + list(channels) + [n_classes]
        # Forma de la L2 (core/loss.py:L2_FORMS); el runner la fija con --l2-form.
        self.l2_form = "mean"

    # ---------- utilidades de indexación ---------- #

    def layer_dim(self, k: int) -> int:
        return self.layer_specs[k].size

    def _masks(self, k: Optional[int], which: int) -> torch.Tensor:
        """Máscara bool (0 = pesos, 1 = sesgos, 2 = ganancias) del bloque k o de todo el
        vector. Cada bloque guarda los pesos (filtros o matriz), el bias y, si tiene
        ganancia, u."""
        specs = self.layer_specs if k is None else [self.layer_specs[k]]
        return torch.cat([segment_masks(s.n_weights, s.n_out, s.gain, self.device)[which]
                          for s in specs])

    def weight_mask(self, k: Optional[int] = None) -> torch.Tensor:
        """Máscara bool de pesos (True); sesgos y ganancias False. Del bloque k o de todo el vector."""
        return self._masks(k, 0)

    def bias_mask(self, k: Optional[int] = None) -> torch.Tensor:
        return self._masks(k, 1)

    def gain_mask(self, k: Optional[int] = None) -> torch.Tensor:
        return self._masks(k, 2)

    def l2_kwargs(self, k: Optional[int] = None) -> dict:
        """Máscaras e índice de ganancia para `l2_penalty` (bloque k o vector completo)."""
        specs = self.layer_specs if k is None else [self.layer_specs[k]]
        gain_index = torch.cat([
            segment_gain_index(s.n_weights, s.n_out, s.gain,
                               s.offset if k is None else 0, self.device) for s in specs])
        return {"weight_mask": self.weight_mask(k),
                "mean_mask": ~self.gain_mask(k) if self.layer_gain else None,
                "gain_index": gain_index if self.layer_gain else None}

    def l2(self, params: torch.Tensor, lambda_l2: float, n_train: int,
           k: Optional[int] = None) -> torch.Tensor:
        """L2 de `params` (vector completo, o sólo el bloque `k`) con la forma `self.l2_form`.
        Con ganancia, sobre los pesos efectivos (s·W)."""
        return l2_penalty(params, lambda_l2, self.n_net, n_train,
                          form=self.l2_form, **self.l2_kwargs(k))

    def _split(self, seg: torch.Tensor, spec: CNNLayerSpec):
        """seg: (..., size) → (W efectivo, b). W: (..., c_out, c_in, k, k) o (..., n_in, n_out)."""
        lead = seg.shape[:-1]
        n_w = spec.n_weights
        if spec.kind == "conv":
            W = seg[..., :n_w].reshape(*lead, spec.n_out, spec.n_in, spec.ksize, spec.ksize)
        else:
            W = seg[..., :n_w].reshape(*lead, spec.n_in, spec.n_out)
        if spec.gain:
            W = W * torch.exp(seg[..., -1]).reshape(*lead, *([1] * (W.dim() - len(lead))))
        return W, seg[..., n_w : n_w + spec.n_out]

    def _layer_from_flat(self, flat: torch.Tensor, k: int):
        spec = self.layer_specs[k]
        return self._split(flat[spec.offset : spec.offset + spec.size], spec)

    def _norm(self, out: torch.Tensor, groups: int = 1) -> torch.Tensor:
        """Normaliza (B, groups·C, H, W) por imagen y por grupo (partícula) sobre (C, H, W)."""
        if self.act_norm == "none":
            return out
        B = out.shape[0]
        z = out.reshape(B, groups, -1)
        var, mean = torch.var_mean(z, dim=-1, unbiased=False, keepdim=True)
        return ((z - mean) * torch.rsqrt(var + self.norm_eps)).reshape(out.shape)

    def _apply_head(self, h: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, F)."""
        return h.mean(dim=(2, 3)) if self.head == "gap" else h.flatten(1)

    # ---------- forward estándar (un solo vector de parámetros) ---------- #

    def _prefix(self, X: torch.Tensor, flat: torch.Tensor, k: int) -> torch.Tensor:
        """Activaciones que ENTRAN a la capa k, con las capas < k tomadas de `flat`."""
        h = X
        for j in range(k):
            W, b = self._layer_from_flat(flat, j)
            h = F.max_pool2d(F.relu(self._norm(F.conv2d(h, W, b))), 2)
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
                out = self._norm(out, P if per_particle else 1)
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
        n_train = X.shape[0]
        l2_kw = self.l2_kwargs(k)
        l2_form = self.l2_form

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
            out = ce + l2_penalty(positions, lambda_l2, self.n_net, n_train,
                                  form=l2_form, **l2_kw)
            return out.squeeze(0) if squeeze else out

        return cost_fn

    def build_full_cost_fn(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lambda_l2: float = 0.0,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Cost fn para un swarm único optimizando TODOS los parámetros (T4)."""
        n_train = X.shape[0]
        l2_kw = self.l2_kwargs()
        l2_form = self.l2_form

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
            out = ce + l2_penalty(positions, lambda_l2, self.n_net, n_train,
                                  form=l2_form, **l2_kw)
            return out.squeeze(0) if squeeze else out

        return cost_fn
