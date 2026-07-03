"""MLP configurable + builder de funciones de costo vectorizadas por capa.

Un MLP se define con `layer_sizes=[d_in, h1, h2, ..., n_classes]` y tiene
`N = len(layer_sizes) - 1` capas con pesos (= número de swarms requerido).

El forward completo es estándar. El valor de la clase está en `build_layer_cost_fn`:
produce una función f(positions) vectorizada que el QDPSO puede llamar para evaluar
P partículas en un solo pase por la GPU, variando SOLO la capa `k` y dejando todas
las demás capas congeladas a los parámetros pasados en `frozen_flat`.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import torch

from .loss import cross_entropy_from_logits, l2_penalty


@dataclass
class LayerSpec:
    idx: int
    n_in: int
    n_out: int
    activation: str      # 'tanh', 'relu', 'identity'
    offset: int          # offset en el vector plano de parámetros
    size: int            # W (n_in*n_out) + b (n_out)


def _activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "tanh":
        return torch.tanh(z)
    if kind == "relu":
        return torch.relu(z)
    if kind == "identity":
        return z
    raise ValueError(f"Activación desconocida: {kind}")


class MLP:
    """MLP con capas configurables.

    `build_layer_cost_fn(X, y, k, frozen_flat, lambda_l2)` retorna un callable
    compatible con `QDPSOTensorOptimized.cf`:
        cost(positions: (P, layer_dim(k))) -> (P,) loss (CE + L2)
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: Optional[List[str]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes necesita al menos entrada y salida")
        self.layer_sizes = list(layer_sizes)
        self.num_layers = len(layer_sizes) - 1
        self.device = device
        self.dtype = dtype

        if activations is None:
            # Hidden: tanh. Last: identity (devolvemos logits; softmax es implícito en CE estable).
            activations = ["tanh"] * (self.num_layers - 1) + ["identity"]
        if len(activations) != self.num_layers:
            raise ValueError(
                f"activations debe tener {self.num_layers} elementos, recibido {len(activations)}"
            )
        self.activations = list(activations)

        specs: List[LayerSpec] = []
        offset = 0
        for i in range(self.num_layers):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            size = n_in * n_out + n_out
            specs.append(
                LayerSpec(idx=i, n_in=n_in, n_out=n_out,
                          activation=activations[i], offset=offset, size=size)
            )
            offset += size
        self.layer_specs = specs
        self.d_total = offset

    # ---------- utilidades de indexación ---------- #

    def layer_dim(self, k: int) -> int:
        return self.layer_specs[k].size

    def split_layer_params(self, flat: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """De un vector plano (d_total,) extrae (W, b) para la capa k."""
        spec = self.layer_specs[k]
        seg = flat[spec.offset : spec.offset + spec.size]
        W = seg[: spec.n_in * spec.n_out].reshape(spec.n_in, spec.n_out)
        b = seg[spec.n_in * spec.n_out :]
        return W, b

    # ---------- forward estándar ---------- #

    def forward(self, X: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        """Forward pass completo. X: (B, d_in), flat: (d_total,). Devuelve logits (B, n_classes)."""
        h = X
        for k, spec in enumerate(self.layer_specs):
            W, b = self.split_layer_params(flat, k)
            z = h @ W + b
            if k < self.num_layers - 1:
                h = _activation(z, spec.activation)
            else:
                h = z  # logits crudos; softmax es implícito al hacer argmax/CE
        return h

    def predict(self, X: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        """Predicción de clase (argmax sobre logits). (B,) long."""
        logits = self.forward(X, flat)
        return logits.argmax(dim=-1)

    # ---------- cost function vectorizada por capa ---------- #

    def build_layer_cost_fn(
        self,
        X: torch.Tensor,            # (B, d_in)
        y: torch.Tensor,            # (B,) long
        k: int,                     # capa que se entrena
        frozen_flat: torch.Tensor,  # (d_total,) capas != k congeladas; capa k se ignora
        lambda_l2: float = 0.0,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Retorna cost_fn(positions: (P, layer_dim(k))) -> (P,) con CE + L2.

        Estrategia:
            - Precomputa `a_before` = activaciones hasta antes de la capa k (2D, (B, n_in_k)).
            - Por partícula, "levanta" la capa k con einsum → (P, B, n_out_k).
            - Aplica capas post con pesos congelados (broadcast sobre P).
            - CE contra y (broadcast a (P, B)) + L2 sobre positions.
        """
        spec_k = self.layer_specs[k]
        # Constantes para la normalización del L2 (ver docstring de l2_penalty).
        d_total = self.d_total
        n_train = X.shape[0]

        # Pre-cómputo de activaciones antes de la capa k (una sola vez).
        with torch.no_grad():
            h = X
            for i in range(k):
                W_i, b_i = self.split_layer_params(frozen_flat, i)
                z_i = h @ W_i + b_i
                h = _activation(z_i, self.layer_specs[i].activation)
            a_before = h.detach()  # (B, spec_k.n_in)

        # Pre-extraer parámetros congelados de las capas posteriores.
        frozen_post: List[Tuple[torch.Tensor, torch.Tensor, str, bool]] = []
        for j in range(k + 1, self.num_layers):
            W_j, b_j = self.split_layer_params(frozen_flat, j)
            is_last = (j == self.num_layers - 1)
            frozen_post.append((W_j.detach(), b_j.detach(), self.layer_specs[j].activation, is_last))

        k_is_last = (k == self.num_layers - 1)
        n_w_k = spec_k.n_in * spec_k.n_out

        def cost_fn(positions: torch.Tensor) -> torch.Tensor:
            # positions: (P, size_k) o (size_k,) si el QDPSO llama individual
            if positions.dim() == 1:
                positions = positions.unsqueeze(0)
                squeeze_out = True
            else:
                squeeze_out = False

            P = positions.shape[0]
            W_batch = positions[:, :n_w_k].reshape(P, spec_k.n_in, spec_k.n_out)
            b_batch = positions[:, n_w_k:]  # (P, n_out)

            # Levantar capa k a (P, B, n_out)
            # einsum('bi,pij->pbj', a_before, W_batch)
            z_k = torch.einsum("bi,pij->pbj", a_before, W_batch) + b_batch.unsqueeze(1)

            if k_is_last:
                logits = z_k
            else:
                h = _activation(z_k, spec_k.activation)
                logits = None
                for (W_j, b_j, act_j, is_last) in frozen_post:
                    z = h @ W_j + b_j  # (P, B, d_in) @ (d_in, d_out) -> (P, B, d_out)
                    if is_last:
                        logits = z
                    else:
                        h = _activation(z, act_j)

            # CE vectorizada: logits (P, B, C), y (B,) -> (P, B)
            y_exp = y.unsqueeze(0).expand(P, -1)
            ce_per = cross_entropy_from_logits(logits, y_exp, dim_classes=-1)
            ce = ce_per.mean(dim=-1)  # (P,)

            l2 = l2_penalty(positions, lambda_l2, d_total, n_train)  # (P,)

            out = ce + l2
            return out.squeeze(0) if squeeze_out else out

        return cost_fn

    def build_full_cost_fn(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lambda_l2: float = 0.0,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Cost fn para un swarm único optimizando TODOS los parámetros (para T4)."""
        # Constantes para la normalización del L2 (ver docstring de l2_penalty).
        d_total = self.d_total
        n_train = X.shape[0]

        def cost_fn(positions: torch.Tensor) -> torch.Tensor:
            if positions.dim() == 1:
                positions = positions.unsqueeze(0)
                squeeze_out = True
            else:
                squeeze_out = False
            P = positions.shape[0]
            # Broadcast X a (P, B, d_in)
            h = X.unsqueeze(0).expand(P, -1, -1)
            logits = None
            for k, spec in enumerate(self.layer_specs):
                seg = positions[:, spec.offset : spec.offset + spec.size]
                W = seg[:, : spec.n_in * spec.n_out].reshape(P, spec.n_in, spec.n_out)
                b = seg[:, spec.n_in * spec.n_out :]
                z = torch.bmm(h, W) + b.unsqueeze(1)
                if k == self.num_layers - 1:
                    logits = z
                else:
                    h = _activation(z, spec.activation)

            y_exp = y.unsqueeze(0).expand(P, -1)
            ce = cross_entropy_from_logits(logits, y_exp, dim_classes=-1).mean(dim=-1)
            l2 = l2_penalty(positions, lambda_l2, d_total, n_train)
            out = ce + l2
            return out.squeeze(0) if squeeze_out else out

        return cost_fn
