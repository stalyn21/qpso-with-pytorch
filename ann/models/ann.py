"""
Red Neuronal Artificial Compatible con QPSO

Este modulo implementa una red neuronal feedforward que puede ser
optimizada usando QPSO. La caracteristica principal es que los
pesos de la red se pueden representar como un vector plano que
el algoritmo QPSO usa como posiciones de particulas.

Caracteristicas:
    - Arquitectura flexible con capas ocultas configurables
    - Conversion de pesos a/desde tensor plano
    - Soporte para diferentes funciones de activacion
    - Inicializacion de pesos controlada
    - Compatible con GPU/CPU
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuracion del modelo de red neuronal."""
    input_dim: int
    output_dim: int
    hidden_layers: List[int]
    activation: str = "relu"
    output_activation: Optional[str] = "softmax"
    dropout: float = 0.0
    use_batch_norm: bool = False


def create_scaled_architecture(
    input_dim: int,
    scale_factor: float = 1.0,
    pattern: List[int] = [3, 2, 1]
) -> List[int]:
    """
    Genera arquitectura escalada basada en un patron.

    Args:
        input_dim: Dimension de entrada
        scale_factor: Factor de escala (1.0 = arquitectura base)
        pattern: Patron de multiplicadores para cada capa [3:2:1]

    Returns:
        Lista con las dimensiones de las capas ocultas

    Example:
        >>> create_scaled_architecture(10, 1.5, [3, 2, 1])
        [45, 30, 15]  # 10 * 3 * 1.5, 10 * 2 * 1.5, 10 * 1 * 1.5
    """
    return [int(input_dim * mult * scale_factor) for mult in pattern]


class QPSOCompatibleANN(nn.Module):
    """
    Red Neuronal Artificial optimizable con QPSO.

    Esta red permite que sus pesos sean tratados como posiciones
    de particulas en el algoritmo QPSO.

    Args:
        input_dim: Dimension de entrada
        output_dim: Dimension de salida (clases)
        hidden_layers: Lista con neuronas por capa oculta
        activation: Funcion de activacion ('relu', 'tanh', 'sigmoid', 'leaky_relu')
        output_activation: Activacion de salida ('softmax', 'sigmoid', None)
        dropout: Probabilidad de dropout (0.0 = sin dropout)
        use_batch_norm: Usar batch normalization
        device: Dispositivo ('cpu', 'cuda', 'auto')

    Example:
        >>> model = QPSOCompatibleANN(
        ...     input_dim=10,
        ...     output_dim=3,
        ...     hidden_layers=[30, 20, 10],
        ...     activation='relu'
        ... )
        >>> params = model.get_flat_params()
        >>> model.set_flat_params(params)
    """

    ACTIVATIONS = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'gelu': nn.GELU,
    }

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        activation: str = "relu",
        output_activation: Optional[str] = "softmax",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        device: str = "auto"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.dropout_prob = dropout
        self.use_batch_norm = use_batch_norm

        # Configurar dispositivo
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Construir arquitectura
        self._build_network()

        # Calcular numero total de parametros
        self._num_params = sum(p.numel() for p in self.parameters())

        # Cache para indices de parametros (optimizacion)
        self._param_indices = self._compute_param_indices()

        # Mover al dispositivo
        self.to(self._device)

    def _build_network(self):
        """Construye las capas de la red."""
        layers = []

        # Obtener funcion de activacion
        if self.activation_name not in self.ACTIVATIONS:
            raise ValueError(f"Activacion no soportada: {self.activation_name}")
        activation_class = self.ACTIVATIONS[self.activation_name]

        # Capa de entrada
        prev_dim = self.input_dim

        # Capas ocultas
        for i, hidden_dim in enumerate(self.hidden_layers):
            # Capa lineal
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization (opcional)
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activacion
            layers.append(activation_class())

            # Dropout (opcional)
            if self.dropout_prob > 0:
                layers.append(nn.Dropout(self.dropout_prob))

            prev_dim = hidden_dim

        # Capa de salida
        layers.append(nn.Linear(prev_dim, self.output_dim))

        # Activacion de salida
        if self.output_activation_name == "softmax":
            layers.append(nn.Softmax(dim=-1))
        elif self.output_activation_name == "sigmoid":
            layers.append(nn.Sigmoid())
        elif self.output_activation_name == "log_softmax":
            layers.append(nn.LogSoftmax(dim=-1))

        self.network = nn.Sequential(*layers)

    def _compute_param_indices(self) -> List[Tuple[int, int]]:
        """
        Pre-computa los indices de cada parametro para acceso rapido.

        Returns:
            Lista de tuplas (inicio, fin) para cada parametro
        """
        indices = []
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            indices.append((offset, offset + numel))
            offset += numel
        return indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red.

        Args:
            x: Tensor de entrada [batch_size, input_dim]

        Returns:
            Tensor de salida [batch_size, output_dim]
        """
        return self.network(x)

    @property
    def num_params(self) -> int:
        """Numero total de parametros de la red."""
        return self._num_params

    @property
    def device(self) -> torch.device:
        """Dispositivo actual del modelo."""
        return self._device

    def get_flat_params(self) -> torch.Tensor:
        """
        Obtiene todos los parametros como un tensor plano 1D.

        Returns:
            Tensor 1D con todos los parametros concatenados

        Example:
            >>> params = model.get_flat_params()
            >>> print(params.shape)  # torch.Size([num_params])
        """
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat_params: torch.Tensor) -> None:
        """
        Establece los parametros desde un tensor plano.

        Args:
            flat_params: Tensor 1D con todos los parametros

        Raises:
            ValueError: Si el tamano no coincide

        Example:
            >>> params = torch.randn(model.num_params)
            >>> model.set_flat_params(params)
        """
        if flat_params.numel() != self._num_params:
            raise ValueError(
                f"Tamano incorrecto: esperado {self._num_params}, "
                f"recibido {flat_params.numel()}"
            )

        # Asegurar que esta en el dispositivo correcto
        flat_params = flat_params.to(self._device)

        # Asignar parametros usando indices pre-computados
        for param, (start, end) in zip(self.parameters(), self._param_indices):
            param.data.copy_(flat_params[start:end].view(param.shape))

    def get_param_bounds(
        self,
        bound: Union[float, Tuple[float, float]] = 1.0
    ) -> List[Tuple[float, float]]:
        """
        Genera limites para cada parametro (usado por QPSO).

        Args:
            bound: Limite simetrico o tupla (min, max)

        Returns:
            Lista de tuplas (min, max) para cada parametro

        Example:
            >>> bounds = model.get_param_bounds(1.0)
            >>> # [(-1.0, 1.0), (-1.0, 1.0), ...]
        """
        if isinstance(bound, (int, float)):
            min_val, max_val = -bound, bound
        else:
            min_val, max_val = bound

        return [(min_val, max_val) for _ in range(self._num_params)]

    def clone(self) -> "QPSOCompatibleANN":
        """
        Crea una copia profunda del modelo.

        Returns:
            Nuevo modelo con la misma arquitectura y pesos
        """
        new_model = QPSOCompatibleANN(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.hidden_layers.copy(),
            activation=self.activation_name,
            output_activation=self.output_activation_name,
            dropout=self.dropout_prob,
            use_batch_norm=self.use_batch_norm,
            device=str(self._device)
        )
        new_model.set_flat_params(self.get_flat_params().clone())
        return new_model

    def reset_parameters(self) -> None:
        """Reinicializa los parametros de la red."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_architecture_info(self) -> dict:
        """
        Retorna informacion sobre la arquitectura del modelo.

        Returns:
            Diccionario con informacion de la arquitectura
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation_name,
            "output_activation": self.output_activation_name,
            "dropout": self.dropout_prob,
            "use_batch_norm": self.use_batch_norm,
            "num_params": self._num_params,
            "device": str(self._device),
        }

    def __repr__(self) -> str:
        arch = " -> ".join([str(self.input_dim)] +
                          [str(h) for h in self.hidden_layers] +
                          [str(self.output_dim)])
        return (
            f"QPSOCompatibleANN(\n"
            f"  architecture: {arch}\n"
            f"  activation: {self.activation_name}\n"
            f"  params: {self._num_params:,}\n"
            f"  device: {self._device}\n"
            f")"
        )


class QPSOCompatibleCNN(nn.Module):
    """
    Red Neuronal Convolucional optimizable con QPSO.

    Para tareas de clasificacion de imagenes.
    (Implementacion futura)
    """
    pass
