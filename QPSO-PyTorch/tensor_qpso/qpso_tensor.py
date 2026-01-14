"""
QPSO y QDPSO - Implementacion con Tensores PyTorch

Esta implementacion usa tensores PyTorch para aprovechar:
- Operaciones vectorizadas (sin loops explicitos)
- Aceleracion por GPU (CUDA) o CPU
- Mejor rendimiento para problemas de alta dimensionalidad

Uso:
    # CPU
    optimizer = QPSOTensor(cf, size, dim, bounds, maxIters, device='cpu')

    # GPU (CUDA)
    optimizer = QPSOTensor(cf, size, dim, bounds, maxIters, device='cuda')

    # GPU automatico si esta disponible
    optimizer = QPSOTensor(cf, size, dim, bounds, maxIters, device='auto')
"""

import torch
from typing import Callable, List, Tuple, Union, Optional


def get_device(device: str = 'auto') -> torch.device:
    """
    Obtiene el dispositivo PyTorch segun la especificacion.

    Args:
        device: 'auto', 'cpu', 'cuda', o 'cuda:N'

    Returns:
        torch.device configurado
    """
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


class SwarmTensor:
    """
    Enjambre de particulas usando tensores PyTorch.

    Almacena todas las particulas en tensores de forma (n_particles, dim)
    para operaciones vectorizadas eficientes.

    Atributos:
        positions: Tensor (n_particles, dim) - posiciones actuales
        pbest: Tensor (n_particles, dim) - mejores posiciones personales
        pbest_values: Tensor (n_particles,) - valores de fitness de pbest
        gbest: Tensor (dim,) - mejor posicion global
        gbest_value: float - mejor valor de fitness global
    """

    def __init__(self, size: int, dim: int, bounds: List[Tuple[float, float]],
                 device: str = 'auto'):
        """
        Inicializa el enjambre.

        Args:
            size: Numero de particulas
            dim: Dimensionalidad del problema
            bounds: Lista de tuplas (min, max) para cada dimension
            device: 'auto', 'cpu', 'cuda', o 'cuda:N'
        """
        self._device = get_device(device)
        self._size = size
        self._dim = dim
        self._bounds = bounds

        # Crear tensores de limites
        bounds_tensor = torch.tensor(bounds, dtype=torch.float32, device=self._device)
        self._lower = bounds_tensor[:, 0]  # (dim,)
        self._upper = bounds_tensor[:, 1]  # (dim,)

        # Inicializar posiciones aleatorias dentro de los limites
        # positions: (n_particles, dim)
        self._positions = (
            torch.rand(size, dim, device=self._device) *
            (self._upper - self._lower) + self._lower
        )

        # Inicializar pbest como copia de posiciones iniciales
        self._pbest = self._positions.clone()
        self._pbest_values = torch.full((size,), float('inf'), device=self._device)

        # Inicializar gbest
        self._gbest = torch.zeros(dim, device=self._device)
        self._gbest_value = float('inf')

    @property
    def device(self) -> torch.device:
        """Retorna el dispositivo actual."""
        return self._device

    @property
    def size(self) -> int:
        """Retorna el numero de particulas."""
        return self._size

    @property
    def dim(self) -> int:
        """Retorna la dimensionalidad."""
        return self._dim

    @property
    def positions(self) -> torch.Tensor:
        """Retorna las posiciones actuales (n_particles, dim)."""
        return self._positions

    @property
    def pbest(self) -> torch.Tensor:
        """Retorna las mejores posiciones personales (n_particles, dim)."""
        return self._pbest

    @property
    def pbest_values(self) -> torch.Tensor:
        """Retorna los valores de pbest (n_particles,)."""
        return self._pbest_values

    @property
    def gbest(self) -> torch.Tensor:
        """Retorna la mejor posicion global (dim,)."""
        return self._gbest

    @property
    def gbest_value(self) -> float:
        """Retorna el mejor valor global."""
        return self._gbest_value

    def mean_best(self) -> torch.Tensor:
        """
        Calcula el promedio de todas las mejores posiciones personales (mbest).

        Returns:
            Tensor (dim,) con el mbest
        """
        return self._pbest.mean(dim=0)

    def update_gbest(self) -> None:
        """Actualiza gbest basado en los pbest actuales."""
        min_idx = torch.argmin(self._pbest_values)
        min_value = self._pbest_values[min_idx].item()

        if min_value < self._gbest_value:
            self._gbest = self._pbest[min_idx].clone()
            self._gbest_value = min_value


class QPSOBaseTensor(SwarmTensor):
    """
    Clase base para algoritmos QPSO con tensores PyTorch.

    Contiene la logica comun de inicializacion, evaluacion y actualizacion.
    Las subclases deben implementar kernel_update() con su formula especifica.

    La funcion de costo puede operar de dos formas:
    1. Vectorizada: recibe tensor (n_particles, dim), retorna tensor (n_particles,)
    2. Individual: recibe tensor (dim,), retorna escalar

    Se detecta automaticamente segun el resultado de la primera evaluacion.
    """

    def __init__(self, cf: Callable, size: int, dim: int,
                 bounds: List[Tuple[float, float]], maxIters: int,
                 device: str = 'auto'):
        """
        Inicializa el optimizador QPSO base.

        Args:
            cf: Funcion de costo a minimizar
            size: Numero de particulas
            dim: Dimensionalidad del problema
            bounds: Lista de tuplas (min, max) para cada dimension
            maxIters: Numero maximo de iteraciones
            device: 'auto', 'cpu', 'cuda', o 'cuda:N'
        """
        super().__init__(size, dim, bounds, device)
        self._cf = cf
        self._maxIters = maxIters
        self._iters = 0
        self._vectorized_cf = None  # Se detecta en init_eval

        self.init_eval()

    def _evaluate(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Evalua la funcion de costo para las posiciones dadas.

        Detecta automaticamente si la funcion es vectorizada o individual.

        Args:
            positions: Tensor (n_particles, dim) o (dim,)

        Returns:
            Tensor (n_particles,) o escalar con los valores de fitness
        """
        if self._vectorized_cf is None:
            # Detectar si la funcion es vectorizada
            try:
                result = self._cf(positions)
                if positions.dim() == 2 and result.dim() == 1 and result.shape[0] == positions.shape[0]:
                    self._vectorized_cf = True
                    return result
                else:
                    self._vectorized_cf = False
            except Exception:
                self._vectorized_cf = False

        if self._vectorized_cf:
            return self._cf(positions)
        else:
            # Funcion individual - evaluar particula por particula
            if positions.dim() == 1:
                return self._cf(positions)
            else:
                return torch.stack([self._cf(p) for p in positions])

    def init_eval(self) -> None:
        """Evalua todas las particulas inicialmente."""
        self._pbest_values = self._evaluate(self._positions)
        self.update_gbest()

    def update_best(self) -> None:
        """Actualiza pbest de cada particula si mejoro."""
        # Evaluar posiciones actuales
        current_values = self._evaluate(self._positions)

        # Mascara de particulas que mejoraron
        improved = current_values < self._pbest_values

        # Actualizar pbest donde mejoro
        self._pbest[improved] = self._positions[improved].clone()
        self._pbest_values[improved] = current_values[improved]

        # Actualizar gbest
        self.update_gbest()

    def kernel_update(self) -> None:
        """
        Actualiza las posiciones de las particulas.
        Debe ser implementado por las subclases.
        """
        raise NotImplementedError("Subclasses must implement kernel_update()")

    def update(self, callback: Optional[Callable] = None,
               interval: Optional[int] = None) -> None:
        """
        Ejecuta el loop principal de optimizacion.

        Args:
            callback: Funcion opcional llamada cada 'interval' iteraciones
            interval: Intervalo de iteraciones para el callback
        """
        while self._iters <= self._maxIters:
            self.kernel_update()
            self.update_best()

            if callback and interval and (self._iters % interval == 0):
                callback(self)

            self._iters += 1

    @property
    def iters(self) -> int:
        """Retorna la iteracion actual."""
        return self._iters

    @property
    def maxIters(self) -> int:
        """Retorna el numero maximo de iteraciones."""
        return self._maxIters


class QPSOTensor(QPSOBaseTensor):
    """
    QPSO Original (Sun et al., 2004) - Implementacion con Tensores PyTorch

    Implementacion del Quantum Particle Swarm Optimization segun el paper original.
    Usa el Mean Best Position (mbest) para calcular la longitud caracteristica.

    Formula:
        mbest = (1/N) * sum(pbest_j) para j=1..N
        c = phi * pbest + (1-phi) * gbest, donde phi ~ U(0,1)
        L = alpha * |mbest - x|
        x_nuevo = c +/- L * ln(1/u)

    Caracteristicas de la version con tensores:
        - Operaciones vectorizadas (todas las particulas en paralelo)
        - Soporte para GPU (CUDA) y CPU
        - Mejor rendimiento para problemas de alta dimensionalidad

    Args:
        cf: Funcion de costo a minimizar
        size: Numero de particulas
        dim: Dimensionalidad del problema
        bounds: Lista de tuplas (min, max) para cada dimension
        maxIters: Numero maximo de iteraciones
        alpha: Coeficiente de contraccion-expansion (default: 0.75)
               - Puede ser un valor fijo (float)
               - O una tupla (alpha_max, alpha_min) para decrecimiento lineal
        device: 'auto', 'cpu', 'cuda', o 'cuda:N'
    """

    def __init__(self, cf: Callable, size: int, dim: int,
                 bounds: List[Tuple[float, float]], maxIters: int,
                 alpha: Union[float, Tuple[float, float]] = 0.75,
                 device: str = 'auto'):
        self._alpha = alpha
        self._alpha_max = alpha[0] if isinstance(alpha, tuple) else alpha
        self._alpha_min = alpha[1] if isinstance(alpha, tuple) else alpha
        super().__init__(cf, size, dim, bounds, maxIters, device)

    def _get_alpha(self) -> float:
        """Calcula alpha actual (fijo o con decrecimiento lineal)."""
        if isinstance(self._alpha, tuple):
            t = self._iters
            T = self._maxIters
            return self._alpha_max - (self._alpha_max - self._alpha_min) * t / T
        return self._alpha

    def kernel_update(self) -> None:
        """
        Actualiza posiciones usando la formula QPSO original (vectorizado).

        Todas las operaciones se realizan en tensores para maxima eficiencia.
        """
        n = self._size
        d = self._dim

        # Calcular mbest: promedio de todos los pbest
        mbest = self.mean_best()  # (dim,)

        # Obtener alpha actual
        alpha = self._get_alpha()

        # Generar numeros aleatorios para todas las particulas y dimensiones
        phi = torch.rand(n, d, device=self._device)  # (n, d)
        u = torch.rand(n, d, device=self._device)    # (n, d)
        u = torch.clamp(u, min=1e-10)  # Evitar log(0)

        # Signo aleatorio: +1 o -1 con probabilidad 0.5
        rand_sign = torch.where(
            torch.rand(n, d, device=self._device) > 0.5,
            torch.ones(n, d, device=self._device),
            -torch.ones(n, d, device=self._device)
        )

        # Punto atractor: combinacion convexa de pbest y gbest
        # c = phi * pbest + (1-phi) * gbest
        c = phi * self._pbest + (1 - phi) * self._gbest  # (n, d)

        # Longitud caracteristica usando mbest
        # L = alpha * |mbest - x|
        L = alpha * torch.abs(mbest - self._positions)  # (n, d)

        # Nueva posicion con distribucion de Laplace
        # x_nuevo = c +/- L * ln(1/u)
        self._positions = c + rand_sign * L * torch.log(1.0 / u)


class QDPSOTensor(QPSOBaseTensor):
    """
    QDPSO - Quantum Delta PSO (Variante) - Implementacion con Tensores PyTorch

    Variante del QPSO que usa una formula diferente para el punto atractor
    y la longitud caracteristica.

    Formula:
        c = (u1 * pbest + u2 * gbest) / (u1 + u2), donde u1, u2 ~ U(0,1)
        L = (1/g) * |x - c|
        x_nuevo = c +/- L * ln(1/u)

    Diferencias con QPSO original:
        - Punto atractor: usa promedio ponderado estocastico
        - Longitud L: usa |x - c| en lugar de |mbest - x|
        - Parametro: usa 'g' en lugar de 'alpha'

    Caracteristicas de la version con tensores:
        - Operaciones vectorizadas (todas las particulas en paralelo)
        - Soporte para GPU (CUDA) y CPU
        - Mejor rendimiento para problemas de alta dimensionalidad

    Args:
        cf: Funcion de costo a minimizar
        size: Numero de particulas
        dim: Dimensionalidad del problema
        bounds: Lista de tuplas (min, max) para cada dimension
        maxIters: Numero maximo de iteraciones
        g: Coeficiente de contraccion-expansion (default: 0.96)
        device: 'auto', 'cpu', 'cuda', o 'cuda:N'
    """

    def __init__(self, cf: Callable, size: int, dim: int,
                 bounds: List[Tuple[float, float]], maxIters: int,
                 g: float = 0.96, device: str = 'auto'):
        self._g = g
        super().__init__(cf, size, dim, bounds, maxIters, device)

    def kernel_update(self) -> None:
        """
        Actualiza posiciones usando la formula QDPSO (vectorizado).

        Todas las operaciones se realizan en tensores para maxima eficiencia.
        """
        n = self._size
        d = self._dim

        # Generar numeros aleatorios para todas las particulas y dimensiones
        u1 = torch.rand(n, d, device=self._device)  # (n, d)
        u2 = torch.rand(n, d, device=self._device)  # (n, d)
        u3 = torch.rand(n, d, device=self._device)  # (n, d)
        u3 = torch.clamp(u3, min=1e-10)  # Evitar log(0)

        # Signo aleatorio: +1 o -1 con probabilidad 0.5
        rand_sign = torch.where(
            torch.rand(n, d, device=self._device) > 0.5,
            torch.ones(n, d, device=self._device),
            -torch.ones(n, d, device=self._device)
        )

        # Punto atractor: promedio ponderado estocastico
        # c = (u1 * pbest + u2 * gbest) / (u1 + u2)
        c = (u1 * self._pbest + u2 * self._gbest) / (u1 + u2)  # (n, d)

        # Longitud caracteristica usando distancia al punto atractor
        # L = (1/g) * |x - c|
        L = (1.0 / self._g) * torch.abs(self._positions - c)  # (n, d)

        # Nueva posicion con distribucion de Laplace
        # x_nuevo = c +/- L * ln(1/u)
        self._positions = c + rand_sign * L * torch.log(1.0 / u3)
