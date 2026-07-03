"""
QPSO y QDPSO - Implementacion Optimizada con Tensores PyTorch

Version optimizada que incluye todas las mejoras de rendimiento, estabilidad,
funcionalidad, robustez, usabilidad y extensibilidad.

Mejoras implementadas:
    - Rendimiento: Generacion eficiente de signos, unificacion de torch.rand(),
                   memory pool, torch.no_grad()
    - Estabilidad: Division segura, dtype configurable, epsilon correcto
    - Funcionalidad: Boundary handling, convergencia temprana, historial, semilla
    - Robustez: Validacion de parametros, manejo de NaN/Inf
    - Usabilidad: OptimizationResult, context manager
    - Extensibilidad: Callbacks avanzados, soporte para maximizacion

Uso:
    from tensor_qpso.qpso_tensor_optimized import QPSOTensorOptimized, QDPSOTensorOptimized

    # Uso basico
    optimizer = QPSOTensorOptimized(cf, size, dim, bounds, maxIters)
    result = optimizer.optimize()

    # Con context manager
    with QPSOTensorOptimized(cf, size, dim, bounds, maxIters) as opt:
        result = opt.optimize()

    # Con todas las opciones
    optimizer = QPSOTensorOptimized(
        cf=cost_function,
        size=50,
        dim=10,
        bounds=[(-5, 5)] * 10,
        maxIters=1000,
        alpha=(1.0, 0.5),
        device='auto',
        dtype=torch.float32,
        seed=42,
        boundary_strategy='clamp',
        tol=1e-10,
        patience=50,
        track_history=True,
        minimize=True
    )
"""

import torch
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Union, Optional, Dict, Any
from enum import Enum
import warnings


# =============================================================================
# Enumeraciones y Tipos
# =============================================================================

class BoundaryStrategy(Enum):
    """Estrategias para manejar particulas fuera de limites."""
    NONE = "none"           # No aplicar restriccion
    CLAMP = "clamp"         # Truncar a limites
    REFLECT = "reflect"     # Rebotar en limites
    WRAP = "wrap"           # Wrap-around (circular)
    RANDOM = "random"       # Re-inicializar aleatoriamente


class CallbackEvent(Enum):
    """Eventos disponibles para callbacks."""
    ON_INIT = "on_init"
    ON_ITERATION_START = "on_iteration_start"
    ON_ITERATION_END = "on_iteration_end"
    ON_NEW_BEST = "on_new_best"
    ON_CONVERGENCE = "on_convergence"
    ON_FINISH = "on_finish"


# =============================================================================
# Estructuras de Datos
# =============================================================================

@dataclass
class OptimizationResult:
    """
    Resultado estructurado de la optimizacion.

    Attributes:
        best_position: Mejor posicion encontrada (tensor)
        best_value: Mejor valor de fitness
        iterations: Numero de iteraciones ejecutadas
        converged: Si el algoritmo convergio antes de maxIters
        convergence_reason: Razon de la convergencia
        history: Historial de la optimizacion (si track_history=True)
        device: Dispositivo usado
        elapsed_time: Tiempo de ejecucion en segundos
    """
    best_position: torch.Tensor
    best_value: float
    iterations: int
    converged: bool
    convergence_reason: str = ""
    history: Optional[Dict[str, List]] = None
    device: str = ""
    elapsed_time: float = 0.0

    def to_numpy(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario con arrays NumPy."""
        result = {
            'best_position': self.best_position.cpu().numpy(),
            'best_value': self.best_value,
            'iterations': self.iterations,
            'converged': self.converged,
            'convergence_reason': self.convergence_reason,
            'device': self.device,
            'elapsed_time': self.elapsed_time
        }
        if self.history:
            result['history'] = {
                k: [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in vals]
                for k, vals in self.history.items()
            }
        return result

    def __str__(self) -> str:
        return (
            f"OptimizationResult(\n"
            f"  best_value={self.best_value:.6E},\n"
            f"  iterations={self.iterations},\n"
            f"  converged={self.converged},\n"
            f"  reason='{self.convergence_reason}',\n"
            f"  device='{self.device}',\n"
            f"  time={self.elapsed_time:.3f}s\n"
            f")"
        )


# =============================================================================
# Callback Manager
# =============================================================================

class CallbackManager:
    """
    Gestor de callbacks para eventos de optimizacion.

    Permite registrar multiples callbacks para diferentes eventos
    durante el proceso de optimizacion.

    Ejemplo:
        callbacks = CallbackManager()
        callbacks.register(CallbackEvent.ON_NEW_BEST, lambda opt: print(opt.gbest_value))
        callbacks.register(CallbackEvent.ON_ITERATION_END, my_logging_function)
    """

    def __init__(self):
        self._callbacks: Dict[CallbackEvent, List[Callable]] = {
            event: [] for event in CallbackEvent
        }

    def register(self, event: CallbackEvent, callback: Callable) -> None:
        """
        Registra un callback para un evento especifico.

        Args:
            event: Evento que dispara el callback
            callback: Funcion a llamar (recibe el optimizador como argumento)
        """
        if event not in self._callbacks:
            raise ValueError(f"Evento desconocido: {event}")
        self._callbacks[event].append(callback)

    def unregister(self, event: CallbackEvent, callback: Callable) -> bool:
        """
        Elimina un callback registrado.

        Returns:
            True si se elimino, False si no existia
        """
        try:
            self._callbacks[event].remove(callback)
            return True
        except ValueError:
            return False

    def trigger(self, event: CallbackEvent, optimizer: Any) -> None:
        """Dispara todos los callbacks registrados para un evento."""
        for callback in self._callbacks[event]:
            try:
                callback(optimizer)
            except Exception as e:
                warnings.warn(f"Error en callback {event.value}: {e}")

    def clear(self, event: Optional[CallbackEvent] = None) -> None:
        """Limpia callbacks de un evento o todos si event=None."""
        if event:
            self._callbacks[event] = []
        else:
            for e in CallbackEvent:
                self._callbacks[e] = []


# =============================================================================
# Funciones Auxiliares
# =============================================================================

def get_device(device: str = 'auto') -> torch.device:
    """
    Obtiene el dispositivo PyTorch segun la especificacion.

    Args:
        device: 'auto', 'cpu', 'cuda', 'cuda:N', o 'mps' (Apple Silicon)

    Returns:
        torch.device configurado

    Raises:
        ValueError: Si el dispositivo especificado no esta disponible
    """
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')  # Apple Silicon
        return torch.device('cpu')

    if device.startswith('cuda') and not torch.cuda.is_available():
        warnings.warn(f"CUDA no disponible, usando CPU en su lugar")
        return torch.device('cpu')

    return torch.device(device)


def validate_bounds(bounds: List[Tuple[float, float]], dim: int) -> None:
    """
    Valida que los limites sean correctos.

    Args:
        bounds: Lista de tuplas (min, max)
        dim: Dimensionalidad esperada

    Raises:
        ValueError: Si los limites son invalidos
    """
    if len(bounds) != dim:
        raise ValueError(
            f"bounds debe tener {dim} elementos, recibido: {len(bounds)}"
        )

    for i, (lo, hi) in enumerate(bounds):
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            raise ValueError(
                f"bounds[{i}]: valores deben ser numericos, recibido: ({type(lo)}, {type(hi)})"
            )
        if lo >= hi:
            raise ValueError(
                f"bounds[{i}]: lower ({lo}) debe ser < upper ({hi})"
            )


# =============================================================================
# Swarm Tensor Optimizado
# =============================================================================

class SwarmTensorOptimized:
    """
    Enjambre de particulas optimizado usando tensores PyTorch.

    Mejoras sobre SwarmTensor:
        - dtype configurable (float32, float64, float16)
        - Tensores de trabajo pre-alocados
        - Validacion de parametros
        - Soporte para MPS (Apple Silicon)

    Attributes:
        positions: Tensor (n_particles, dim) - posiciones actuales
        pbest: Tensor (n_particles, dim) - mejores posiciones personales
        pbest_values: Tensor (n_particles,) - valores de fitness de pbest
        gbest: Tensor (dim,) - mejor posicion global
        gbest_value: float - mejor valor de fitness global
    """

    def __init__(
        self,
        size: int,
        dim: int,
        bounds: List[Tuple[float, float]],
        device: str = 'auto',
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None
    ):
        """
        Inicializa el enjambre.

        Args:
            size: Numero de particulas (debe ser > 0)
            dim: Dimensionalidad del problema (debe ser > 0)
            bounds: Lista de tuplas (min, max) para cada dimension
            device: 'auto', 'cpu', 'cuda', 'cuda:N', o 'mps'
            dtype: Tipo de dato para tensores (float32, float64, float16)
            seed: Semilla para reproducibilidad (opcional)

        Raises:
            ValueError: Si los parametros son invalidos
        """
        # Validaciones
        if size <= 0:
            raise ValueError(f"size debe ser positivo, recibido: {size}")
        if dim <= 0:
            raise ValueError(f"dim debe ser positivo, recibido: {dim}")
        validate_bounds(bounds, dim)

        # Configurar semilla si se proporciona
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Configuracion basica
        self._device = get_device(device)
        self._dtype = dtype
        self._size = size
        self._dim = dim
        self._bounds = bounds

        # Epsilon para estabilidad numerica segun dtype
        self._eps = torch.finfo(dtype).eps

        # Crear tensores de limites
        bounds_tensor = torch.tensor(
            bounds, dtype=dtype, device=self._device
        )
        self._lower = bounds_tensor[:, 0]  # (dim,)
        self._upper = bounds_tensor[:, 1]  # (dim,)

        # Inicializar posiciones aleatorias dentro de los limites
        self._positions = self._random_positions(size)

        # Inicializar pbest como copia de posiciones iniciales
        self._pbest = self._positions.clone()
        self._pbest_values = torch.full(
            (size,), float('inf'), dtype=dtype, device=self._device
        )

        # Inicializar gbest
        self._gbest = torch.zeros(dim, dtype=dtype, device=self._device)
        self._gbest_value = float('inf')

    def _random_positions(self, n: int) -> torch.Tensor:
        """Genera n posiciones aleatorias dentro de los limites."""
        return (
            torch.rand(n, self._dim, dtype=self._dtype, device=self._device) *
            (self._upper - self._lower) + self._lower
        )

    @property
    def device(self) -> torch.device:
        """Retorna el dispositivo actual."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Retorna el tipo de dato de los tensores."""
        return self._dtype

    @property
    def size(self) -> int:
        """Retorna el numero de particulas."""
        return self._size

    @property
    def dim(self) -> int:
        """Retorna la dimensionalidad."""
        return self._dim

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """Retorna los limites del espacio de busqueda."""
        return self._bounds

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

    def update_gbest(self, minimize: bool = True) -> bool:
        """
        Actualiza gbest basado en los pbest actuales.

        Args:
            minimize: True para minimizar, False para maximizar

        Returns:
            True si gbest fue actualizado, False si no
        """
        if minimize:
            idx = torch.argmin(self._pbest_values)
            value = self._pbest_values[idx].item()
            improved = value < self._gbest_value
        else:
            idx = torch.argmax(self._pbest_values)
            value = self._pbest_values[idx].item()
            improved = value > self._gbest_value

        if improved:
            self._gbest = self._pbest[idx].clone()
            self._gbest_value = value
            return True
        return False


# =============================================================================
# QPSO Base Tensor Optimizado
# =============================================================================

class QPSOBaseTensorOptimized(SwarmTensorOptimized):
    """
    Clase base optimizada para algoritmos QPSO con tensores PyTorch.

    Mejoras implementadas:
        - torch.no_grad() para deshabilitar autograd
        - Memory pool para tensores de trabajo
        - Boundary handling configurable
        - Convergencia temprana
        - Historial de optimizacion
        - Callbacks avanzados
        - Manejo de NaN/Inf
        - Soporte para minimizacion y maximizacion
        - Context manager

    La funcion de costo puede operar de dos formas:
        1. Vectorizada: recibe tensor (n_particles, dim), retorna tensor (n_particles,)
        2. Individual: recibe tensor (dim,), retorna escalar

    Se detecta automaticamente segun el resultado de la primera evaluacion.
    """

    def __init__(
        self,
        cf: Callable,
        size: int,
        dim: int,
        bounds: List[Tuple[float, float]],
        maxIters: int,
        device: str = 'auto',
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
        boundary_strategy: Union[str, BoundaryStrategy] = BoundaryStrategy.CLAMP,
        tol: float = 1e-12,
        patience: int = 100,
        track_history: bool = False,
        minimize: bool = True
    ):
        """
        Inicializa el optimizador QPSO base optimizado.

        Args:
            cf: Funcion de costo a optimizar
            size: Numero de particulas
            dim: Dimensionalidad del problema
            bounds: Lista de tuplas (min, max) para cada dimension
            maxIters: Numero maximo de iteraciones
            device: Dispositivo de computo ('auto', 'cpu', 'cuda', etc.)
            dtype: Tipo de dato para tensores
            seed: Semilla para reproducibilidad
            boundary_strategy: Estrategia para particulas fuera de limites
                - 'none': No aplicar restriccion
                - 'clamp': Truncar a limites (default)
                - 'reflect': Rebotar en limites
                - 'wrap': Wrap-around circular
                - 'random': Re-inicializar aleatoriamente
            tol: Tolerancia para convergencia temprana
            patience: Iteraciones sin mejora antes de parar
            track_history: Si guardar historial de optimizacion
            minimize: True para minimizar, False para maximizar

        Raises:
            ValueError: Si los parametros son invalidos
        """
        # Validar maxIters
        if maxIters <= 0:
            raise ValueError(f"maxIters debe ser positivo, recibido: {maxIters}")

        super().__init__(size, dim, bounds, device, dtype, seed)

        self._cf = cf
        self._maxIters = maxIters
        self._iters = 0
        self._minimize = minimize
        self._vectorized_cf: Optional[bool] = None

        # Ajustar gbest_value inicial segun modo de optimizacion
        if not minimize:
            self._gbest_value = float('-inf')  # Para maximizacion

        # Configuracion de boundary handling
        if isinstance(boundary_strategy, str):
            boundary_strategy = BoundaryStrategy(boundary_strategy.lower())
        self._boundary_strategy = boundary_strategy

        # Configuracion de convergencia temprana
        self._tol = tol
        self._patience = patience
        self._no_improvement_count = 0
        self._prev_gbest_value = float('inf') if minimize else float('-inf')

        # Historial
        self._track_history = track_history
        self._history: Dict[str, List] = {
            'gbest_value': [],
            'mean_fitness': [],
            'std_fitness': [],
            'diversity': []
        } if track_history else {}

        # Callback manager
        self._callbacks = CallbackManager()

        # Pre-alocar tensores de trabajo (memory pool)
        self._init_work_tensors()

        # Tiempo de ejecucion
        self._start_time: Optional[float] = None
        self._elapsed_time: float = 0.0

        # Razon de convergencia
        self._convergence_reason = ""

        # Evaluacion inicial
        self.init_eval()

        # Trigger callback de inicializacion
        self._callbacks.trigger(CallbackEvent.ON_INIT, self)

    def _init_work_tensors(self) -> None:
        """Pre-aloca tensores de trabajo para evitar allocaciones repetidas."""
        n, d = self._size, self._dim
        self._work = {
            # Tensor para todos los numeros aleatorios (4 canales)
            'random': torch.empty(n, d, 4, dtype=self._dtype, device=self._device),
            # Tensor para punto atractor
            'c': torch.empty(n, d, dtype=self._dtype, device=self._device),
            # Tensor para longitud caracteristica
            'L': torch.empty(n, d, dtype=self._dtype, device=self._device),
            # Tensor para signos aleatorios
            'signs': torch.empty(n, d, dtype=self._dtype, device=self._device),
        }

    def _generate_random_batch(self, num_channels: int = 4) -> torch.Tensor:
        """
        Genera un batch de numeros aleatorios eficientemente.

        En lugar de multiples llamadas a torch.rand(), genera todos
        los aleatorios necesarios en una sola operacion.

        Args:
            num_channels: Numero de tensores aleatorios necesarios

        Returns:
            Tensor (n_particles, dim, num_channels)
        """
        return torch.rand(
            self._size, self._dim, num_channels,
            dtype=self._dtype, device=self._device
        )

    def _generate_signs(self) -> torch.Tensor:
        """
        Genera signos aleatorios (+1 o -1) eficientemente.

        Usa torch.randint en lugar de torch.where con multiples tensores.

        Returns:
            Tensor (n_particles, dim) con valores +1 o -1
        """
        # randint genera 0 o 1, multiplicamos por 2 y restamos 1 para obtener -1 o +1
        return torch.randint(
            0, 2, (self._size, self._dim),
            dtype=self._dtype, device=self._device
        ) * 2 - 1

    def _apply_boundary(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Aplica la estrategia de boundary handling.

        Args:
            positions: Tensor de posiciones (n_particles, dim)

        Returns:
            Tensor con posiciones dentro de limites segun estrategia
        """
        strategy = self._boundary_strategy

        if strategy == BoundaryStrategy.NONE:
            return positions

        elif strategy == BoundaryStrategy.CLAMP:
            return torch.clamp(positions, min=self._lower, max=self._upper)

        elif strategy == BoundaryStrategy.REFLECT:
            # Reflejar posiciones que salen de los limites
            # Usamos operaciones vectorizadas con broadcasting
            result = positions.clone()

            # Expandir limites para broadcasting
            lower = self._lower.unsqueeze(0)  # (1, dim)
            upper = self._upper.unsqueeze(0)  # (1, dim)
            range_size = upper - lower

            # Reflejar: usar formula que maneja multiples rebotes
            # Normalizar posicion al rango [0, 2*range]
            normalized = (result - lower) % (2 * range_size)
            # Si esta en [range, 2*range], reflejar
            result = torch.where(
                normalized > range_size,
                2 * range_size - normalized + lower,
                normalized + lower
            )

            # Clamp final para seguridad numerica
            result = torch.clamp(result, min=self._lower, max=self._upper)

            return result

        elif strategy == BoundaryStrategy.WRAP:
            # Wrap-around circular
            range_size = self._upper - self._lower
            return self._lower + (positions - self._lower) % range_size

        elif strategy == BoundaryStrategy.RANDOM:
            # Re-inicializar posiciones fuera de limites
            result = positions.clone()

            # Expandir limites para comparacion
            lower = self._lower.unsqueeze(0)  # (1, dim)
            upper = self._upper.unsqueeze(0)  # (1, dim)

            outside = (result < lower) | (result > upper)

            if outside.any():
                # Generar valores aleatorios para reemplazar
                random_vals = torch.rand_like(result)
                range_size = upper - lower
                new_positions = lower + random_vals * range_size

                # Solo reemplazar los que estan fuera
                result = torch.where(outside, new_positions, result)

            return result

        return positions

    def _evaluate(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Evalua la funcion de costo para las posiciones dadas.

        Incluye:
            - Deteccion automatica de funcion vectorizada
            - Manejo de NaN/Inf
            - Soporte para minimizacion/maximizacion

        Args:
            positions: Tensor (n_particles, dim) o (dim,)

        Returns:
            Tensor (n_particles,) o escalar con los valores de fitness
        """
        # Detectar si es funcion vectorizada
        if self._vectorized_cf is None:
            try:
                result = self._cf(positions)
                if (positions.dim() == 2 and
                    isinstance(result, torch.Tensor) and
                    result.dim() == 1 and
                    result.shape[0] == positions.shape[0]):
                    self._vectorized_cf = True
                else:
                    self._vectorized_cf = False
                    # Re-evaluar con metodo individual
                    if positions.dim() == 2:
                        result = torch.stack([self._cf(p) for p in positions])
            except Exception:
                self._vectorized_cf = False
                result = torch.stack([self._cf(p) for p in positions])
        else:
            if self._vectorized_cf:
                result = self._cf(positions)
            else:
                if positions.dim() == 1:
                    result = self._cf(positions)
                else:
                    result = torch.stack([self._cf(p) for p in positions])

        # Asegurar que es tensor
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result, dtype=self._dtype, device=self._device)

        # Manejar NaN/Inf
        invalid = torch.isnan(result) | torch.isinf(result)
        if invalid.any():
            # Penalizar con valor extremo
            penalty = torch.finfo(self._dtype).max if self._minimize else torch.finfo(self._dtype).min
            result = torch.where(invalid, torch.full_like(result, penalty), result)
            warnings.warn(
                f"Se detectaron {invalid.sum().item()} valores NaN/Inf en evaluacion"
            )

        return result

    def init_eval(self) -> None:
        """Evalua todas las particulas inicialmente."""
        with torch.no_grad():
            self._pbest_values = self._evaluate(self._positions)
            self.update_gbest(self._minimize)

    def update_best(self) -> bool:
        """
        Actualiza pbest de cada particula si mejoro.

        Returns:
            True si gbest fue actualizado, False si no
        """
        with torch.no_grad():
            # Evaluar posiciones actuales
            current_values = self._evaluate(self._positions)

            # Mascara de particulas que mejoraron
            if self._minimize:
                improved = current_values < self._pbest_values
            else:
                improved = current_values > self._pbest_values

            # Actualizar pbest donde mejoro
            if improved.any():
                self._pbest[improved] = self._positions[improved].clone()
                self._pbest_values[improved] = current_values[improved]

            # Actualizar gbest y retornar si mejoro
            return self.update_gbest(self._minimize)

    def _check_convergence(self) -> bool:
        """
        Verifica si el algoritmo ha convergido.

        Criterios:
            - Mejora menor que tolerancia por 'patience' iteraciones consecutivas

        Returns:
            True si convergio, False si no
        """
        if self._minimize:
            improvement = self._prev_gbest_value - self._gbest_value
        else:
            improvement = self._gbest_value - self._prev_gbest_value

        if improvement < self._tol:
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0

        self._prev_gbest_value = self._gbest_value

        if self._no_improvement_count >= self._patience:
            self._convergence_reason = (
                f"Convergencia: sin mejora > {self._tol} por {self._patience} iteraciones"
            )
            return True

        return False

    def _record_history(self) -> None:
        """Registra metricas en el historial."""
        if not self._track_history:
            return

        self._history['gbest_value'].append(self._gbest_value)
        self._history['mean_fitness'].append(self._pbest_values.mean().item())
        self._history['std_fitness'].append(self._pbest_values.std().item())

        # Diversidad: desviacion estandar promedio de las posiciones
        diversity = self._positions.std(dim=0).mean().item()
        self._history['diversity'].append(diversity)

    def kernel_update(self) -> None:
        """
        Actualiza las posiciones de las particulas.
        Debe ser implementado por las subclases.
        """
        raise NotImplementedError("Subclasses must implement kernel_update()")

    def update(
        self,
        callback: Optional[Callable] = None,
        interval: Optional[int] = None
    ) -> None:
        """
        Ejecuta el loop principal de optimizacion.

        Args:
            callback: Funcion opcional llamada cada 'interval' iteraciones
                     (adicional a los callbacks registrados)
            interval: Intervalo de iteraciones para el callback
        """
        import time
        self._start_time = time.time()

        with torch.no_grad():
            while self._iters <= self._maxIters:
                # Callback de inicio de iteracion
                self._callbacks.trigger(CallbackEvent.ON_ITERATION_START, self)

                # Actualizar posiciones
                self.kernel_update()

                # Aplicar boundary handling
                self._positions = self._apply_boundary(self._positions)

                # Actualizar mejores y verificar si gbest mejoro
                gbest_improved = self.update_best()

                # Callback si gbest mejoro
                if gbest_improved:
                    self._callbacks.trigger(CallbackEvent.ON_NEW_BEST, self)

                # Registrar historial
                self._record_history()

                # Callback de fin de iteracion
                self._callbacks.trigger(CallbackEvent.ON_ITERATION_END, self)

                # Callback simple del usuario
                if callback and interval and (self._iters % interval == 0):
                    callback(self)

                # Verificar convergencia temprana
                if self._check_convergence():
                    self._callbacks.trigger(CallbackEvent.ON_CONVERGENCE, self)
                    break

                self._iters += 1

        # Tiempo total
        self._elapsed_time = time.time() - self._start_time

        # Si no convergio, establecer razon
        if not self._convergence_reason:
            self._convergence_reason = f"Maximo de iteraciones alcanzado ({self._maxIters})"

        # Callback de finalizacion
        self._callbacks.trigger(CallbackEvent.ON_FINISH, self)

    def optimize(
        self,
        callback: Optional[Callable] = None,
        interval: Optional[int] = None
    ) -> OptimizationResult:
        """
        Ejecuta la optimizacion y retorna resultado estructurado.

        Args:
            callback: Funcion opcional para monitoreo
            interval: Intervalo de iteraciones para callback

        Returns:
            OptimizationResult con todos los datos de la optimizacion
        """
        self.update(callback, interval)

        return OptimizationResult(
            best_position=self._gbest.clone(),
            best_value=self._gbest_value,
            iterations=self._iters,
            converged=self._no_improvement_count >= self._patience,
            convergence_reason=self._convergence_reason,
            history=self._history if self._track_history else None,
            device=str(self._device),
            elapsed_time=self._elapsed_time
        )

    def reuse_with(self, new_cf: Callable) -> None:
        """
        Re-arma el optimizador para una nueva cost function manteniendo el estado
        espacial (positions, pbest geometry) — warm-start.

        Reset:
            - Iteraciones (_iters, _no_improvement_count)
            - Razon de convergencia
            - Detector de cf vectorizada (re-detect en primera eval)

        Mantiene:
            - positions: la diversidad espacial explorada en la corrida anterior
            - pbest geometry (las posiciones), pero re-evaluadas con la nueva cf

        Re-evalua pbest_values con la nueva cf y recalcula gbest. Esto es necesario
        porque los valores anteriores miden contra la cf vieja y no son comparables.

        Util cuando el problema cambia ligeramente entre llamadas (e.g. block coordinate
        descent: las contrapartes congeladas se actualizaron pero queremos seguir
        explorando desde donde estabamos).
        """
        self._cf = new_cf
        self._vectorized_cf = None  # re-detect
        self._iters = 0
        self._no_improvement_count = 0
        self._convergence_reason = ""
        self._prev_gbest_value = float('inf') if self._minimize else float('-inf')

        # Re-evaluar pbest contra la nueva cf (los valores viejos no son comparables).
        # Mantenemos pbest = positions (la geometria explorada) pero medimos de nuevo.
        with torch.no_grad():
            self._pbest = self._positions.clone()
            self._pbest_values = self._evaluate(self._positions)
            # Reset gbest para forzar recalculo via update_gbest
            self._gbest_value = float('inf') if self._minimize else float('-inf')
            self.update_gbest(self._minimize)

    @property
    def callbacks(self) -> CallbackManager:
        """Retorna el gestor de callbacks."""
        return self._callbacks

    @property
    def iters(self) -> int:
        """Retorna la iteracion actual."""
        return self._iters

    @property
    def maxIters(self) -> int:
        """Retorna el numero maximo de iteraciones."""
        return self._maxIters

    @property
    def history(self) -> Dict[str, List]:
        """Retorna el historial de optimizacion."""
        return self._history

    @property
    def minimize(self) -> bool:
        """Retorna si el optimizador esta minimizando."""
        return self._minimize

    @property
    def elapsed_time(self) -> float:
        """Retorna el tiempo de ejecucion en segundos."""
        return self._elapsed_time

    # Context Manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Libera recursos al salir del context manager."""
        # Limpiar tensores de trabajo
        if hasattr(self, '_work'):
            del self._work

        # Limpiar tensores principales
        del self._positions
        del self._pbest
        del self._pbest_values
        del self._gbest

        # Liberar memoria GPU si aplica
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()

        return False


# =============================================================================
# QPSO Tensor Optimizado
# =============================================================================

class QPSOTensorOptimized(QPSOBaseTensorOptimized):
    """
    QPSO Original (Sun et al., 2004) - Implementacion Optimizada con Tensores

    Implementacion del Quantum Particle Swarm Optimization segun el paper original
    con todas las optimizaciones de rendimiento, estabilidad y funcionalidad.

    Formula:
        mbest = (1/N) * sum(pbest_j) para j=1..N
        c = phi * pbest + (1-phi) * gbest, donde phi ~ U(0,1)
        L = alpha * |mbest - x|
        x_nuevo = c +/- L * ln(1/u)

    Optimizaciones:
        - torch.no_grad() para deshabilitar autograd
        - Generacion eficiente de signos aleatorios
        - Batch de numeros aleatorios
        - Memory pool para tensores de trabajo
        - Boundary handling configurable
        - Convergencia temprana
        - Callbacks avanzados
        - Y todas las demas mejoras de QPSOBaseTensorOptimized

    Args:
        cf: Funcion de costo a optimizar
        size: Numero de particulas
        dim: Dimensionalidad del problema
        bounds: Lista de tuplas (min, max) para cada dimension
        maxIters: Numero maximo de iteraciones
        alpha: Coeficiente de contraccion-expansion (default: 0.75)
               - float: Valor fijo
               - tuple (max, min): Decrecimiento lineal de max a min
        device: Dispositivo de computo
        dtype: Tipo de dato para tensores
        seed: Semilla para reproducibilidad
        boundary_strategy: Estrategia para limites ('clamp', 'reflect', etc.)
        tol: Tolerancia para convergencia
        patience: Iteraciones sin mejora para convergencia
        track_history: Si guardar historial
        minimize: True para minimizar, False para maximizar
    """

    def __init__(
        self,
        cf: Callable,
        size: int,
        dim: int,
        bounds: List[Tuple[float, float]],
        maxIters: int,
        alpha: Union[float, Tuple[float, float]] = 0.75,
        device: str = 'auto',
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
        boundary_strategy: Union[str, BoundaryStrategy] = BoundaryStrategy.CLAMP,
        tol: float = 1e-12,
        patience: int = 100,
        track_history: bool = False,
        minimize: bool = True
    ):
        # Validar y configurar alpha
        if isinstance(alpha, tuple):
            if len(alpha) != 2:
                raise ValueError("alpha tuple debe tener exactamente 2 elementos (max, min)")
            if alpha[0] < alpha[1]:
                raise ValueError(f"alpha[0] ({alpha[0]}) debe ser >= alpha[1] ({alpha[1]})")
            self._alpha = alpha
            self._alpha_max = alpha[0]
            self._alpha_min = alpha[1]
        else:
            if alpha <= 0:
                raise ValueError(f"alpha debe ser positivo, recibido: {alpha}")
            self._alpha = alpha
            self._alpha_max = alpha
            self._alpha_min = alpha

        super().__init__(
            cf=cf,
            size=size,
            dim=dim,
            bounds=bounds,
            maxIters=maxIters,
            device=device,
            dtype=dtype,
            seed=seed,
            boundary_strategy=boundary_strategy,
            tol=tol,
            patience=patience,
            track_history=track_history,
            minimize=minimize
        )

    def _get_alpha(self) -> float:
        """
        Calcula alpha actual (fijo o con decrecimiento lineal).

        Decrecimiento lineal:
            alpha(t) = alpha_max - (alpha_max - alpha_min) * (t / T)

        Returns:
            Valor de alpha para la iteracion actual
        """
        if isinstance(self._alpha, tuple):
            t = self._iters
            T = self._maxIters
            return self._alpha_max - (self._alpha_max - self._alpha_min) * t / T
        return self._alpha

    def kernel_update(self) -> None:
        """
        Actualiza posiciones usando la formula QPSO original (optimizado).

        Optimizaciones aplicadas:
            1. torch.no_grad() (aplicado en update())
            2. Batch de aleatorios: genera todos en una operacion
            3. Signos eficientes: usa randint en lugar de where
            4. Evita allocaciones: reutiliza tensores cuando posible
        """
        n = self._size
        d = self._dim

        # Calcular mbest: promedio de todos los pbest
        mbest = self.mean_best()  # (dim,)

        # Obtener alpha actual
        alpha = self._get_alpha()

        # Generar todos los aleatorios de una vez (optimizacion 2)
        all_random = self._generate_random_batch(num_channels=2)
        phi = all_random[:, :, 0]  # (n, d)
        u = all_random[:, :, 1]    # (n, d)

        # Clamp u para evitar log(0) con epsilon correcto
        u = torch.clamp(u, min=self._eps, max=1.0 - self._eps)

        # Generar signos eficientemente (optimizacion 3)
        rand_sign = self._generate_signs()  # (n, d)

        # Punto atractor: combinacion convexa de pbest y gbest
        c = phi * self._pbest + (1.0 - phi) * self._gbest  # (n, d)

        # Longitud caracteristica usando mbest
        L = alpha * torch.abs(mbest - self._positions)  # (n, d)

        # Nueva posicion con distribucion de Laplace
        self._positions = c + rand_sign * L * torch.log(1.0 / u)


# =============================================================================
# QDPSO Tensor Optimizado
# =============================================================================

class QDPSOTensorOptimized(QPSOBaseTensorOptimized):
    """
    QDPSO - Quantum Delta PSO (Variante) - Implementacion Optimizada

    Variante del QPSO con todas las optimizaciones de rendimiento,
    estabilidad y funcionalidad.

    Formula:
        c = (u1 * pbest + u2 * gbest) / (u1 + u2), donde u1, u2 ~ U(0,1)
        L = (1/g) * |x - c|
        x_nuevo = c +/- L * ln(1/u)

    Diferencias con QPSO original:
        - Punto atractor: promedio ponderado estocastico
        - Longitud L: usa |x - c| en lugar de |mbest - x|
        - Parametro: usa 'g' en lugar de 'alpha'

    Optimizaciones:
        - Todas las de QPSOBaseTensorOptimized
        - Division segura para evitar NaN

    Args:
        cf: Funcion de costo a optimizar
        size: Numero de particulas
        dim: Dimensionalidad del problema
        bounds: Lista de tuplas (min, max) para cada dimension
        maxIters: Numero maximo de iteraciones
        g: Coeficiente de contraccion-expansion (default: 0.96)
        device: Dispositivo de computo
        dtype: Tipo de dato para tensores
        seed: Semilla para reproducibilidad
        boundary_strategy: Estrategia para limites
        tol: Tolerancia para convergencia
        patience: Iteraciones sin mejora para convergencia
        track_history: Si guardar historial
        minimize: True para minimizar, False para maximizar
    """

    def __init__(
        self,
        cf: Callable,
        size: int,
        dim: int,
        bounds: List[Tuple[float, float]],
        maxIters: int,
        g: float = 0.96,
        device: str = 'auto',
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
        boundary_strategy: Union[str, BoundaryStrategy] = BoundaryStrategy.CLAMP,
        tol: float = 1e-12,
        patience: int = 100,
        track_history: bool = False,
        minimize: bool = True
    ):
        # Validar g
        if g <= 0:
            raise ValueError(f"g debe ser positivo, recibido: {g}")

        self._g = g

        super().__init__(
            cf=cf,
            size=size,
            dim=dim,
            bounds=bounds,
            maxIters=maxIters,
            device=device,
            dtype=dtype,
            seed=seed,
            boundary_strategy=boundary_strategy,
            tol=tol,
            patience=patience,
            track_history=track_history,
            minimize=minimize
        )

    def kernel_update(self) -> None:
        """
        Actualiza posiciones usando la formula QDPSO (optimizado).

        Optimizaciones aplicadas:
            1. torch.no_grad() (aplicado en update())
            2. Batch de aleatorios
            3. Signos eficientes
            4. Division segura (evita division por ~0)
        """
        n = self._size
        d = self._dim

        # Generar todos los aleatorios de una vez
        all_random = self._generate_random_batch(num_channels=3)
        u1 = all_random[:, :, 0]  # (n, d)
        u2 = all_random[:, :, 1]  # (n, d)
        u3 = all_random[:, :, 2]  # (n, d)

        # Clamp u3 para evitar log(0)
        u3 = torch.clamp(u3, min=self._eps, max=1.0 - self._eps)

        # Generar signos eficientemente
        rand_sign = self._generate_signs()  # (n, d)

        # Division segura para punto atractor (optimizacion 4)
        divisor = u1 + u2
        divisor = torch.clamp(divisor, min=self._eps)  # Evitar division por ~0

        # Punto atractor: promedio ponderado estocastico
        c = (u1 * self._pbest + u2 * self._gbest) / divisor  # (n, d)

        # Longitud caracteristica usando distancia al punto atractor
        L = (1.0 / self._g) * torch.abs(self._positions - c)  # (n, d)

        # Nueva posicion con distribucion de Laplace
        self._positions = c + rand_sign * L * torch.log(1.0 / u3)


# =============================================================================
# Funciones de Conveniencia
# =============================================================================

def create_optimizer(
    algorithm: str,
    cf: Callable,
    size: int,
    dim: int,
    bounds: List[Tuple[float, float]],
    maxIters: int,
    **kwargs
) -> Union[QPSOTensorOptimized, QDPSOTensorOptimized]:
    """
    Factory function para crear optimizadores.

    Args:
        algorithm: 'qpso' o 'qdpso'
        cf: Funcion de costo
        size: Numero de particulas
        dim: Dimensionalidad
        bounds: Limites
        maxIters: Iteraciones maximas
        **kwargs: Argumentos adicionales para el optimizador

    Returns:
        Instancia del optimizador correspondiente

    Raises:
        ValueError: Si el algoritmo no es reconocido
    """
    algorithm = algorithm.lower()

    if algorithm == 'qpso':
        return QPSOTensorOptimized(cf, size, dim, bounds, maxIters, **kwargs)
    elif algorithm == 'qdpso':
        return QDPSOTensorOptimized(cf, size, dim, bounds, maxIters, **kwargs)
    else:
        raise ValueError(f"Algoritmo no reconocido: {algorithm}. Use 'qpso' o 'qdpso'")
