# Comparacion de Implementaciones QPSO

[🏠 README](../../README_ES.md) | [📚 Index](index_es.md) | [⬅️ Prev: Optimizado](docs_qpso_tensor_optimized.md) | **Comparacion** | [🇬🇧 English](implementation_comparison.md)

---

> **Version**: 2.0.0
> **Modulo**: QPSO-PyTorch/tensor_qpso/

Este documento proporciona una comparacion detallada de las tres implementaciones de QPSO en terminos de enfoque de codificacion, rendimiento y caracteristicas.

---

## Descripcion General

El modulo `tensor_qpso/` contiene tres implementaciones evolutivas de QPSO:

| Archivo | Nombre | Enfoque | Soporte GPU |
|---------|--------|---------|-------------|
| `qpso.py` | Referencia PyPI | Escalar/Iterativo (NumPy) | No |
| `qpso_tensor.py` | Tensor Basico | Vectorizado (PyTorch) | Si |
| `qpso_tensor_optimized.py` | Tensor Optimizado | Vectorizado + 17 Optimizaciones | Si + MPS |

---

## 1. Comparacion de Arquitectura

### 1.1 Jerarquia de Clases

#### qpso.py (Referencia PyPI)
```
Particle          <- Particula individual con posicion y mejor
    |
Swarm             <- Coleccion de particulas, gestiona gbest
    |
QPSOBase          <- Logica de optimizacion base
    |
+---+---+
|       |
QPSO    QDPSO     <- Algoritmos especificos
```

#### qpso_tensor.py (Tensor Basico)
```
SwarmTensor       <- Todas las particulas como tensores
    |
QPSOBaseTensor    <- Optimizacion base con operaciones tensoriales
    |
+---+---+
|       |
QPSOTensor  QDPSOTensor  <- Algoritmos especificos
```

#### qpso_tensor_optimized.py (Tensor Optimizado)
```
CallbackManager           <- Sistema de callbacks basado en eventos
OptimizationResult        <- Dataclass de resultados estructurados
BoundaryStrategy (Enum)   <- Opciones de manejo de limites
    |
SwarmTensorOptimized      <- Swarm optimizado con validacion
    |
QPSOBaseTensorOptimized   <- Clase base con todas las caracteristicas
    |
+---+---+
|       |
QPSOTensorOptimized  QDPSOTensorOptimized
```

---

## 2. Comparacion de Estructuras de Datos

### 2.1 Representacion de Particulas

#### qpso.py - Orientado a Objetos
```python
class Particle(object):
    def __init__(self, bounds):
        self._x = np.zeros(len(bounds))        # Posicion: array 1D
        for idx, (lo, hi) in enumerate(bounds):
            self._x[idx] = random.uniform(lo, hi)
        self._best = self._x.copy()            # Mejor personal
        self._best_value = np.nan              # Valor escalar
```
- Cada particula es un objeto separado
- Propiedades accedidas via getters/setters
- Memoria: N objetos con arrays individuales

#### qpso_tensor.py - Basado en Tensores
```python
class SwarmTensor:
    def __init__(self, size, dim, bounds, device='auto'):
        # Todas las particulas en un solo tensor
        self._positions = torch.rand(size, dim, device=self._device) * \
                         (self._upper - self._lower) + self._lower
        self._pbest = self._positions.clone()
        self._pbest_values = torch.full((size,), float('inf'), device=self._device)
```
- Todas las particulas en un tensor 2D `(n_particles, dim)`
- Disposicion de memoria eficiente para GPU
- Una sola asignacion para todas las particulas

#### qpso_tensor_optimized.py - Tensores Optimizados
```python
class SwarmTensorOptimized:
    def __init__(self, size, dim, bounds, device='auto',
                 dtype=torch.float32, seed=None):
        # Validacion
        if size <= 0:
            raise ValueError(f"size debe ser positivo")
        validate_bounds(bounds, dim)

        # dtype y seed configurables
        if seed is not None:
            torch.manual_seed(seed)

        self._dtype = dtype
        self._eps = torch.finfo(dtype).eps  # Estabilidad numerica

        # Inicializacion optimizada
        self._positions = self._random_positions(size)
```
- Validacion de parametros
- dtype configurable (float32, float64, float16)
- Reproducibilidad con seed
- Epsilon para estabilidad numerica

---

## 3. Implementacion del Algoritmo Central

### 3.1 Actualizacion del Kernel QPSO

#### qpso.py - Bucles Escalares
```python
def kernel_update(self, **kwargs):
    mbest = self.mean_best()
    alpha = self._get_alpha()

    for p in self._particles:              # Bucle sobre particulas
        for i in range(0, self._dim):      # Bucle sobre dimensiones
            phi = random.uniform(0., 1.)
            u = random.uniform(0., 1.)
            rand_sign = 1 if random.random() > 0.5 else -1

            c = phi * p.best[i] + (1 - phi) * self._gbest[i]
            L = alpha * abs(mbest[i] - p[i])
            p[i] = c + rand_sign * L * np.log(1. / u)
```
**Caracteristicas:**
- Doble bucle anidado: O(n * d) iteraciones
- Generacion individual de numeros aleatorios por dimension
- Sin aceleracion GPU posible
- Simple pero lento para altas dimensiones

#### qpso_tensor.py - Vectorizado
```python
def kernel_update(self) -> None:
    mbest = self.mean_best()  # (dim,)
    alpha = self._get_alpha()

    # Generar todos los numeros aleatorios de una vez
    phi = torch.rand(n, d, device=self._device)
    u = torch.rand(n, d, device=self._device)
    u = torch.clamp(u, min=1e-10)  # Evitar log(0)

    # Signos aleatorios usando torch.where
    rand_sign = torch.where(
        torch.rand(n, d, device=self._device) > 0.5,
        torch.ones(n, d, device=self._device),
        -torch.ones(n, d, device=self._device)
    )

    c = phi * self._pbest + (1 - phi) * self._gbest
    L = alpha * torch.abs(mbest - self._positions)
    self._positions = c + rand_sign * L * torch.log(1.0 / u)
```
**Caracteristicas:**
- Sin bucles explicitos - todo vectorizado
- Ejecucion paralela en GPU
- Multiples asignaciones de tensores por iteracion
- 3x llamadas torch.rand() por iteracion

#### qpso_tensor_optimized.py - Vectorizado Optimizado
```python
def kernel_update(self) -> None:
    mbest = self.mean_best()
    alpha = self._get_alpha()

    # OPTIMIZACION 1: Generacion aleatoria en lote unico
    all_random = self._generate_random_batch(num_channels=2)
    phi = all_random[:, :, 0]
    u = all_random[:, :, 1]

    # OPTIMIZACION 2: Epsilon apropiado segun dtype
    u = torch.clamp(u, min=self._eps, max=1.0 - self._eps)

    # OPTIMIZACION 3: Generacion eficiente de signos
    rand_sign = self._generate_signs()

    c = phi * self._pbest + (1.0 - phi) * self._gbest
    L = alpha * torch.abs(mbest - self._positions)
    self._positions = c + rand_sign * L * torch.log(1.0 / u)

def _generate_random_batch(self, num_channels: int = 4) -> torch.Tensor:
    """Una sola llamada para todos los numeros aleatorios"""
    return torch.rand(
        self._size, self._dim, num_channels,
        dtype=self._dtype, device=self._device
    )

def _generate_signs(self) -> torch.Tensor:
    """Generacion eficiente de signos usando randint"""
    return torch.randint(
        0, 2, (self._size, self._dim),
        dtype=self._dtype, device=self._device
    ) * 2 - 1
```
**Caracteristicas:**
- Una sola llamada torch.rand() en lugar de 3
- randint para signos (mas rapido que where + comparaciones)
- Epsilon correcto basado en dtype
- Tensores de trabajo preasignados

---

## 4. Tabla Comparativa de Caracteristicas

| Caracteristica | qpso.py | qpso_tensor.py | qpso_tensor_optimized.py |
|----------------|---------|----------------|-----------------------------|
| **Ejecucion** | | | |
| Soporte GPU | No | Si (CUDA) | Si (CUDA + MPS) |
| Operaciones Vectorizadas | No | Si | Si |
| Actualizacion Paralela de Particulas | No | Si | Si |
| **Memoria** | | | |
| Memoria Preasignada | No | No | Si (pool de memoria) |
| dtype Configurable | No | No | Si (float16/32/64) |
| Limpieza de Memoria | No | No | Si (context manager) |
| **Estabilidad Numerica** | | | |
| Proteccion Division por Cero | No | Parcial | Si |
| Manejo de NaN/Inf | No | No | Si |
| Epsilon segun dtype | No | Fijo | Si |
| **Funcionalidad** | | | |
| Manejo de Limites | No | No | Si (5 estrategias) |
| Convergencia Temprana | No | No | Si (tolerancia + paciencia) |
| Seguimiento de Historial | No | No | Si |
| Reproducibilidad (seed) | No | No | Si |
| Maximizar/Minimizar | Solo minimizar | Solo minimizar | Ambos |
| **Extensibilidad** | | | |
| Sistema de Callbacks | Basico | Basico | Avanzado (6 eventos) |
| Resultados Estructurados | No | No | Si (OptimizationResult) |
| Context Manager | No | No | Si |
| Validacion de Parametros | No | No | Si |

---

## 5. Comparacion de Generacion de Numeros Aleatorios

### 5.1 qpso.py - Modulo random de Python
```python
# 4 llamadas por particula por dimension por iteracion
phi = random.uniform(0., 1.)
u = random.uniform(0., 1.)
rand_sign = 1 if random.random() > 0.5 else -1
# Para QDPSO: u1, u2, u3 = 3 llamadas mas
```
**Total de llamadas por iteracion**: 4 * n_particles * dim (QPSO) o 5 * n_particles * dim (QDPSO)

### 5.2 qpso_tensor.py - Multiples torch.rand()
```python
phi = torch.rand(n, d, device=self._device)      # Llamada 1
u = torch.rand(n, d, device=self._device)        # Llamada 2
torch.rand(n, d, device=self._device)            # Llamada 3 (para signos)
```
**Total de llamadas por iteracion**: 3 (o 4 para QDPSO)

### 5.3 qpso_tensor_optimized.py - Generacion en lote
```python
# Una sola llamada genera todos los numeros aleatorios
all_random = torch.rand(n, d, num_channels, ...)  # Llamada 1
# Los signos usan metodo eficiente diferente
rand_sign = torch.randint(0, 2, (n, d), ...)      # Llamada 2
```
**Total de llamadas por iteracion**: 2

---

## 6. Comparacion de Manejo de Limites

### 6.1 qpso.py & qpso_tensor.py
```python
# Sin manejo de limites - las particulas pueden salir de los limites
```

### 6.2 qpso_tensor_optimized.py - 5 Estrategias
```python
class BoundaryStrategy(Enum):
    NONE = "none"       # Sin restriccion
    CLAMP = "clamp"     # Recortar a limites
    REFLECT = "reflect" # Rebotar en limites
    WRAP = "wrap"       # Envolver circularmente
    RANDOM = "random"   # Reinicializar aleatoriamente

def _apply_boundary(self, positions):
    if strategy == BoundaryStrategy.CLAMP:
        return torch.clamp(positions, min=self._lower, max=self._upper)

    elif strategy == BoundaryStrategy.REFLECT:
        # Formula de reflexion manejando rebotes multiples
        normalized = (result - lower) % (2 * range_size)
        result = torch.where(
            normalized > range_size,
            2 * range_size - normalized + lower,
            normalized + lower
        )

    elif strategy == BoundaryStrategy.WRAP:
        return self._lower + (positions - self._lower) % range_size

    elif strategy == BoundaryStrategy.RANDOM:
        outside = (result < lower) | (result > upper)
        if outside.any():
            new_positions = lower + random_vals * range_size
            result = torch.where(outside, new_positions, result)
```

---

## 7. Comparacion del Sistema de Callbacks

### 7.1 qpso.py & qpso_tensor.py - Callback Basico
```python
def update(self, callback=None, interval=None):
    while self._iters <= self._maxIters:
        self.kernel_update()
        self.update_best()
        if callback and (self._iters % interval == 0):
            callback(self)  # Callback simple
        self._iters += 1
```

### 7.2 qpso_tensor_optimized.py - Sistema Basado en Eventos
```python
class CallbackEvent(Enum):
    ON_INIT = "on_init"
    ON_ITERATION_START = "on_iteration_start"
    ON_ITERATION_END = "on_iteration_end"
    ON_NEW_BEST = "on_new_best"
    ON_CONVERGENCE = "on_convergence"
    ON_FINISH = "on_finish"

class CallbackManager:
    def register(self, event: CallbackEvent, callback: Callable):
        self._callbacks[event].append(callback)

    def trigger(self, event: CallbackEvent, optimizer):
        for callback in self._callbacks[event]:
            callback(optimizer)

# Uso en el bucle de actualizacion:
def update(self, ...):
    while self._iters <= self._maxIters:
        self._callbacks.trigger(CallbackEvent.ON_ITERATION_START, self)
        self.kernel_update()

        gbest_improved = self.update_best()
        if gbest_improved:
            self._callbacks.trigger(CallbackEvent.ON_NEW_BEST, self)

        if self._check_convergence():
            self._callbacks.trigger(CallbackEvent.ON_CONVERGENCE, self)
            break

        self._callbacks.trigger(CallbackEvent.ON_ITERATION_END, self)
```

---

## 8. Comparacion de Estructura de Resultados

### 8.1 qpso.py & qpso_tensor.py
```python
# Acceso a resultados via propiedades despues de optimizar
optimizer.update()
best_position = optimizer.gbest
best_value = optimizer.gbest_value
iterations = optimizer.iters
```

### 8.2 qpso_tensor_optimized.py - OptimizationResult
```python
@dataclass
class OptimizationResult:
    best_position: torch.Tensor
    best_value: float
    iterations: int
    converged: bool
    convergence_reason: str
    history: Optional[Dict[str, List]]
    device: str
    elapsed_time: float

    def to_numpy(self) -> Dict[str, Any]:
        """Convertir a diccionario con arrays NumPy"""
        ...

# Uso
result = optimizer.optimize()
print(result)
# OptimizationResult(
#   best_value=1.234567E-10,
#   iterations=543,
#   converged=True,
#   reason='Convergencia: sin mejora > 1e-12 por 100 iteraciones',
#   device='cuda:0',
#   time=2.345s
# )
```

---

## 9. Resumen de Rendimiento

### Rendimiento Relativo (mayor es mejor)

| Metrica | qpso.py | qpso_tensor.py | qpso_tensor_optimized.py |
|---------|---------|----------------|-----------------------------|
| **CPU (dim pequena)** | 1.0x | 2-3x | 3-5x |
| **CPU (dim grande)** | 1.0x | 5-10x | 10-20x |
| **GPU (dim pequena)** | N/A | 5-10x | 10-20x |
| **GPU (dim grande)** | N/A | 50-100x | 100-200x |
| **Eficiencia de memoria** | 1.0x | 2x | 3-4x |

### Cuando Usar Cada Implementacion

| Implementacion | Mejor Caso de Uso |
|----------------|-------------------|
| **qpso.py** | Aprendizaje, referencia, depuracion |
| **qpso_tensor.py** | Aceleracion GPU rapida, problemas simples |
| **qpso_tensor_optimized.py** | Produccion, problemas a gran escala, investigacion |

---

## 10. Ejemplos de Codigo

### 10.1 Optimizacion Basica (Todas las Implementaciones)

#### qpso.py
```python
from tensor_qpso.qpso import QPSO

def sphere(x):
    return sum(xi**2 for xi in x)

optimizer = QPSO(
    cf=sphere,
    size=50,
    dim=10,
    bounds=[(-5, 5)] * 10,
    maxIters=1000,
    alpha=(1.0, 0.5)
)
optimizer.update()
print(f"Mejor: {optimizer.gbest_value}")
```

#### qpso_tensor.py
```python
from tensor_qpso.qpso_tensor import QPSOTensor

def sphere(x):
    return (x ** 2).sum(dim=-1)  # Vectorizado

optimizer = QPSOTensor(
    cf=sphere,
    size=50,
    dim=10,
    bounds=[(-5, 5)] * 10,
    maxIters=1000,
    alpha=(1.0, 0.5),
    device='cuda'
)
optimizer.update()
print(f"Mejor: {optimizer.gbest_value}")
```

#### qpso_tensor_optimized.py
```python
from tensor_qpso.qpso_tensor_optimized import QPSOTensorOptimized

def sphere(x):
    return (x ** 2).sum(dim=-1)

# Uso con todas las caracteristicas
optimizer = QPSOTensorOptimized(
    cf=sphere,
    size=50,
    dim=10,
    bounds=[(-5, 5)] * 10,
    maxIters=1000,
    alpha=(1.0, 0.5),
    device='cuda',
    seed=42,
    boundary_strategy='clamp',
    tol=1e-12,
    patience=100,
    track_history=True
)

result = optimizer.optimize()
print(result)
print(f"Longitud del historial: {len(result.history['gbest_value'])}")
```

---

## 11. Resumen

Las tres implementaciones representan una progresion evolutiva:

1. **qpso.py**: Implementacion de referencia PyPI fiel para aprendizaje y depuracion
2. **qpso_tensor.py**: Vectorizacion basada en tensores para aceleracion GPU basica
3. **qpso_tensor_optimized.py**: Listo para produccion con todas las optimizaciones y caracteristicas

Para nuevos proyectos, se recomienda **qpso_tensor_optimized.py** ya que proporciona:
- Mejor rendimiento mediante multiples optimizaciones
- Estabilidad numerica para ejecucion robusta
- Caracteristicas completas para investigacion y produccion
- Compatibilidad total con el modulo de entrenamiento de redes neuronales `ann/`

---

## Documentos Relacionados

- [📘 Implementacion NumPy](docs_qpso.md) - Documentacion detallada de la implementacion de referencia
- [📗 Implementacion Tensor](docs_qpso_tensor.md) - Documentacion de la version con tensores PyTorch
- [📙 Implementacion Optimizada](docs_qpso_tensor_optimized.md) - Documentacion completa con 17 mejoras
- [📦 Modulo ANN](../../ann/docs/index_es.md) - Entrenamiento de redes neuronales usando QPSO

---

<div align="center">

**[⬆️ Volver Arriba](#comparacion-de-implementaciones-qpso)** | **[📚 Index](index_es.md)** | **[🏠 README](../../README_ES.md)** | **[🇬🇧 English](implementation_comparison.md)**

</div>
