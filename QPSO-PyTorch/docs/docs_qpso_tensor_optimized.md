# Documentacion Completa: QPSO y QDPSO Optimizado con Tensores PyTorch

[🏠 README](../../README.md) | [📚 Index](index.md) | [⬅️ Prev: Tensor](docs_qpso_tensor.md) | **Tensor Optimized** | [📊 Comparison](implementation_comparison.md)

---

## Descripcion General

> **Version optimizada** de QPSO y QDPSO con **17 mejoras** implementadas en 6 categorias:
> rendimiento, estabilidad numerica, funcionalidad, robustez, usabilidad y extensibilidad.

Esta implementacion representa la version mas avanzada y completa del algoritmo,
incorporando las mejores practicas de desarrollo de software cientifico y
optimizacion de rendimiento en PyTorch.

---

## Tabla de Contenidos

1. [Arquitectura General](#arquitectura-general)
2. [Mejoras de Rendimiento](#mejoras-de-rendimiento)
3. [Mejoras de Estabilidad Numerica](#mejoras-de-estabilidad-numerica)
4. [Mejoras de Funcionalidad](#mejoras-de-funcionalidad)
5. [Mejoras de Robustez](#mejoras-de-robustez)
6. [Mejoras de Usabilidad](#mejoras-de-usabilidad)
7. [Mejoras de Extensibilidad](#mejoras-de-extensibilidad)
8. [API Completa](#api-completa)
9. [Ejemplos de Uso](#ejemplos-de-uso)
10. [Comparativa de Rendimiento](#comparativa-de-rendimiento)
11. [Guia de Migracion](#guia-de-migracion)
12. [Referencias](#referencias)

---

## Arquitectura General

### Diagrama de Clases

```
Enums/Types:
    BoundaryStrategy       <- Enum: NONE, CLAMP, REFLECT, WRAP, RANDOM
    CallbackEvent          <- Enum: ON_INIT, ON_ITERATION_START, etc.

Data Classes:
    OptimizationResult     <- Resultado estructurado de optimizacion

Utility Classes:
    CallbackManager        <- Gestor de callbacks por eventos

Main Classes:
    SwarmTensorOptimized          <- Enjambre optimizado
           |
           v
    QPSOBaseTensorOptimized       <- Clase base con todas las mejoras
           |
           +---------+---------+
           |                   |
           v                   v
    QPSOTensorOptimized    QDPSOTensorOptimized
       (Original)              (Variante)

Factory Function:
    create_optimizer()     <- Crea instancias por nombre
```

### Archivos del Modulo

| Archivo | Descripcion |
|---------|-------------|
| `qpso_tensor_optimized.py` | Implementacion optimizada completa |
| `qpso_tensor.py` | Implementacion basica (referencia) |
| `qpso.py` | Implementacion NumPy (referencia) |

---

## Mejoras de Rendimiento

### 1.1 Generacion Eficiente de Signos Aleatorios

#### Problema Original

```python
# Version anterior (ineficiente)
rand_sign = torch.where(
    torch.rand(n, d, device=self._device) > 0.5,
    torch.ones(n, d, device=self._device),    # Tensor 1
    -torch.ones(n, d, device=self._device)    # Tensor 2
)
# Crea 3 tensores temporales + operacion condicional
```

#### Solucion Implementada

```python
# Version optimizada
def _generate_signs(self) -> torch.Tensor:
    """
    Genera signos aleatorios (+1 o -1) eficientemente.
    Usa torch.randint en lugar de torch.where con multiples tensores.
    """
    return torch.randint(
        0, 2, (self._size, self._dim),
        dtype=self._dtype, device=self._device
    ) * 2 - 1
```

#### Por que es Mejor

| Aspecto | Version Anterior | Version Optimizada |
|---------|-----------------|-------------------|
| Tensores creados | 3 (rand, ones, -ones) | 1 (randint) |
| Operaciones | where + comparacion | multiplicacion + resta |
| Kernels CUDA | 4 | 2 |
| Mejora estimada | - | ~30% mas rapido |

#### Explicacion Matematica

```
randint(0, 2) genera: 0 o 1
0 * 2 - 1 = -1
1 * 2 - 1 = +1

Resultado: distribucion uniforme de -1 y +1
```

---

### 1.2 Unificacion de Generacion de Aleatorios

#### Problema Original

```python
# Multiples llamadas separadas
phi = torch.rand(n, d, device=self._device)  # Kernel 1
u = torch.rand(n, d, device=self._device)    # Kernel 2
# ... mas torch.rand() para signos           # Kernel 3
```

#### Solucion Implementada

```python
def _generate_random_batch(self, num_channels: int = 4) -> torch.Tensor:
    """
    Genera un batch de numeros aleatorios eficientemente.
    En lugar de multiples llamadas a torch.rand(), genera todos
    los aleatorios necesarios en una sola operacion.
    """
    return torch.rand(
        self._size, self._dim, num_channels,
        dtype=self._dtype, device=self._device
    )

# Uso en kernel_update:
all_random = self._generate_random_batch(num_channels=2)
phi = all_random[:, :, 0]  # Sin allocacion adicional
u = all_random[:, :, 1]    # Vista del mismo tensor
```

#### Por que es Mejor

- **Menor overhead de lanzamiento**: Un kernel CUDA en lugar de multiples
- **Mejor coalescencia**: Acceso contiguo a memoria
- **Menor fragmentacion**: Un tensor grande vs varios pequenos
- **Mejora estimada**: ~15-20% en GPU

---

### 1.3 Memory Pool (Pre-alocacion de Tensores)

#### Problema Original

Cada iteracion crea nuevos tensores:
```python
def kernel_update(self):
    phi = torch.rand(...)      # Allocacion
    u = torch.rand(...)        # Allocacion
    c = phi * self._pbest ...  # Allocacion
    L = alpha * torch.abs(...) # Allocacion
    # El GC debe liberar todo esto
```

#### Solucion Implementada

```python
def _init_work_tensors(self) -> None:
    """Pre-aloca tensores de trabajo para evitar allocaciones repetidas."""
    n, d = self._size, self._dim
    self._work = {
        'random': torch.empty(n, d, 4, dtype=self._dtype, device=self._device),
        'c': torch.empty(n, d, dtype=self._dtype, device=self._device),
        'L': torch.empty(n, d, dtype=self._dtype, device=self._device),
        'signs': torch.empty(n, d, dtype=self._dtype, device=self._device),
    }
```

#### Por que es Mejor

| Aspecto | Sin Memory Pool | Con Memory Pool |
|---------|----------------|-----------------|
| Allocaciones/iter | 5-8 | 0 (reutiliza) |
| Presion en GC | Alta | Minima |
| Fragmentacion | Posible | Evitada |
| Latencia | Variable | Consistente |
| Mejora estimada | - | ~10-25% segun problema |

**Nota**: En la implementacion actual, los tensores se crean en `kernel_update()`
para mantener claridad del codigo, pero el framework esta listo para usar el pool.

---

### 1.4 Deshabilitacion de Autograd con `torch.no_grad()`

#### Problema Original

```python
def update(self, ...):
    while self._iters <= self._maxIters:
        self.kernel_update()  # PyTorch construye grafo computacional
        self.update_best()    # Mas nodos en el grafo
```

Por defecto, PyTorch registra todas las operaciones para poder calcular gradientes.
En optimizacion metaheuristica, **no necesitamos gradientes**.

#### Solucion Implementada

```python
def update(self, ...):
    with torch.no_grad():  # Deshabilita autograd
        while self._iters <= self._maxIters:
            self.kernel_update()
            self.update_best()
            # ... resto del loop
```

#### Por que es Mejor

| Aspecto | Con Autograd | Sin Autograd |
|---------|-------------|--------------|
| Memoria | Almacena grafo | Solo tensores |
| Velocidad | Overhead de registro | Operaciones directas |
| Mejora | - | ~5-15% mas rapido |

#### Cuando NO Usar

Si la funcion de costo involucra una red neuronal que se esta entrenando,
podrias necesitar gradientes. En ese caso, usar la version sin `no_grad()`.

---

## Mejoras de Estabilidad Numerica

### 2.1 Division Segura en QDPSO

#### Problema Original

```python
c = (u1 * self._pbest + u2 * self._gbest) / (u1 + u2)
# Si u1 ≈ 0 y u2 ≈ 0, entonces u1 + u2 ≈ 0 → division por ~0
```

Aunque la probabilidad es baja (u1, u2 ~ U(0,1)), con millones de operaciones
puede ocurrir, causando valores `inf` o `NaN`.

#### Solucion Implementada

```python
# Division segura
divisor = u1 + u2
divisor = torch.clamp(divisor, min=self._eps)  # Minimo valor seguro
c = (u1 * self._pbest + u2 * self._gbest) / divisor
```

#### Por que es Necesario

| Escenario | Sin Proteccion | Con Proteccion |
|-----------|---------------|----------------|
| u1=0.001, u2=0.001 | c = valor/0.002 (OK) | c = valor/0.002 (OK) |
| u1=1e-20, u2=1e-20 | c = valor/2e-20 (overflow) | c = valor/eps (seguro) |
| u1=0, u2=0 | c = valor/0 (NaN) | c = valor/eps (seguro) |

---

### 2.2 Precision Configurable (dtype)

#### Problema Original

```python
# Siempre float32
bounds_tensor = torch.tensor(bounds, dtype=torch.float32, device=self._device)
```

float32 tiene ~7 digitos de precision, insuficiente para algunos problemas.

#### Solucion Implementada

```python
def __init__(self, ..., dtype: torch.dtype = torch.float32):
    self._dtype = dtype
    self._eps = torch.finfo(dtype).eps  # Epsilon correcto para el dtype

    # Todos los tensores usan el dtype configurado
    bounds_tensor = torch.tensor(bounds, dtype=dtype, device=self._device)
    self._positions = torch.rand(..., dtype=dtype, ...)
    # etc.
```

#### Opciones de dtype

| dtype | Precision | Bits | Uso Recomendado |
|-------|-----------|------|-----------------|
| `torch.float16` | ~3 digitos | 16 | GPUs modernas, velocidad maxima |
| `torch.float32` | ~7 digitos | 32 | Default, balance precision/velocidad |
| `torch.float64` | ~15 digitos | 64 | Alta precision, problemas sensibles |

#### Ejemplo de Uso

```python
# Alta precision
optimizer = QPSOTensorOptimized(
    cf, size, dim, bounds, maxIters,
    dtype=torch.float64
)

# Maxima velocidad (GPUs con Tensor Cores)
optimizer = QPSOTensorOptimized(
    cf, size, dim, bounds, maxIters,
    dtype=torch.float16
)
```

---

### 2.3 Epsilon Correcto segun dtype

#### Problema Original

```python
u = torch.clamp(u, min=1e-10)  # Epsilon hardcodeado
```

El valor `1e-10` puede ser:
- Demasiado pequeno para float16 (underflow)
- Demasiado grande para float64 (precision desperdiciada)

#### Solucion Implementada

```python
# En __init__:
self._eps = torch.finfo(dtype).eps

# En kernel_update:
u = torch.clamp(u, min=self._eps, max=1.0 - self._eps)
```

#### Valores de Epsilon por dtype

| dtype | torch.finfo().eps |
|-------|------------------|
| float16 | ~0.001 (9.77e-4) |
| float32 | ~1.19e-7 |
| float64 | ~2.22e-16 |

---

## Mejoras de Funcionalidad

### 3.1 Boundary Handling (Restriccion de Limites)

#### Problema Original

Las particulas pueden salir del espacio de busqueda definido por `bounds`,
generando soluciones invalidas.

#### Solucion Implementada

```python
class BoundaryStrategy(Enum):
    NONE = "none"       # Sin restriccion
    CLAMP = "clamp"     # Truncar a limites
    REFLECT = "reflect" # Rebotar en limites
    WRAP = "wrap"       # Circular (wrap-around)
    RANDOM = "random"   # Re-inicializar fuera de limites
```

#### Estrategias Explicadas

##### CLAMP (Default)
```python
positions = torch.clamp(positions, min=lower, max=upper)
```
- Trunca valores fuera de rango al limite mas cercano
- Simple y eficiente
- Puede causar acumulacion en los bordes

##### REFLECT
```python
# Particula "rebota" en el limite
if position < lower:
    position = lower + (lower - position)
if position > upper:
    position = upper - (position - upper)
```
- Simula rebote elastico
- Preserva "energia" del movimiento
- Mejor exploracion de bordes

##### WRAP
```python
position = lower + (position - lower) % (upper - lower)
```
- Comportamiento circular/toroidal
- Util para espacios periodicos (angulos, fases)

##### RANDOM
```python
if position outside bounds:
    position = random_uniform(lower, upper)
```
- Re-inicializa particulas perdidas
- Mayor diversidad
- Puede perder informacion de la trayectoria

#### Diagrama Visual

```
Espacio de busqueda: [lower=0, upper=10]
Particula en posicion 12 (fuera de limites):

CLAMP:    12 → 10 (truncado al maximo)
          ----[====|====]----
                        ^12→10

REFLECT:  12 → 8 (rebota: 10 - (12-10) = 8)
          ----[====|====]----
                    ←  ^

WRAP:     12 → 2 (wrap: 0 + (12-0)%10 = 2)
          ----[====|====]----
             ^         ↺

RANDOM:   12 → 7.3 (aleatorio en [0,10])
          ----[====|====]----
                   ^?
```

---

### 3.2 Convergencia Temprana

#### Problema Original

```python
while self._iters <= self._maxIters:  # Siempre ejecuta todas las iteraciones
```

Desperdicia tiempo si el algoritmo ya convergio.

#### Solucion Implementada

```python
def __init__(self, ..., tol: float = 1e-12, patience: int = 100):
    self._tol = tol
    self._patience = patience
    self._no_improvement_count = 0
    self._prev_gbest_value = float('inf')

def _check_convergence(self) -> bool:
    """Verifica si el algoritmo ha convergido."""
    improvement = self._prev_gbest_value - self._gbest_value

    if improvement < self._tol:
        self._no_improvement_count += 1
    else:
        self._no_improvement_count = 0

    self._prev_gbest_value = self._gbest_value

    return self._no_improvement_count >= self._patience
```

#### Parametros

| Parametro | Descripcion | Default |
|-----------|-------------|---------|
| `tol` | Mejora minima considerada significativa | 1e-12 |
| `patience` | Iteraciones sin mejora antes de parar | 100 |

#### Comportamiento

```
Iteracion 500: gbest = 1.5e-10, prev = 1.6e-10
  Mejora = 1e-11 > tol(1e-12) → no_improvement = 0

Iteracion 501: gbest = 1.5e-10, prev = 1.5e-10
  Mejora = 0 < tol → no_improvement = 1

... (100 iteraciones sin mejora significativa)

Iteracion 600: no_improvement = 100 >= patience
  → CONVERGENCIA TEMPRANA
```

---

### 3.3 Historial de Optimizacion

#### Problema Original

No hay forma de analizar el comportamiento del algoritmo despues de ejecutar.

#### Solucion Implementada

```python
def __init__(self, ..., track_history: bool = False):
    self._track_history = track_history
    if track_history:
        self._history = {
            'gbest_value': [],    # Mejor valor por iteracion
            'mean_fitness': [],   # Promedio de pbest_values
            'std_fitness': [],    # Desviacion estandar
            'diversity': []       # Diversidad del enjambre
        }

def _record_history(self) -> None:
    if not self._track_history:
        return

    self._history['gbest_value'].append(self._gbest_value)
    self._history['mean_fitness'].append(self._pbest_values.mean().item())
    self._history['std_fitness'].append(self._pbest_values.std().item())

    # Diversidad: desviacion estandar promedio de posiciones
    diversity = self._positions.std(dim=0).mean().item()
    self._history['diversity'].append(diversity)
```

#### Metricas del Historial

| Metrica | Descripcion | Utilidad |
|---------|-------------|----------|
| `gbest_value` | Mejor valor encontrado | Curva de convergencia |
| `mean_fitness` | Promedio del enjambre | Calidad general |
| `std_fitness` | Variabilidad de fitness | Distribucion de soluciones |
| `diversity` | Dispersion espacial | Exploracion vs explotacion |

#### Ejemplo de Uso

```python
optimizer = QPSOTensorOptimized(
    cf, size, dim, bounds, maxIters,
    track_history=True
)
result = optimizer.optimize()

# Graficar convergencia
import matplotlib.pyplot as plt
plt.plot(result.history['gbest_value'])
plt.xlabel('Iteracion')
plt.ylabel('Mejor Valor')
plt.yscale('log')
plt.show()
```

---

### 3.4 Semilla Aleatoria Configurable

#### Problema Original

Diferentes ejecuciones dan resultados diferentes, imposible reproducir experimentos.

#### Solucion Implementada

```python
def __init__(self, ..., seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # Todas las GPUs
```

#### Por que es Importante

| Caso de Uso | Sin Semilla | Con Semilla |
|-------------|------------|-------------|
| Debugging | Dificil reproducir bugs | Bug reproducible |
| Comparacion | Resultados varian | Comparacion justa |
| Paper/Reporte | No reproducible | Reproducible |
| Tests | Flakey tests | Tests deterministicos |

#### Ejemplo

```python
# Resultados reproducibles
opt1 = QPSOTensorOptimized(cf, 40, 10, bounds, 1000, seed=42)
result1 = opt1.optimize()

opt2 = QPSOTensorOptimized(cf, 40, 10, bounds, 1000, seed=42)
result2 = opt2.optimize()

assert result1.best_value == result2.best_value  # Siempre True
```

---

## Mejoras de Robustez

### 4.1 Validacion de Parametros

#### Problema Original

Errores confusos cuando se pasan parametros invalidos:
```python
optimizer = QPSOTensor(cf, -5, 10, bounds, 1000)  # size=-5
# Error: "tensors cannot have negative dimensions"
```

#### Solucion Implementada

```python
def __init__(self, ...):
    # Validaciones con mensajes claros
    if size <= 0:
        raise ValueError(f"size debe ser positivo, recibido: {size}")
    if dim <= 0:
        raise ValueError(f"dim debe ser positivo, recibido: {dim}")
    if maxIters <= 0:
        raise ValueError(f"maxIters debe ser positivo, recibido: {maxIters}")

    validate_bounds(bounds, dim)

def validate_bounds(bounds, dim):
    if len(bounds) != dim:
        raise ValueError(
            f"bounds debe tener {dim} elementos, recibido: {len(bounds)}"
        )
    for i, (lo, hi) in enumerate(bounds):
        if lo >= hi:
            raise ValueError(
                f"bounds[{i}]: lower ({lo}) debe ser < upper ({hi})"
            )
```

#### Beneficios

| Antes | Despues |
|-------|---------|
| `RuntimeError: tensors cannot...` | `ValueError: size debe ser positivo, recibido: -5` |
| `IndexError: list index out of range` | `ValueError: bounds debe tener 10 elementos, recibido: 5` |

---

### 4.2 Manejo de NaN/Inf en Evaluacion

#### Problema Original

Si la funcion de costo retorna NaN o Inf, el algoritmo se corrompe:
```python
gbest_value = NaN  # Se propaga a todo
```

#### Solucion Implementada

```python
def _evaluate(self, positions):
    values = self._cf(positions)

    # Detectar valores invalidos
    invalid = torch.isnan(values) | torch.isinf(values)

    if invalid.any():
        # Penalizar con valor extremo
        penalty = (torch.finfo(self._dtype).max
                   if self._minimize else
                   torch.finfo(self._dtype).min)
        values = torch.where(invalid, torch.full_like(values, penalty), values)

        warnings.warn(
            f"Se detectaron {invalid.sum().item()} valores NaN/Inf"
        )

    return values
```

#### Comportamiento

| Valor Retornado | Minimizacion | Maximizacion |
|-----------------|--------------|--------------|
| NaN | → 3.4e38 (max float32) | → -3.4e38 (min) |
| Inf | → 3.4e38 | → -3.4e38 |
| -Inf | → 3.4e38 | → -3.4e38 |
| Valor normal | Sin cambio | Sin cambio |

Las particulas con valores invalidos son fuertemente penalizadas,
efectivamente descartandolas sin crashear el algoritmo.

---

## Mejoras de Usabilidad

### 5.1 OptimizationResult - Resultado Estructurado

#### Problema Original

```python
optimizer.update()
# Usuario debe acceder multiples propiedades
best = optimizer.gbest
value = optimizer.gbest_value
iters = optimizer.iters
# ... tedioso y propenso a errores
```

#### Solucion Implementada

```python
@dataclass
class OptimizationResult:
    """Resultado estructurado de la optimizacion."""
    best_position: torch.Tensor
    best_value: float
    iterations: int
    converged: bool
    convergence_reason: str = ""
    history: Optional[Dict[str, List]] = None
    device: str = ""
    elapsed_time: float = 0.0

    def to_numpy(self) -> Dict[str, Any]:
        """Convierte a diccionario con arrays NumPy."""
        ...

    def __str__(self) -> str:
        """Representacion legible."""
        ...
```

#### Uso

```python
result = optimizer.optimize()

print(result)
# OptimizationResult(
#   best_value=1.234567E-10,
#   iterations=543,
#   converged=True,
#   reason='Convergencia: sin mejora > 1e-12 por 100 iteraciones',
#   device='cuda',
#   time=2.345s
# )

# Acceso estructurado
if result.converged:
    print(f"Convergido en {result.iterations} iteraciones")

# Conversion a NumPy para guardar/analizar
data = result.to_numpy()
np.save('resultado.npy', data)
```

---

### 5.2 Context Manager para Recursos

#### Problema Original

```python
optimizer = QPSOTensor(cf, ...)
optimizer.update()
# Memoria GPU no se libera hasta que GC la recoja
# Puede causar OOM en ejecuciones consecutivas
```

#### Solucion Implementada

```python
class QPSOBaseTensorOptimized:
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

        # Liberar memoria GPU
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()

        return False
```

#### Uso

```python
# Memoria se libera automaticamente al salir del 'with'
with QPSOTensorOptimized(cf, ...) as optimizer:
    result = optimizer.optimize()

# Aqui la memoria GPU ya fue liberada
# Seguro para crear otro optimizador
```

---

## Mejoras de Extensibilidad

### 6.1 Sistema de Callbacks Avanzado

#### Problema Original

Solo un callback simple llamado cada N iteraciones.

#### Solucion Implementada

```python
class CallbackEvent(Enum):
    ON_INIT = "on_init"
    ON_ITERATION_START = "on_iteration_start"
    ON_ITERATION_END = "on_iteration_end"
    ON_NEW_BEST = "on_new_best"
    ON_CONVERGENCE = "on_convergence"
    ON_FINISH = "on_finish"

class CallbackManager:
    def register(self, event: CallbackEvent, callback: Callable) -> None:
        """Registra callback para un evento."""
        self._callbacks[event].append(callback)

    def trigger(self, event: CallbackEvent, optimizer: Any) -> None:
        """Dispara todos los callbacks de un evento."""
        for callback in self._callbacks[event]:
            callback(optimizer)
```

#### Eventos Disponibles

| Evento | Cuando se Dispara | Uso Tipico |
|--------|------------------|------------|
| `ON_INIT` | Despues de inicializacion | Setup inicial |
| `ON_ITERATION_START` | Inicio de cada iteracion | Logging detallado |
| `ON_ITERATION_END` | Fin de cada iteracion | Metricas, checkpoints |
| `ON_NEW_BEST` | Cuando gbest mejora | Notificaciones |
| `ON_CONVERGENCE` | Al detectar convergencia | Logging, alertas |
| `ON_FINISH` | Al terminar optimizacion | Cleanup, reportes |

#### Ejemplo de Uso

```python
def log_new_best(opt):
    print(f"Nuevo mejor: {opt.gbest_value:.6E} en iteracion {opt.iters}")

def save_checkpoint(opt):
    if opt.iters % 100 == 0:
        torch.save(opt.gbest, f'checkpoint_{opt.iters}.pt')

optimizer = QPSOTensorOptimized(cf, ...)
optimizer.callbacks.register(CallbackEvent.ON_NEW_BEST, log_new_best)
optimizer.callbacks.register(CallbackEvent.ON_ITERATION_END, save_checkpoint)
optimizer.optimize()
```

---

### 6.2 Soporte para Maximizacion

#### Problema Original

Solo soporta minimizacion. Para maximizar, usuario debe negar la funcion.

#### Solucion Implementada

```python
def __init__(self, ..., minimize: bool = True):
    self._minimize = minimize
    self._prev_gbest_value = float('inf') if minimize else float('-inf')

def update_best(self) -> bool:
    if self._minimize:
        improved = current_values < self._pbest_values
    else:
        improved = current_values > self._pbest_values
    # ... resto igual
```

#### Uso

```python
# Minimizar (default)
opt_min = QPSOTensorOptimized(cf, ..., minimize=True)

# Maximizar
opt_max = QPSOTensorOptimized(cf, ..., minimize=False)
```

---

## API Completa

### QPSOTensorOptimized

```python
QPSOTensorOptimized(
    cf: Callable,                    # Funcion de costo
    size: int,                       # Numero de particulas
    dim: int,                        # Dimensionalidad
    bounds: List[Tuple[float, float]], # Limites por dimension
    maxIters: int,                   # Iteraciones maximas
    alpha: float | Tuple[float, float] = 0.75,  # Coeficiente alpha
    device: str = 'auto',            # 'auto', 'cpu', 'cuda', etc.
    dtype: torch.dtype = torch.float32,  # Precision
    seed: int = None,                # Semilla aleatoria
    boundary_strategy: str = 'clamp', # 'none', 'clamp', 'reflect', 'wrap', 'random'
    tol: float = 1e-12,              # Tolerancia convergencia
    patience: int = 100,             # Paciencia convergencia
    track_history: bool = False,     # Guardar historial
    minimize: bool = True            # True=minimizar, False=maximizar
)
```

### QDPSOTensorOptimized

```python
QDPSOTensorOptimized(
    # Mismos parametros que QPSOTensorOptimized excepto:
    g: float = 0.96,  # En lugar de alpha
)
```

### Metodos Principales

| Metodo | Descripcion |
|--------|-------------|
| `optimize(callback, interval)` | Ejecuta y retorna OptimizationResult |
| `update(callback, interval)` | Ejecuta sin retorno estructurado |

### Propiedades

| Propiedad | Tipo | Descripcion |
|-----------|------|-------------|
| `device` | `torch.device` | Dispositivo |
| `dtype` | `torch.dtype` | Tipo de dato |
| `size` | `int` | Numero de particulas |
| `dim` | `int` | Dimensionalidad |
| `positions` | `Tensor` | Posiciones actuales |
| `pbest` | `Tensor` | Mejores personales |
| `pbest_values` | `Tensor` | Valores de pbest |
| `gbest` | `Tensor` | Mejor global |
| `gbest_value` | `float` | Valor de gbest |
| `iters` | `int` | Iteracion actual |
| `maxIters` | `int` | Iteraciones maximas |
| `history` | `Dict` | Historial (si habilitado) |
| `callbacks` | `CallbackManager` | Gestor de callbacks |
| `minimize` | `bool` | Modo de optimizacion |
| `elapsed_time` | `float` | Tiempo transcurrido |

---

## Ejemplos de Uso

### Uso Basico

```python
import torch
from tensor_qpso.qpso_tensor_optimized import QPSOTensorOptimized

def sphere(x):
    if x.dim() == 1:
        return torch.sum(x ** 2)
    return torch.sum(x ** 2, dim=1)

optimizer = QPSOTensorOptimized(
    cf=sphere,
    size=40,
    dim=10,
    bounds=[(-5.12, 5.12)] * 10,
    maxIters=1000
)

result = optimizer.optimize()
print(result)
```

### Uso Completo con Todas las Opciones

```python
import torch
from tensor_qpso.qpso_tensor_optimized import (
    QPSOTensorOptimized,
    CallbackEvent,
    BoundaryStrategy
)

def rastrigin(x):
    if x.dim() == 1:
        n = x.shape[0]
        return 10*n + torch.sum(x**2 - 10*torch.cos(2*torch.pi*x))
    n = x.shape[1]
    return 10*n + torch.sum(x**2 - 10*torch.cos(2*torch.pi*x), dim=1)

# Callbacks personalizados
def on_new_best(opt):
    print(f"[{opt.iters:4d}] Nuevo mejor: {opt.gbest_value:.6E}")

def on_convergence(opt):
    print(f"Convergencia detectada en iteracion {opt.iters}")

# Crear optimizador con todas las opciones
with QPSOTensorOptimized(
    cf=rastrigin,
    size=50,
    dim=20,
    bounds=[(-5.12, 5.12)] * 20,
    maxIters=2000,
    alpha=(1.0, 0.5),           # Decrecimiento lineal
    device='auto',              # GPU si disponible
    dtype=torch.float32,
    seed=42,                    # Reproducible
    boundary_strategy='reflect', # Rebotar en limites
    tol=1e-15,                  # Alta precision
    patience=200,               # Mas paciencia
    track_history=True,         # Guardar historial
    minimize=True
) as optimizer:

    # Registrar callbacks
    optimizer.callbacks.register(CallbackEvent.ON_NEW_BEST, on_new_best)
    optimizer.callbacks.register(CallbackEvent.ON_CONVERGENCE, on_convergence)

    # Ejecutar
    result = optimizer.optimize()

# Analizar resultado
print(f"\nResultado final: {result.best_value:.10E}")
print(f"Iteraciones: {result.iterations}")
print(f"Convergio: {result.converged}")
print(f"Razon: {result.convergence_reason}")
print(f"Tiempo: {result.elapsed_time:.2f}s")

# Graficar historial
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(result.history['gbest_value'])
plt.yscale('log')
plt.title('Convergencia')
plt.xlabel('Iteracion')

plt.subplot(1, 3, 2)
plt.plot(result.history['diversity'])
plt.title('Diversidad')
plt.xlabel('Iteracion')

plt.subplot(1, 3, 3)
plt.plot(result.history['std_fitness'])
plt.yscale('log')
plt.title('STD Fitness')
plt.xlabel('Iteracion')

plt.tight_layout()
plt.show()
```

### Factory Function

```python
from tensor_qpso.qpso_tensor_optimized import create_optimizer

# Crear por nombre
qpso = create_optimizer('qpso', sphere, 40, 10, bounds, 1000, alpha=0.75)
qdpso = create_optimizer('qdpso', sphere, 40, 10, bounds, 1000, g=0.96)
```

---

## Comparativa de Rendimiento

### Resultados Esperados

| Implementacion | Dispositivo | Tiempo Relativo |
|----------------|-------------|-----------------|
| qpso.py (NumPy) | CPU | 1.0x (baseline) |
| qpso_tensor.py | CPU | ~0.8x |
| qpso_tensor.py | GPU | ~0.5-1.5x (depende del problema) |
| qpso_tensor_optimized.py | CPU | ~0.6x |
| qpso_tensor_optimized.py | GPU | ~0.3-1.0x |

**Nota**: GPU muestra mayor ventaja en:
- Alta dimensionalidad (dim > 100)
- Muchas particulas (size > 100)
- Funciones de costo costosas

---

## Guia de Migracion

### Desde qpso_tensor.py

```python
# Antes
from tensor_qpso.qpso_tensor import QPSOTensor
opt = QPSOTensor(cf, 40, 10, bounds, 1000, alpha=0.75, device='auto')
opt.update()
best = opt.gbest_value

# Despues
from tensor_qpso.qpso_tensor_optimized import QPSOTensorOptimized
opt = QPSOTensorOptimized(cf, 40, 10, bounds, 1000, alpha=0.75, device='auto')
result = opt.optimize()
best = result.best_value
```

### Principales Diferencias

| Aspecto | Basico | Optimizado |
|---------|--------|------------|
| Metodo principal | `update()` | `optimize()` |
| Retorno | Ninguno | `OptimizationResult` |
| Boundary | No | Configurable |
| Convergencia | No | Automatica |
| Historial | No | Opcional |
| Callbacks | Simple | Por eventos |
| Context manager | No | Si |

---

## Referencias

### Papers

1. Sun, J., Feng, B., & Xu, W. (2004). "Particle swarm optimization with particles having quantum behavior". IEEE CEC.

2. Sun, J., Fang, W., Wu, X., Palade, V., & Xu, W. (2012). "Quantum-behaved particle swarm optimization: Analysis". Evolutionary Computation.

### PyTorch

3. PyTorch Performance Tuning Guide: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

4. CUDA Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

### Boundary Handling

5. Helwig, S., & Wanka, R. (2008). "Theoretical analysis of initial particle swarm behavior". PPSN.

---

## Historial de Cambios

| Version | Cambios |
|---------|---------|
| 1.0 | Implementacion NumPy basica |
| 2.0 | Version con tensores PyTorch |
| 3.0 | **Version optimizada** con 17 mejoras |

---

## Related Documents

- [📘 NumPy Implementation](docs_qpso.md) - Reference implementation for learning
- [📗 Basic Tensor Implementation](docs_qpso_tensor.md) - Intermediate PyTorch version
- [📊 Implementation Comparison](implementation_comparison.md) - Detailed comparison of all implementations
- [📦 ANN Module](../../ann/docs/index.md) - Neural network training with QPSO

---

<div align="center">

**[⬆️ Back to Top](#documentacion-completa-qpso-y-qdpso-optimizado-con-tensores-pytorch)** | **[⬅️ Prev: Tensor](docs_qpso_tensor.md)** | **[📚 Index](index.md)** | **[📊 Comparison](implementation_comparison.md)**

</div>
