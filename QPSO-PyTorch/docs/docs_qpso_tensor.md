# Documentacion Completa: QPSO y QDPSO con Tensores PyTorch

[🏠 README](../../README.md) | [📚 Index](index.md) | [⬅️ Prev: NumPy](docs_qpso.md) | **Tensor Basic** | [Next: Optimized ➡️](docs_qpso_tensor_optimized.md)

---

## Descripcion General

> Esta implementacion utiliza **tensores PyTorch** para ejecutar QPSO y QDPSO con:
> - **Operaciones vectorizadas**: Sin loops explicitos, todas las particulas se procesan en paralelo
> - **Aceleracion por GPU**: Soporte completo para CUDA
> - **Flexibilidad de dispositivo**: Seleccion entre CPU, GPU, o automatica

---

## Tabla de Contenidos

1. [Introduccion](#introduccion)
2. [Diferencias con la Version NumPy](#diferencias-con-la-version-numpy)
3. [Arquitectura de Clases](#arquitectura-de-clases)
4. [Seleccion de Dispositivo](#seleccion-de-dispositivo)
5. [Estructura de Tensores](#estructura-de-tensores)
6. [QPSOTensor (Original Paper)](#qpsotensor-original-paper)
7. [QDPSOTensor (Variante Delta)](#qdpsotensor-variante-delta)
8. [Funciones de Costo Vectorizadas](#funciones-de-costo-vectorizadas)
9. [Parametros y Configuracion](#parametros-y-configuracion)
10. [Ejemplos de Uso](#ejemplos-de-uso)
11. [Rendimiento: GPU vs CPU](#rendimiento-gpu-vs-cpu)
12. [Guia de Optimizacion](#guia-de-optimizacion)
13. [Troubleshooting](#troubleshooting)
14. [Referencias](#referencias)

---

## Introduccion

La implementacion con tensores PyTorch transforma los algoritmos QPSO y QDPSO de un enfoque **iterativo** (particula por particula, dimension por dimension) a un enfoque **vectorizado** (todas las operaciones en paralelo usando tensores).

### Ventajas de la Version con Tensores

| Aspecto | Version NumPy | Version PyTorch |
|---------|--------------|-----------------|
| **Procesamiento** | Secuencial (loops) | Paralelo (vectorizado) |
| **Hardware** | Solo CPU | CPU y GPU (CUDA) |
| **Escalabilidad** | Limitada | Alta dimensionalidad |
| **Memoria** | Objetos individuales | Tensores contiguos |
| **Gradientes** | No disponible | Disponible (autograd) |

### Requisitos

```bash
# Activar ambiente conda
conda activate pytorch_qpso_gpu

# Verificar instalacion
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Diferencias con la Version NumPy

### Comparativa de Implementaciones

| Aspecto | `qpso.py` (NumPy) | `qpso_tensor.py` (PyTorch) |
|---------|-------------------|---------------------------|
| **Estructura de datos** | Lista de objetos `Particle` | Tensores `(n_particles, dim)` |
| **Posiciones** | `particle._x` (array 1D) | `self._positions` (tensor 2D) |
| **pbest** | `particle._best` por particula | `self._pbest` tensor 2D |
| **Actualizacion** | Loop doble (particulas x dimensiones) | Operacion matricial unica |
| **Numeros aleatorios** | `random.uniform()` por valor | `torch.rand()` tensor completo |
| **Device** | Solo CPU | CPU, CUDA, o automatico |

### Ejemplo de Diferencia en kernel_update

**Version NumPy (iterativa):**
```python
for p in self._particles:           # Loop por particulas
    for i in range(self._dim):      # Loop por dimensiones
        phi = random.uniform(0., 1.)
        # ... operaciones escalares
```

**Version PyTorch (vectorizada):**
```python
# Sin loops - todas las particulas y dimensiones en paralelo
phi = torch.rand(n_particles, dim, device=self._device)
c = phi * self._pbest + (1 - phi) * self._gbest
# ... operaciones tensoriales
```

---

## Arquitectura de Clases

```
SwarmTensor              <- Enjambre con tensores (n_particles, dim)
     |
     v
QPSOBaseTensor           <- Clase base abstracta (logica comun)
     |
     +--------+--------+
     |                 |
     v                 v
QPSOTensor         QDPSOTensor
(Original)         (Variante)
```

### Descripcion de Clases

| Clase | Archivo | Descripcion |
|-------|---------|-------------|
| `SwarmTensor` | `qpso_tensor.py` | Gestiona tensores de posiciones, pbest, gbest |
| `QPSOBaseTensor` | `qpso_tensor.py` | Logica comun: evaluacion, actualizacion, loop principal |
| `QPSOTensor` | `qpso_tensor.py` | QPSO original con mbest (vectorizado) |
| `QDPSOTensor` | `qpso_tensor.py` | Variante QDPSO (vectorizado) |

### Funcion Auxiliar

| Funcion | Descripcion |
|---------|-------------|
| `get_device(device)` | Convierte string a `torch.device` |

---

## Seleccion de Dispositivo

### Opciones de Device

| Valor | Descripcion | Uso |
|-------|-------------|-----|
| `'cpu'` | Fuerza ejecucion en CPU | Depuracion, compatibilidad |
| `'cuda'` | Fuerza ejecucion en GPU (default) | Maxima velocidad con GPU |
| `'cuda:0'` | GPU especifica (indice 0) | Sistemas multi-GPU |
| `'cuda:1'` | GPU especifica (indice 1) | Sistemas multi-GPU |
| `'auto'` | GPU si disponible, sino CPU | **Recomendado** |

### Funcion get_device()

```python
from tensor_qpso.qpso_tensor import get_device

# Uso
device = get_device('auto')
print(device)  # cuda o cpu segun disponibilidad
```

### Implementacion Interna

```python
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
```

### Verificar Dispositivo Actual

```python
optimizer = QPSOTensor(cf, size, dim, bounds, maxIters, device='auto')
print(f"Ejecutando en: {optimizer.device}")
```

---

## Estructura de Tensores

### Tensores del Enjambre (SwarmTensor)

| Tensor | Forma | Descripcion |
|--------|-------|-------------|
| `_positions` | `(n_particles, dim)` | Posiciones actuales de todas las particulas |
| `_pbest` | `(n_particles, dim)` | Mejores posiciones personales |
| `_pbest_values` | `(n_particles,)` | Valores de fitness de pbest |
| `_gbest` | `(dim,)` | Mejor posicion global |
| `_lower` | `(dim,)` | Limites inferiores por dimension |
| `_upper` | `(dim,)` | Limites superiores por dimension |

### Diagrama de Tensores

```
positions (n_particles=4, dim=3):

          dim_0   dim_1   dim_2
        +-------+-------+-------+
part_0  | x_0,0 | x_0,1 | x_0,2 |
        +-------+-------+-------+
part_1  | x_1,0 | x_1,1 | x_1,2 |
        +-------+-------+-------+
part_2  | x_2,0 | x_2,1 | x_2,2 |
        +-------+-------+-------+
part_3  | x_3,0 | x_3,1 | x_3,2 |
        +-------+-------+-------+

pbest_values (n_particles=4):
        +-------+-------+-------+-------+
        | f_0   | f_1   | f_2   | f_3   |
        +-------+-------+-------+-------+

gbest (dim=3):
        +-------+-------+-------+
        | g_0   | g_1   | g_2   |
        +-------+-------+-------+
```

### Inicializacion de Tensores

```python
# Crear tensores de limites
bounds_tensor = torch.tensor(bounds, dtype=torch.float32, device=device)
lower = bounds_tensor[:, 0]  # (dim,)
upper = bounds_tensor[:, 1]  # (dim,)

# Inicializar posiciones aleatorias dentro de los limites
positions = torch.rand(size, dim, device=device) * (upper - lower) + lower

# Inicializar pbest
pbest = positions.clone()
pbest_values = torch.full((size,), float('inf'), device=device)

# Inicializar gbest
gbest = torch.zeros(dim, device=device)
gbest_value = float('inf')
```

---

## QPSOTensor (Original Paper)

### Formula Matematica (Vectorizada)

Las mismas formulas del paper, pero aplicadas a tensores completos:

```
1. mbest = mean(pbest, dim=0)                    # (dim,)

2. phi ~ U(0, 1)                                  # (n_particles, dim)

3. c = phi * pbest + (1 - phi) * gbest           # (n_particles, dim)

4. alpha = alpha_max - (alpha_max - alpha_min) * t / T

5. L = alpha * |mbest - positions|               # (n_particles, dim)

6. positions = c +/- L * ln(1/u)                 # (n_particles, dim)
```

### Implementacion Vectorizada

```python
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

    # Generar numeros aleatorios para TODAS las particulas y dimensiones
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
    # Broadcasting: gbest (dim,) se expande a (n, dim)
    c = phi * self._pbest + (1 - phi) * self._gbest  # (n, d)

    # Longitud caracteristica usando mbest
    # Broadcasting: mbest (dim,) se expande a (n, dim)
    L = alpha * torch.abs(mbest - self._positions)  # (n, d)

    # Nueva posicion con distribucion de Laplace
    self._positions = c + rand_sign * L * torch.log(1.0 / u)
```

### Broadcasting en PyTorch

La clave de la eficiencia es el **broadcasting**:

```
gbest:       (dim,)        ->  se expande a  (n_particles, dim)
mbest:       (dim,)        ->  se expande a  (n_particles, dim)
pbest:       (n_particles, dim)
positions:   (n_particles, dim)

Operacion: c = phi * pbest + (1 - phi) * gbest
           (n,d)   (n,d)              (dim,) <- broadcasting automatico
```

---

## QDPSOTensor (Variante Delta)

### Formula Matematica (Vectorizada)

```
1. u1, u2, u3 ~ U(0, 1)                          # (n_particles, dim) cada uno

2. c = (u1 * pbest + u2 * gbest) / (u1 + u2)     # (n_particles, dim)

3. L = (1/g) * |positions - c|                   # (n_particles, dim)

4. positions = c +/- L * ln(1/u3)                # (n_particles, dim)
```

### Implementacion Vectorizada

```python
def kernel_update(self) -> None:
    """
    Actualiza posiciones usando la formula QDPSO (vectorizado).
    Todas las operaciones se realizan en tensores para maxima eficiencia.
    """
    n = self._size
    d = self._dim

    # Generar numeros aleatorios para TODAS las particulas y dimensiones
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
    # Broadcasting: gbest (dim,) se expande a (n, dim)
    c = (u1 * self._pbest + u2 * self._gbest) / (u1 + u2)  # (n, d)

    # Longitud caracteristica usando distancia al punto atractor
    L = (1.0 / self._g) * torch.abs(self._positions - c)  # (n, d)

    # Nueva posicion con distribucion de Laplace
    self._positions = c + rand_sign * L * torch.log(1.0 / u3)
```

---

## Funciones de Costo Vectorizadas

### Concepto

Las funciones de costo pueden implementarse de dos formas:

| Tipo | Entrada | Salida | Eficiencia |
|------|---------|--------|------------|
| **Individual** | `(dim,)` | escalar | Baja (requiere loop) |
| **Vectorizada** | `(n_particles, dim)` | `(n_particles,)` | Alta (paralelo) |

El optimizador detecta automaticamente el tipo de funcion.

### Funcion Esfera (Ejemplo)

**Version Individual (no recomendada):**
```python
def sphere_individual(x: torch.Tensor) -> torch.Tensor:
    """Recibe (dim,), retorna escalar."""
    return torch.sum(x ** 2)
```

**Version Vectorizada (recomendada):**
```python
def sphere_vectorized(x: torch.Tensor) -> torch.Tensor:
    """
    Recibe (n_particles, dim), retorna (n_particles,).
    Tambien funciona con (dim,) retornando escalar.
    """
    if x.dim() == 1:
        return torch.sum(x ** 2)
    return torch.sum(x ** 2, dim=1)  # Suma por filas
```

### Funciones de Benchmark Vectorizadas

#### Esfera (Sphere)
```python
def sphere(x: torch.Tensor) -> torch.Tensor:
    """f(x) = sum(x_i^2), minimo en (0, 0, ..., 0)"""
    if x.dim() == 1:
        return torch.sum(x ** 2)
    return torch.sum(x ** 2, dim=1)
```

#### Rastrigin
```python
def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i)), minimo en (0, 0, ..., 0)"""
    if x.dim() == 1:
        n = x.shape[0]
        return 10 * n + torch.sum(x ** 2 - 10 * torch.cos(2 * torch.pi * x))
    n = x.shape[1]
    return 10 * n + torch.sum(x ** 2 - 10 * torch.cos(2 * torch.pi * x), dim=1)
```

#### Rosenbrock
```python
def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    """f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2), minimo en (1, 1, ..., 1)"""
    if x.dim() == 1:
        return torch.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    return torch.sum(
        100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2,
        dim=1
    )
```

#### Ackley
```python
def ackley(x: torch.Tensor) -> torch.Tensor:
    """Funcion Ackley, minimo en (0, 0, ..., 0)"""
    a, b, c = 20, 0.2, 2 * torch.pi
    if x.dim() == 1:
        d = x.shape[0]
        sum1 = torch.sum(x ** 2)
        sum2 = torch.sum(torch.cos(c * x))
        return -a * torch.exp(-b * torch.sqrt(sum1 / d)) - torch.exp(sum2 / d) + a + torch.e
    d = x.shape[1]
    sum1 = torch.sum(x ** 2, dim=1)
    sum2 = torch.sum(torch.cos(c * x), dim=1)
    return -a * torch.exp(-b * torch.sqrt(sum1 / d)) - torch.exp(sum2 / d) + a + torch.e
```

### Deteccion Automatica

El optimizador detecta automaticamente si la funcion es vectorizada:

```python
def _evaluate(self, positions: torch.Tensor) -> torch.Tensor:
    if self._vectorized_cf is None:
        # Primera evaluacion: detectar tipo
        try:
            result = self._cf(positions)
            if positions.dim() == 2 and result.dim() == 1:
                self._vectorized_cf = True
                return result
            else:
                self._vectorized_cf = False
        except Exception:
            self._vectorized_cf = False

    if self._vectorized_cf:
        return self._cf(positions)
    else:
        # Fallback: evaluar una por una
        return torch.stack([self._cf(p) for p in positions])
```

---

## Parametros y Configuracion

### Parametros QPSOTensor

| Parametro | Tipo | Descripcion | Valor Recomendado |
|-----------|------|-------------|-------------------|
| `cf` | `Callable` | Funcion de costo (vectorizada preferida) | - |
| `size` | `int` | Numero de particulas | 20-100 |
| `dim` | `int` | Dimensionalidad del problema | Depende del problema |
| `bounds` | `List[Tuple]` | Limites `[(min, max), ...]` | Depende del problema |
| `maxIters` | `int` | Iteraciones maximas | 500-2000 |
| `alpha` | `float` o `Tuple` | Coeficiente alpha | `0.75` o `(1.0, 0.5)` |
| `device` | `str` | Dispositivo: `'auto'`, `'cpu'`, `'cuda'` | `'auto'` |

### Parametros QDPSOTensor

| Parametro | Tipo | Descripcion | Valor Recomendado |
|-----------|------|-------------|-------------------|
| `cf` | `Callable` | Funcion de costo (vectorizada preferida) | - |
| `size` | `int` | Numero de particulas | 20-100 |
| `dim` | `int` | Dimensionalidad del problema | Depende del problema |
| `bounds` | `List[Tuple]` | Limites `[(min, max), ...]` | Depende del problema |
| `maxIters` | `int` | Iteraciones maximas | 500-2000 |
| `g` | `float` | Coeficiente g | `0.96` |
| `device` | `str` | Dispositivo: `'auto'`, `'cpu'`, `'cuda'` | `'auto'` |

### Propiedades Disponibles

| Propiedad | Tipo | Descripcion |
|-----------|------|-------------|
| `device` | `torch.device` | Dispositivo actual |
| `size` | `int` | Numero de particulas |
| `dim` | `int` | Dimensionalidad |
| `positions` | `Tensor (n, d)` | Posiciones actuales |
| `pbest` | `Tensor (n, d)` | Mejores posiciones personales |
| `pbest_values` | `Tensor (n,)` | Valores de fitness de pbest |
| `gbest` | `Tensor (d,)` | Mejor posicion global |
| `gbest_value` | `float` | Mejor valor de fitness |
| `iters` | `int` | Iteracion actual |
| `maxIters` | `int` | Iteraciones maximas |

---

## Ejemplos de Uso

### Ejemplo Basico: CPU

```python
import torch
from tensor_qpso.qpso_tensor import QPSOTensor

# Funcion de costo vectorizada
def sphere(x):
    if x.dim() == 1:
        return torch.sum(x ** 2)
    return torch.sum(x ** 2, dim=1)

# Configuracion
size = 40
dim = 10
bounds = [(-5.12, 5.12) for _ in range(dim)]
maxIters = 1000

# Crear optimizador en CPU
optimizer = QPSOTensor(
    cf=sphere,
    size=size,
    dim=dim,
    bounds=bounds,
    maxIters=maxIters,
    alpha=0.75,
    device='cpu'
)

# Ejecutar
optimizer.update()

# Resultados
print(f"Mejor valor: {optimizer.gbest_value}")
print(f"Mejor posicion: {optimizer.gbest}")
```

### Ejemplo con GPU

```python
# Crear optimizador en GPU
optimizer = QPSOTensor(
    cf=sphere,
    size=40,
    dim=10,
    bounds=[(-5.12, 5.12) for _ in range(10)],
    maxIters=1000,
    alpha=(1.0, 0.5),  # Alpha decreciente
    device='cuda'  # Forzar GPU
)

optimizer.update()
print(f"Ejecutado en: {optimizer.device}")
print(f"Mejor valor: {optimizer.gbest_value}")
```

### Ejemplo con Seleccion Automatica

```python
# Usa GPU si esta disponible, sino CPU
optimizer = QPSOTensor(
    cf=sphere,
    size=40,
    dim=10,
    bounds=[(-5.12, 5.12) for _ in range(10)],
    maxIters=1000,
    alpha=0.75,
    device='auto'  # Seleccion automatica
)

print(f"Dispositivo seleccionado: {optimizer.device}")
optimizer.update()
```

### Ejemplo con Callback

```python
def log_progress(optimizer):
    """Callback para monitorear progreso."""
    if optimizer.iters % 100 == 0:
        print(f"Iter {optimizer.iters:4d}: "
              f"Best = {optimizer.gbest_value:.6E}, "
              f"Mean = {optimizer.pbest_values.mean():.6E}")

optimizer = QPSOTensor(
    cf=sphere,
    size=40,
    dim=10,
    bounds=[(-5.12, 5.12) for _ in range(10)],
    maxIters=1000,
    alpha=0.75,
    device='auto'
)

# Ejecutar con callback cada 100 iteraciones
optimizer.update(callback=log_progress, interval=100)
```

### Ejemplo QDPSOTensor

```python
from tensor_qpso.qpso_tensor import QDPSOTensor

optimizer = QDPSOTensor(
    cf=sphere,
    size=40,
    dim=10,
    bounds=[(-5.12, 5.12) for _ in range(10)],
    maxIters=1000,
    g=0.96,
    device='auto'
)

optimizer.update()
print(f"Mejor valor: {optimizer.gbest_value}")
```

### Ejemplo Comparativo: QPSO vs QDPSO

```python
import time
from tensor_qpso.qpso_tensor import QPSOTensor, QDPSOTensor

def sphere(x):
    if x.dim() == 1:
        return torch.sum(x ** 2)
    return torch.sum(x ** 2, dim=1)

config = {
    'size': 40,
    'dim': 10,
    'bounds': [(-5.12, 5.12) for _ in range(10)],
    'maxIters': 1000,
    'device': 'auto'
}

# QPSO
start = time.time()
qpso = QPSOTensor(cf=sphere, **config, alpha=(1.0, 0.5))
qpso.update()
qpso_time = time.time() - start

# QDPSO
start = time.time()
qdpso = QDPSOTensor(cf=sphere, **config, g=0.96)
qdpso.update()
qdpso_time = time.time() - start

print(f"QPSO:  {qpso.gbest_value:.6E} en {qpso_time:.3f}s")
print(f"QDPSO: {qdpso.gbest_value:.6E} en {qdpso_time:.3f}s")
```

---

## Rendimiento: GPU vs CPU

### Factores que Afectan el Rendimiento

| Factor | Favorece GPU | Favorece CPU |
|--------|--------------|--------------|
| Dimensionalidad | Alta (>100) | Baja (<50) |
| Numero de particulas | Alto (>100) | Bajo (<50) |
| Funcion de costo | Costosa computacionalmente | Simple |
| Iteraciones | Muchas (>1000) | Pocas (<500) |
| Transferencias | Pocas | Muchas |

### Overhead de GPU

La GPU tiene overhead por:
1. **Transferencia de datos**: CPU <-> GPU
2. **Sincronizacion**: `torch.cuda.synchronize()`
3. **Lanzamiento de kernels**: Inicializacion de operaciones

Para problemas pequenos, este overhead puede superar el beneficio.

### Resultados de Benchmark (NVIDIA GTX 1050 Ti)

```
Configuracion: size=40, dim=10, maxIters=1000

Algoritmo                      Device   Tiempo
----------------------------------------------
QPSOTensor (alpha=1.0->0.5)    CPU      0.269s
QDPSOTensor (g=0.96)           CPU      0.176s
QPSOTensor (alpha=1.0->0.5)    GPU      1.067s
QDPSOTensor (g=0.96)           GPU      0.583s

Configuracion: size=100, dim=100, maxIters=500

Algoritmo                      Device   Tiempo
----------------------------------------------
QPSOTensor                     CPU      0.208s
QPSOTensor                     GPU      0.259s
```

### Cuando Usar GPU

| Escenario | Recomendacion |
|-----------|---------------|
| `dim < 50, size < 50` | CPU |
| `dim > 100, size > 100` | GPU |
| Funcion de costo simple | CPU |
| Funcion de costo con redes neuronales | GPU |
| Multiples ejecuciones | GPU (amortiza overhead) |

---

## Guia de Optimizacion

### Optimizar Funcion de Costo

```python
# MAL: Loop interno
def sphere_slow(x):
    total = 0
    for i in range(x.shape[-1]):
        total += x[..., i] ** 2
    return total

# BIEN: Operacion vectorizada
def sphere_fast(x):
    return torch.sum(x ** 2, dim=-1 if x.dim() > 1 else None)
```

### Evitar Transferencias CPU-GPU

```python
# MAL: Transferencia en cada callback
def bad_callback(opt):
    # .cpu() crea transferencia
    print(opt.gbest.cpu().numpy())

# BIEN: Solo transferir al final
def good_callback(opt):
    # .item() para escalares es eficiente
    print(opt.gbest_value)

# Transferir resultados solo al final
optimizer.update()
final_result = optimizer.gbest.cpu().numpy()
```

### Warmup de GPU

```python
# La primera operacion GPU tiene overhead de inicializacion
# Hacer warmup antes de medir tiempos

import torch
_ = torch.rand(100, 100, device='cuda')  # Warmup

# Ahora medir
start = time.time()
optimizer.update()
torch.cuda.synchronize()  # Esperar a que GPU termine
elapsed = time.time() - start
```

### Precision Numerica

```python
# Los tensores usan float32 por defecto
# Para mayor precision, usar float64 (mas lento)

# Modificar dtype en SwarmTensor.__init__:
self._positions = torch.rand(size, dim, dtype=torch.float64, device=device)
```

---

## Troubleshooting

### Error: CUDA out of memory

**Causa**: Problema demasiado grande para la memoria GPU.

**Solucion**:
```python
# Reducir tamano
optimizer = QPSOTensor(cf, size=50, dim=100, ...)  # Menos particulas

# O usar CPU
optimizer = QPSOTensor(cf, size=200, dim=500, ..., device='cpu')

# Liberar memoria
torch.cuda.empty_cache()
```

### Error: Expected all tensors to be on the same device

**Causa**: Funcion de costo crea tensores en CPU cuando el optimizador esta en GPU.

**Solucion**:
```python
# MAL
def bad_func(x):
    constant = torch.tensor([1, 2, 3])  # CPU por defecto
    return torch.sum((x - constant) ** 2)

# BIEN
def good_func(x):
    constant = torch.tensor([1, 2, 3], device=x.device)  # Mismo device
    return torch.sum((x - constant) ** 2, dim=-1 if x.dim() > 1 else None)
```

### Resultados Diferentes entre CPU y GPU

**Causa**: Diferentes generadores de numeros aleatorios.

**Solucion**:
```python
# Fijar semilla para reproducibilidad
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
```

### GPU no Detectada

**Causa**: CUDA no instalado o version incompatible.

**Verificacion**:
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## Flujo de Ejecucion

```
INICIALIZACION
     |
     +-- get_device(device) -> torch.device
     +-- Crear tensores en device:
     |       positions: (n_particles, dim) aleatorio
     |       pbest: clone de positions
     |       pbest_values: (n_particles,) = inf
     |       gbest: (dim,) = zeros
     +-- init_eval():
     |       Evaluar cf(positions) -> pbest_values
     |       update_gbest()
     |
LOOP PRINCIPAL (t = 0 hasta maxIters)
     |
     +-- kernel_update():
     |       [QPSOTensor]
     |       |   mbest = mean(pbest, dim=0)
     |       |   alpha = _get_alpha()
     |       |   phi, u, rand_sign = torch.rand(...)
     |       |   c = phi * pbest + (1-phi) * gbest
     |       |   L = alpha * |mbest - positions|
     |       |   positions = c + rand_sign * L * log(1/u)
     |       |
     |       [QDPSOTensor]
     |           u1, u2, u3, rand_sign = torch.rand(...)
     |           c = (u1 * pbest + u2 * gbest) / (u1 + u2)
     |           L = (1/g) * |positions - c|
     |           positions = c + rand_sign * L * log(1/u3)
     |
     +-- update_best():
     |       current_values = cf(positions)
     |       improved = current_values < pbest_values
     |       pbest[improved] = positions[improved]
     |       pbest_values[improved] = current_values[improved]
     |       update_gbest()
     |
     +-- callback(self) si corresponde
     +-- t = t + 1
     |
RESULTADO
     |
     +-- gbest: mejor posicion encontrada
     +-- gbest_value: mejor valor de fitness
```

---

## Referencias

### Papers Originales

1. **QPSO Original**:
   Sun, J., Feng, B., & Xu, W. (2004). "Particle swarm optimization with particles having quantum behavior". *IEEE Congress on Evolutionary Computation*.

2. **Analisis de QPSO**:
   Sun, J., Fang, W., Wu, X., Palade, V., & Xu, W. (2012). "Quantum-behaved particle swarm optimization: Analysis of individual particle behavior and parameter selection". *Evolutionary Computation*, 20(3).

### PyTorch

3. **Documentacion PyTorch**:
   https://pytorch.org/docs/stable/index.html

4. **CUDA Semantics**:
   https://pytorch.org/docs/stable/notes/cuda.html

5. **Broadcasting**:
   https://pytorch.org/docs/stable/notes/broadcasting.html

---

## Historial de Cambios

| Version | Cambio |
|---------|--------|
| 1.0 | Implementacion inicial con NumPy |
| 2.0 | Agregada version con tensores PyTorch |
| 2.0 | Soporte para GPU (CUDA) |
| 2.0 | Operaciones vectorizadas |
| 2.0 | Deteccion automatica de funcion vectorizada |
| 2.0 | Documentacion completa de version tensor |

---

## Related Documents

- [📘 NumPy Implementation](docs_qpso.md) - Reference implementation for learning
- [📙 Optimized Implementation](docs_qpso_tensor_optimized.md) - Production-ready with 17 improvements
- [📊 Implementation Comparison](implementation_comparison.md) - Detailed comparison of all implementations

---

<div align="center">

**[⬆️ Back to Top](#documentacion-completa-qpso-y-qdpso-con-tensores-pytorch)** | **[⬅️ Prev: NumPy](docs_qpso.md)** | **[📚 Index](index.md)** | **[Next: Optimized ➡️](docs_qpso_tensor_optimized.md)**

</div>
