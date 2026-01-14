# Modulo Optimizers - Documentacion

[🏠 README](../../README.md) | [📚 Index](index.md) | [⬅️ Models](models.md) | **Optimizers** | [Next: Trainers ➡️](trainers.md)

---

## Descripcion General

El modulo `optimizers` contiene los optimizadores basados en QPSO para entrenar redes neuronales. Estos optimizadores actuan como puente entre el modelo de red neuronal (`QPSOCompatibleANN`) y los algoritmos QPSO implementados en `tensor_qpso/qpso_tensor_optimized.py`.

**Ubicacion:** `ann/optimizers/`

**Archivo principal:** `qpso_nn.py`

---

## Arquitectura del Optimizador

```
┌─────────────────────────────────────────────────────────────────┐
│                      QPSONNOptimizer                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────────────────────┐  │
│  │ QPSOCompatibleANN│    │  QPSOTensorOptimized             │  │
│  │                  │◄───│  (de qpso_tensor_optimized.py)   │  │
│  │  - get_flat_params│    │                                  │  │
│  │  - set_flat_params│    │  - Particulas = Pesos de red    │  │
│  │  - forward        │    │  - Fitness = Loss de la red     │  │
│  └──────────────────┘    └──────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Fitness Function                       │  │
│  │  Para cada particula:                                     │  │
│  │    1. model.set_flat_params(particula)                   │  │
│  │    2. outputs = model(X_train)                           │  │
│  │    3. loss = loss_fn(outputs, y_train)                   │  │
│  │    4. return loss (para minimizar)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Clases Disponibles

| Clase | Descripcion |
|-------|-------------|
| `QPSONNOptimizer` | Optimizador QPSO para redes neuronales |
| `QDPSONNOptimizer` | Optimizador QDPSO para redes neuronales |
| `NNOptimizationConfig` | Configuracion del optimizador |

---

## NNOptimizationConfig

### Descripcion

Dataclass que centraliza toda la configuracion del optimizador.

### Importacion

```python
from ann.optimizers.qpso_nn import NNOptimizationConfig
```

### Definicion

```python
@dataclass
class NNOptimizationConfig:
    n_particles: int = 50
    max_iters: int = 100
    alpha: Union[float, Tuple[float, float]] = (1.0, 0.5)
    g: float = 0.96
    weight_bound: float = 1.0
    boundary_strategy: str = "clamp"
    tol: float = 1e-12
    patience: int = 50
    seed: Optional[int] = None
    track_history: bool = True
    device: str = "auto"
    dtype: torch.dtype = torch.float32
```

### Parametros Detallados

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `n_particles` | `int` | `50` | Numero de particulas en el enjambre |
| `max_iters` | `int` | `100` | Numero maximo de iteraciones |
| `alpha` | `float` o `tuple` | `(1.0, 0.5)` | Factor de contraccion-expansion (QPSO) |
| `g` | `float` | `0.96` | Factor de control (QDPSO) |
| `weight_bound` | `float` | `1.0` | Limite para los pesos de la red |
| `boundary_strategy` | `str` | `"clamp"` | Estrategia de limites |
| `tol` | `float` | `1e-12` | Tolerancia para convergencia |
| `patience` | `int` | `50` | Iteraciones sin mejora antes de parar |
| `seed` | `int` | `None` | Semilla para reproducibilidad |
| `track_history` | `bool` | `True` | Registrar historial de entrenamiento |
| `device` | `str` | `"auto"` | Dispositivo: `"cpu"`, `"cuda"`, `"auto"` |
| `dtype` | `torch.dtype` | `float32` | Tipo de datos para tensores |

### Estrategias de Limites (Boundary Strategies)

| Estrategia | Descripcion | Cuando Usar |
|------------|-------------|-------------|
| `"clamp"` | Limita al rango [min, max] | Default, mas estable |
| `"reflect"` | Rebota en los limites | Exploracion activa |
| `"wrap"` | Envuelve al otro extremo | Espacios periodicos |
| `"random"` | Posicion aleatoria en rango | Evitar estancamiento |
| `"none"` | Sin limites | No recomendado |

### Ejemplo de Configuracion

```python
from ann.optimizers.qpso_nn import NNOptimizationConfig

# Configuracion basica
config = NNOptimizationConfig(
    n_particles=30,
    max_iters=100
)

# Configuracion avanzada
config_advanced = NNOptimizationConfig(
    n_particles=50,
    max_iters=200,
    alpha=(1.0, 0.5),      # Alpha con decay lineal
    weight_bound=1.5,       # Pesos entre -1.5 y 1.5
    boundary_strategy="reflect",
    tol=1e-10,
    patience=100,
    seed=42,
    track_history=True,
    device="cuda"
)
```

---

## QPSONNOptimizer

### Descripcion

Optimizador principal que utiliza QPSO para entrenar redes neuronales. Internamente crea una funcion de fitness que evalua la perdida de la red para cada configuracion de pesos (particula).

### Importacion

```python
from ann.optimizers import QPSONNOptimizer
# o
from ann.optimizers.qpso_nn import QPSONNOptimizer
```

### Constructor

```python
QPSONNOptimizer(
    model: nn.Module,
    loss_fn: Optional[Callable] = None,
    config: Optional[NNOptimizationConfig] = None,
    use_qdpso: bool = False
)
```

### Parametros del Constructor

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | - | Red neuronal compatible (debe tener `get_flat_params`/`set_flat_params`) |
| `loss_fn` | `Callable` | `CrossEntropyLoss` | Funcion de perdida |
| `config` | `NNOptimizationConfig` | Default config | Configuracion del optimizador |
| `use_qdpso` | `bool` | `False` | Si True, usa QDPSO en lugar de QPSO |

---

## Metodos Principales

### fit()

Entrena la red neuronal usando QPSO.

```python
def fit(
    self,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    fitness_type: str = "loss",
    callback: Optional[Callable] = None,
    verbose: bool = True
) -> OptimizationResult
```

**Parametros:**

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `X_train` | `torch.Tensor` | - | Features de entrenamiento `[n_samples, n_features]` |
| `y_train` | `torch.Tensor` | - | Labels de entrenamiento `[n_samples]` |
| `X_val` | `torch.Tensor` | `None` | Features de validacion (opcional) |
| `y_val` | `torch.Tensor` | `None` | Labels de validacion (opcional) |
| `fitness_type` | `str` | `"loss"` | Tipo de fitness: `"loss"` o `"accuracy"` |
| `callback` | `Callable` | `None` | Funcion callback por iteracion |
| `verbose` | `bool` | `True` | Imprimir progreso |

**Retorna:** `OptimizationResult` con los resultados de la optimizacion

**Ejemplo:**
```python
from ann.models import QPSOCompatibleANN
from ann.optimizers import QPSONNOptimizer
import torch

# Crear modelo
model = QPSOCompatibleANN(4, 3, [16, 8])

# Crear optimizador
optimizer = QPSONNOptimizer(model)

# Datos de ejemplo
X = torch.randn(100, 4)
y = torch.randint(0, 3, (100,))

# Entrenar
result = optimizer.fit(X, y, verbose=True)

print(f"Mejor loss: {result.best_value:.6f}")
print(f"Iteraciones: {result.iterations}")
```

---

### predict()

Realiza predicciones con el modelo entrenado.

```python
def predict(self, X: torch.Tensor) -> torch.Tensor
```

**Parametros:**
- `X`: Features de entrada `[n_samples, n_features]`

**Retorna:** Tensor con predicciones (clases) `[n_samples]`

**Ejemplo:**
```python
# Despues de entrenar
predictions = optimizer.predict(X_test)
accuracy = (predictions == y_test).float().mean()
print(f"Accuracy: {accuracy:.4f}")
```

---

### predict_proba()

Obtiene probabilidades de clase.

```python
def predict_proba(self, X: torch.Tensor) -> torch.Tensor
```

**Parametros:**
- `X`: Features de entrada `[n_samples, n_features]`

**Retorna:** Tensor con probabilidades `[n_samples, n_classes]`

**Ejemplo:**
```python
probabilities = optimizer.predict_proba(X_test)
# probabilities.shape = [n_samples, 3]

# Obtener la clase mas probable
predictions = probabilities.argmax(dim=1)
```

---

### evaluate()

Evalua el modelo en un conjunto de datos.

```python
def evaluate(
    self,
    X: torch.Tensor,
    y: torch.Tensor
) -> Dict[str, float]
```

**Parametros:**
- `X`: Features
- `y`: Labels verdaderos

**Retorna:** Diccionario con `{'loss': float, 'accuracy': float}`

**Ejemplo:**
```python
metrics = optimizer.evaluate(X_test, y_test)
print(f"Loss: {metrics['loss']:.6f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

---

### get_history()

Retorna el historial de entrenamiento.

```python
def get_history(self) -> Dict[str, List]
```

**Retorna:**
```python
{
    'train_loss': [0.9, 0.8, 0.7, ...],
    'train_acc': [0.5, 0.6, 0.7, ...],
    'val_loss': [0.85, 0.75, 0.65, ...],
    'val_acc': [0.55, 0.65, 0.75, ...]
}
```

**Ejemplo:**
```python
history = optimizer.get_history()

import matplotlib.pyplot as plt
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.legend()
plt.show()
```

---

### Propiedad: best_params

Retorna los mejores parametros encontrados.

```python
@property
def best_params(self) -> Optional[torch.Tensor]
```

**Ejemplo:**
```python
# Despues de entrenar
best_weights = optimizer.best_params
print(f"Shape: {best_weights.shape}")

# Guardar para uso posterior
torch.save(best_weights, 'best_weights.pt')
```

---

## QDPSONNOptimizer

### Descripcion

Wrapper conveniente que usa QDPSO por defecto. Identico a `QPSONNOptimizer` pero con `use_qdpso=True`.

### Importacion

```python
from ann.optimizers import QDPSONNOptimizer
```

### Uso

```python
from ann.optimizers import QDPSONNOptimizer
from ann.optimizers.qpso_nn import NNOptimizationConfig

config = NNOptimizationConfig(
    n_particles=50,
    max_iters=100,
    g=0.96  # Parametro especifico de QDPSO
)

optimizer = QDPSONNOptimizer(model, config=config)
result = optimizer.fit(X_train, y_train)
```

---

## Tipos de Fitness

### fitness_type="loss" (Default)

Minimiza la funcion de perdida (CrossEntropyLoss por defecto).

```python
result = optimizer.fit(X, y, fitness_type="loss")
```

**Funcionamiento interno:**
```python
def fitness(particles):
    for particle in particles:
        model.set_flat_params(particle)
        outputs = model(X_train)
        loss = loss_fn(outputs, y_train)  # Minimizar
```

### fitness_type="accuracy"

Maximiza la precision (internamente minimiza -accuracy).

```python
result = optimizer.fit(X, y, fitness_type="accuracy")
```

**Cuando usar accuracy:**
- Datasets desbalanceados donde loss puede ser enganoso
- Cuando la metrica final es accuracy

**Nota:** `fitness_type="accuracy"` puede ser mas lento ya que requiere calcular predicciones.

---

## Callbacks Personalizados

### Estructura del Callback

```python
def my_callback(optimizer_state):
    # optimizer_state tiene:
    # - iters: iteracion actual
    # - gbest: mejor posicion global
    # - gbest_value: mejor valor de fitness
    pass
```

### Ejemplo: Logging Personalizado

```python
def logging_callback(opt):
    if opt.iters % 10 == 0:
        print(f"Iter {opt.iters}: best={opt.gbest_value:.6f}")

result = optimizer.fit(X, y, callback=logging_callback)
```

### Ejemplo: Early Stopping Personalizado

```python
class EarlyStoppingCallback:
    def __init__(self, target_loss=0.1):
        self.target_loss = target_loss
        self.stopped = False

    def __call__(self, opt):
        if opt.gbest_value < self.target_loss:
            self.stopped = True
            print(f"Target reached at iter {opt.iters}!")

callback = EarlyStoppingCallback(target_loss=0.5)
result = optimizer.fit(X, y, callback=callback)
```

### Ejemplo: Guardar Checkpoints

```python
import torch

def checkpoint_callback(opt):
    if opt.iters % 50 == 0:
        torch.save({
            'iter': opt.iters,
            'best_value': opt.gbest_value,
            'best_params': opt.gbest.clone()
        }, f'checkpoint_iter{opt.iters}.pt')

result = optimizer.fit(X, y, callback=checkpoint_callback)
```

---

## Ejemplos Completos

### Ejemplo 1: Entrenamiento Basico

```python
import torch
from ann.models import QPSOCompatibleANN
from ann.optimizers import QPSONNOptimizer
from ann.utils import load_dataset

# Cargar datos
X_train, X_test, y_train, y_test = load_dataset('iris')

# Crear modelo
model = QPSOCompatibleANN(
    input_dim=4,
    output_dim=3,
    hidden_layers=[16, 8]
)

# Crear optimizador
optimizer = QPSONNOptimizer(model)

# Convertir a tensores
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Entrenar
result = optimizer.fit(X_train_t, y_train_t, verbose=True)

# Evaluar
test_metrics = optimizer.evaluate(X_test_t, y_test_t)
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
```

### Ejemplo 2: Configuracion Avanzada

```python
from ann.optimizers.qpso_nn import NNOptimizationConfig

config = NNOptimizationConfig(
    n_particles=100,           # Mas particulas
    max_iters=300,             # Mas iteraciones
    alpha=(1.0, 0.3),          # Alpha decay mas agresivo
    weight_bound=2.0,          # Pesos mas grandes permitidos
    boundary_strategy="reflect",
    tol=1e-15,
    patience=100,
    seed=42
)

optimizer = QPSONNOptimizer(model, config=config)
result = optimizer.fit(X_train, y_train, X_val, y_val)
```

### Ejemplo 3: QDPSO vs QPSO

```python
from ann.optimizers import QPSONNOptimizer, QDPSONNOptimizer
from ann.optimizers.qpso_nn import NNOptimizationConfig
import time

# Configuracion comun
config = NNOptimizationConfig(
    n_particles=50,
    max_iters=100,
    seed=42
)

results = {}

# QPSO
model_qpso = QPSOCompatibleANN(4, 3, [16, 8])
opt_qpso = QPSONNOptimizer(model_qpso, config=config)
start = time.time()
result_qpso = opt_qpso.fit(X_train, y_train, verbose=False)
results['QPSO'] = {
    'accuracy': opt_qpso.evaluate(X_test, y_test)['accuracy'],
    'time': time.time() - start
}

# QDPSO
model_qdpso = QPSOCompatibleANN(4, 3, [16, 8])
opt_qdpso = QDPSONNOptimizer(model_qdpso, config=config)
start = time.time()
result_qdpso = opt_qdpso.fit(X_train, y_train, verbose=False)
results['QDPSO'] = {
    'accuracy': opt_qdpso.evaluate(X_test, y_test)['accuracy'],
    'time': time.time() - start
}

# Comparar
for algo, res in results.items():
    print(f"{algo}: acc={res['accuracy']:.4f}, time={res['time']:.2f}s")
```

### Ejemplo 4: Loss Personalizado

```python
import torch.nn as nn

# Usar Focal Loss para datos desbalanceados
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Usar con el optimizador
focal_loss = FocalLoss(gamma=2.0)
optimizer = QPSONNOptimizer(model, loss_fn=focal_loss)
result = optimizer.fit(X_train, y_train)
```

---

## Integracion con qpso_tensor_optimized.py

El optimizador usa internamente `QPSOTensorOptimized` o `QDPSOTensorOptimized` del modulo `tensor_qpso/qpso_tensor_optimized.py`, heredando todas sus optimizaciones:

| Optimizacion | Descripcion |
|--------------|-------------|
| Vectorizacion GPU | Operaciones en batch con tensores |
| Memory pooling | Reutilizacion de tensores |
| torch.no_grad() | Evaluacion sin gradientes |
| Random batch generation | Generacion eficiente de aleatorios |
| Boundary strategies | 5 estrategias de limites |
| Early stopping | Convergencia temprana con patience |
| History tracking | Registro de metricas |

---

## Mejoras vs Implementacion Original

| Aspecto | Original (`QDPSOHybridBackwardOptimizer`) | Nueva (`QPSONNOptimizer`) |
|---------|-------------------------------------------|--------------------------|
| Dependencias | Multiples imports personalizados | Solo `qpso_tensor_optimized` |
| Configuracion | Parametros en constructor | `NNOptimizationConfig` centralizado |
| Backward modes | forward, weighted, layerwise, blocks | Fitness function pura |
| Flexibilidad | Acoplado a arquitectura especifica | Compatible con cualquier modelo |
| Metricas | Externas | Integradas |
| Callbacks | No soportados | Completamente soportados |
| Reproducibilidad | Semilla manual | Semilla en config |

---

## Consideraciones de Rendimiento

1. **Numero de particulas**: Mas particulas = mejor exploracion pero mas lento
2. **Evaluacion de fitness**: El cuello de botella es evaluar la red N veces por iteracion
3. **GPU**: Usar GPU para el modelo acelera significativamente
4. **Batch size**: Todo el dataset se evalua en cada fitness (considerar muestreo para datasets grandes)

### Recomendaciones

```python
# Dataset pequeno (< 1000)
config = NNOptimizationConfig(n_particles=30, max_iters=100)

# Dataset mediano (1000-10000)
config = NNOptimizationConfig(n_particles=50, max_iters=200)

# Dataset grande (> 10000)
config = NNOptimizationConfig(n_particles=100, max_iters=500)
# Considerar mini-batch fitness
```

---

## Related Documents

- [📚 Index](index.md) - Module overview
- [🧠 Models](models.md) - QPSO-compatible neural networks
- [🏋️ Trainers](trainers.md) - High-level trainer with cross-validation
- [📖 Examples](examples.md) - Complete usage examples

---

<div align="center">

**[⬆️ Back to Top](#modulo-optimizers---documentacion)** | **[⬅️ Models](models.md)** | **[📚 Index](index.md)** | **[Next: Trainers ➡️](trainers.md)**

</div>
