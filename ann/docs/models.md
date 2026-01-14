# Modulo Models - Documentacion

[🏠 README](../../README.md) | [📚 Index](index.md) | **Models** | [Next: Optimizers ➡️](optimizers.md)

---

## Descripcion General

El modulo `models` contiene las implementaciones de redes neuronales compatibles con optimizacion QPSO. La caracteristica principal de estos modelos es que permiten representar todos sus pesos como un **tensor plano unidimensional**, que el algoritmo QPSO utiliza como posiciones de particulas.

**Ubicacion:** `ann/models/`

**Archivo principal:** `ann.py`

---

## Clases Disponibles

| Clase | Descripcion |
|-------|-------------|
| `QPSOCompatibleANN` | Red neuronal feedforward compatible con QPSO |
| `ModelConfig` | Dataclass para configuracion del modelo |

## Funciones Disponibles

| Funcion | Descripcion |
|---------|-------------|
| `create_scaled_architecture` | Genera arquitectura escalada con patron [3:2:1] |

---

## QPSOCompatibleANN

### Descripcion

`QPSOCompatibleANN` es una red neuronal artificial feedforward (perceptron multicapa) disenada especificamente para ser optimizada con QPSO. A diferencia de los modelos tradicionales de PyTorch, esta clase proporciona metodos para:

- Obtener todos los pesos como un vector plano (`get_flat_params`)
- Establecer todos los pesos desde un vector plano (`set_flat_params`)
- Generar limites para el espacio de busqueda (`get_param_bounds`)

### Importacion

```python
from ann.models import QPSOCompatibleANN
# o
from ann.models.ann import QPSOCompatibleANN
```

### Constructor

```python
QPSOCompatibleANN(
    input_dim: int,
    output_dim: int,
    hidden_layers: List[int],
    activation: str = "relu",
    output_activation: Optional[str] = "softmax",
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    device: str = "auto"
)
```

### Parametros

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `input_dim` | `int` | - | Dimension de entrada (numero de features) |
| `output_dim` | `int` | - | Dimension de salida (numero de clases) |
| `hidden_layers` | `List[int]` | - | Lista con neuronas por capa oculta |
| `activation` | `str` | `"relu"` | Funcion de activacion para capas ocultas |
| `output_activation` | `str` | `"softmax"` | Activacion de la capa de salida |
| `dropout` | `float` | `0.0` | Probabilidad de dropout (0.0 = sin dropout) |
| `use_batch_norm` | `bool` | `False` | Usar batch normalization |
| `device` | `str` | `"auto"` | Dispositivo: `"cpu"`, `"cuda"`, o `"auto"` |

### Activaciones Soportadas

| Nombre | Clase PyTorch | Descripcion |
|--------|---------------|-------------|
| `"relu"` | `nn.ReLU` | Rectified Linear Unit |
| `"tanh"` | `nn.Tanh` | Tangente hiperbolica |
| `"sigmoid"` | `nn.Sigmoid` | Funcion sigmoide |
| `"leaky_relu"` | `nn.LeakyReLU` | Leaky ReLU |
| `"elu"` | `nn.ELU` | Exponential Linear Unit |
| `"gelu"` | `nn.GELU` | Gaussian Error Linear Unit |

### Activaciones de Salida

| Nombre | Descripcion | Uso |
|--------|-------------|-----|
| `"softmax"` | Probabilidades normalizadas | Clasificacion multiclase |
| `"sigmoid"` | Valores entre 0 y 1 | Clasificacion binaria/multilabel |
| `"log_softmax"` | Log-probabilidades | Con NLLLoss |
| `None` | Sin activacion | Regresion |

---

## Metodos Principales

### forward(x)

Realiza el forward pass de la red.

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Parametros:**
- `x`: Tensor de entrada con shape `[batch_size, input_dim]`

**Retorna:**
- Tensor de salida con shape `[batch_size, output_dim]`

**Ejemplo:**
```python
model = QPSOCompatibleANN(4, 3, [16, 8])
x = torch.randn(32, 4)  # batch de 32 muestras
y = model(x)  # shape: [32, 3]
```

---

### get_flat_params()

Obtiene todos los parametros del modelo como un tensor plano 1D.

```python
def get_flat_params(self) -> torch.Tensor
```

**Retorna:**
- Tensor 1D con todos los pesos y biases concatenados

**Ejemplo:**
```python
model = QPSOCompatibleANN(4, 3, [16, 8])
params = model.get_flat_params()
print(params.shape)  # torch.Size([243])
```

**Orden de concatenacion:**
```
[weights_layer1, bias_layer1, weights_layer2, bias_layer2, ...]
```

---

### set_flat_params(flat_params)

Establece los parametros del modelo desde un tensor plano.

```python
def set_flat_params(self, flat_params: torch.Tensor) -> None
```

**Parametros:**
- `flat_params`: Tensor 1D con todos los parametros

**Raises:**
- `ValueError`: Si el tamano no coincide con `num_params`

**Ejemplo:**
```python
model = QPSOCompatibleANN(4, 3, [16, 8])

# Obtener parametros actuales
params = model.get_flat_params()

# Modificar (por ejemplo, agregar ruido)
new_params = params + torch.randn_like(params) * 0.01

# Establecer nuevos parametros
model.set_flat_params(new_params)
```

---

### get_param_bounds(bound)

Genera limites para cada parametro (usado por QPSO como bounds del espacio de busqueda).

```python
def get_param_bounds(
    self,
    bound: Union[float, Tuple[float, float]] = 1.0
) -> List[Tuple[float, float]]
```

**Parametros:**
- `bound`: Limite simetrico (float) o tupla (min, max)

**Retorna:**
- Lista de tuplas `(min, max)` para cada parametro

**Ejemplo:**
```python
model = QPSOCompatibleANN(4, 3, [16, 8])

# Limites simetricos
bounds = model.get_param_bounds(1.0)
# [(-1.0, 1.0), (-1.0, 1.0), ...]

# Limites asimetricos
bounds = model.get_param_bounds((-0.5, 1.5))
# [(-0.5, 1.5), (-0.5, 1.5), ...]
```

---

### clone()

Crea una copia profunda del modelo con la misma arquitectura y pesos.

```python
def clone(self) -> QPSOCompatibleANN
```

**Retorna:**
- Nueva instancia del modelo

**Ejemplo:**
```python
model = QPSOCompatibleANN(4, 3, [16, 8])
model_copy = model.clone()

# Verificar que son independientes
model.set_flat_params(torch.zeros(model.num_params))
assert not torch.equal(model.get_flat_params(), model_copy.get_flat_params())
```

---

### reset_parameters()

Reinicializa los parametros de la red usando inicializacion Kaiming.

```python
def reset_parameters(self) -> None
```

**Ejemplo:**
```python
model = QPSOCompatibleANN(4, 3, [16, 8])
model.reset_parameters()  # Reinicializa todos los pesos
```

---

### get_architecture_info()

Retorna informacion detallada sobre la arquitectura del modelo.

```python
def get_architecture_info(self) -> dict
```

**Retorna:**
```python
{
    "input_dim": 4,
    "output_dim": 3,
    "hidden_layers": [16, 8],
    "activation": "relu",
    "output_activation": "softmax",
    "dropout": 0.0,
    "use_batch_norm": False,
    "num_params": 243,
    "device": "cuda"
}
```

---

## Propiedades

| Propiedad | Tipo | Descripcion |
|-----------|------|-------------|
| `num_params` | `int` | Numero total de parametros |
| `device` | `torch.device` | Dispositivo actual del modelo |

---

## Funcion create_scaled_architecture

### Descripcion

Genera una arquitectura de capas ocultas escalada basada en un patron multiplicador. Por defecto usa el patron `[3:2:1]` que crea tres capas donde cada una tiene un tamano proporcional a la entrada.

### Firma

```python
def create_scaled_architecture(
    input_dim: int,
    scale_factor: float = 1.0,
    pattern: List[int] = [3, 2, 1]
) -> List[int]
```

### Parametros

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `input_dim` | `int` | - | Dimension de entrada |
| `scale_factor` | `float` | `1.0` | Factor de escala global |
| `pattern` | `List[int]` | `[3, 2, 1]` | Patron de multiplicadores |

### Ejemplo

```python
from ann.models import create_scaled_architecture

# Input de 10 features, escala 1.0
arch = create_scaled_architecture(10, 1.0)
# [30, 20, 10]  -> 10*3, 10*2, 10*1

# Input de 10 features, escala 1.5
arch = create_scaled_architecture(10, 1.5)
# [45, 30, 15]  -> 10*3*1.5, 10*2*1.5, 10*1*1.5

# Input de 10 features, escala 0.5
arch = create_scaled_architecture(10, 0.5)
# [15, 10, 5]   -> 10*3*0.5, 10*2*0.5, 10*1*0.5

# Patron personalizado [4:2:1]
arch = create_scaled_architecture(10, 1.0, [4, 2, 1])
# [40, 20, 10]
```

### Caso de Uso: Busqueda de Arquitectura

```python
from ann.models import QPSOCompatibleANN, create_scaled_architecture
from ann.trainers import Trainer, TrainingConfig

input_dim = 20
output_dim = 5

# Probar diferentes escalas
for scale in [0.5, 1.0, 1.5, 2.0]:
    hidden_layers = create_scaled_architecture(input_dim, scale)

    config = TrainingConfig(
        hidden_layers=hidden_layers,
        n_particles=30,
        max_iters=50
    )

    trainer = Trainer(input_dim, output_dim, config)
    result = trainer.fit(X_train, y_train)

    print(f"Scale {scale}: {hidden_layers} -> Acc: {result.val_accuracy:.4f}")
```

---

## Ejemplos de Uso

### Ejemplo 1: Creacion Basica

```python
from ann.models import QPSOCompatibleANN

# Modelo para clasificacion de Iris (4 features, 3 clases)
model = QPSOCompatibleANN(
    input_dim=4,
    output_dim=3,
    hidden_layers=[16, 8]
)

print(model)
# QPSOCompatibleANN(
#   architecture: 4 -> 16 -> 8 -> 3
#   activation: relu
#   params: 243
#   device: cuda
# )
```

### Ejemplo 2: Arquitectura Compleja

```python
model = QPSOCompatibleANN(
    input_dim=784,           # MNIST
    output_dim=10,
    hidden_layers=[256, 128, 64],
    activation='gelu',
    dropout=0.2,
    use_batch_norm=True
)

print(f"Parametros: {model.num_params:,}")  # ~240,000
```

### Ejemplo 3: Manipulacion de Pesos

```python
import torch
from ann.models import QPSOCompatibleANN

model = QPSOCompatibleANN(4, 3, [16, 8])

# Obtener pesos actuales
weights = model.get_flat_params()
print(f"Shape: {weights.shape}")
print(f"Mean: {weights.mean():.4f}")
print(f"Std: {weights.std():.4f}")

# Escalar todos los pesos
scaled_weights = weights * 0.5
model.set_flat_params(scaled_weights)

# Verificar
new_weights = model.get_flat_params()
print(f"New Std: {new_weights.std():.4f}")  # ~0.5x original
```

### Ejemplo 4: Forward Pass con Batch

```python
import torch
from ann.models import QPSOCompatibleANN

model = QPSOCompatibleANN(10, 5, [32, 16], device='cuda')

# Crear batch de datos
batch_size = 64
X = torch.randn(batch_size, 10, device='cuda')

# Forward pass
with torch.no_grad():
    probs = model(X)  # [64, 5]
    predictions = probs.argmax(dim=1)  # [64]

print(f"Predictions shape: {predictions.shape}")
```

### Ejemplo 5: Guardar y Cargar Pesos

```python
import torch
from ann.models import QPSOCompatibleANN

# Crear y entrenar modelo
model = QPSOCompatibleANN(4, 3, [16, 8])
# ... entrenar ...

# Guardar pesos
weights = model.get_flat_params()
torch.save({
    'weights': weights,
    'architecture': model.get_architecture_info()
}, 'model_weights.pt')

# Cargar en nuevo modelo
checkpoint = torch.load('model_weights.pt')
new_model = QPSOCompatibleANN(
    input_dim=checkpoint['architecture']['input_dim'],
    output_dim=checkpoint['architecture']['output_dim'],
    hidden_layers=checkpoint['architecture']['hidden_layers']
)
new_model.set_flat_params(checkpoint['weights'])
```

---

## Detalles de Implementacion

### Calculo del Numero de Parametros

Para una red con arquitectura `[input] -> [h1] -> [h2] -> [output]`:

```
params = (input + 1) * h1     # Capa 1: pesos + bias
       + (h1 + 1) * h2        # Capa 2: pesos + bias
       + (h2 + 1) * output    # Capa salida: pesos + bias
```

**Ejemplo:** `4 -> 16 -> 8 -> 3`
```
params = (4 + 1) * 16 + (16 + 1) * 8 + (8 + 1) * 3
       = 80 + 136 + 27
       = 243
```

### Indices de Parametros Pre-computados

Para optimizar `set_flat_params`, se pre-computan los indices de cada parametro:

```python
self._param_indices = [
    (0, 80),      # Capa 1
    (80, 216),    # Capa 2
    (216, 243)    # Capa salida
]
```

Esto evita recalcular offsets en cada llamada.

---

## Mejoras vs Implementacion Original

| Caracteristica | Original (`ExtendedModel`) | Nueva (`QPSOCompatibleANN`) |
|----------------|---------------------------|----------------------------|
| Dependencias | Acoplado a optimizador | Independiente |
| Configuracion | Parametros fijos | Totalmente configurable |
| Activaciones | Solo ReLU | 6 opciones |
| Dropout | No soportado | Configurable |
| Batch Norm | No soportado | Configurable |
| Device | Manual | Automatico |
| Clonacion | No disponible | `clone()` |
| Serializacion | Compleja | `get_architecture_info()` |

---

## Consideraciones de Rendimiento

1. **Dispositivo**: El modelo se mueve automaticamente a GPU si esta disponible
2. **Indices pre-computados**: `set_flat_params` es O(n) con bajo overhead
3. **Sin gradientes**: Usar `torch.no_grad()` en evaluacion
4. **Batch processing**: Siempre procesar en batches para eficiencia

---

## Related Documents

- [📚 Index](index.md) - Module overview
- [⚙️ Optimizers](optimizers.md) - Use models with QPSO optimizers
- [🏋️ Trainers](trainers.md) - High-level training interface
- [📖 Examples](examples.md) - Complete examples

---

<div align="center">

**[⬆️ Back to Top](#modulo-models---documentacion)** | **[📚 Index](index.md)** | **[Next: Optimizers ➡️](optimizers.md)**

</div>
