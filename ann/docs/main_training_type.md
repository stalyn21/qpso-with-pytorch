# Documentacion: main_training_type.py

[🏠 README](../../README.md) | [📚 Index](index.md) | [⬅️ MCW](main_mcw.md) | **Training Strategies** | [Next: HPO ➡️](main_hyperparameter_search.md)

---

## Descripcion General

Script de benchmark para comparar diferentes **estrategias de entrenamiento** usando optimizadores QPSO y QDPSO. Permite evaluar cual estrategia funciona mejor para diferentes datasets.

### Estrategias Disponibles

| Estrategia | Descripcion | Caracteristica Principal |
|------------|-------------|-------------------------|
| **Forward** | Entrenamiento estandar | Todos los pesos a la vez |
| **Weighted** | Forward con pesos por capa | Prioriza capas de salida |
| **Layerwise** | Capa por capa (output→input) | Entrena secuencialmente |

---

## Tabla de Contenidos

1. [Concepto de Estrategias](#concepto-de-estrategias)
2. [Estrategia Forward](#estrategia-forward)
3. [Estrategia Weighted](#estrategia-weighted)
4. [Estrategia Layerwise](#estrategia-layerwise)
5. [Configuracion](#configuracion)
6. [Uso del Script](#uso-del-script)
7. [Uso Programatico](#uso-programatico)
8. [Resultados Tipicos](#resultados-tipicos)
9. [Comparativa de Estrategias](#comparativa-de-estrategias)

---

## Concepto de Estrategias

### Por que diferentes estrategias?

En el entrenamiento tradicional con backpropagation, los gradientes fluyen de la salida hacia la entrada, naturalmente dando mas influencia a las capas cercanas a la salida. Con QPSO, optimizamos todos los pesos simultaneamente, lo que puede dificultar el entrenamiento en redes profundas.

Las estrategias alternativas intentan replicar algunos beneficios del entrenamiento por capas:

```
Forward (Estandar):
┌─────────────────────────────────────────────────────┐
│ Particula = [w1, w2, w3, ..., wN] (todos los pesos) │
│                     ↓                               │
│              Evaluar fitness                        │
│                     ↓                               │
│              Actualizar QPSO                        │
└─────────────────────────────────────────────────────┘

Weighted:
┌─────────────────────────────────────────────────────┐
│ Particula = [w1, w2, w3, ..., wN]                   │
│                     ↓                               │
│ Fitness = Loss + α × Regularizacion(capas)          │
│           ↑                                         │
│   Capas output: peso alto                           │
│   Capas input: peso bajo                            │
└─────────────────────────────────────────────────────┘

Layerwise:
┌─────────────────────────────────────────────────────┐
│ Fase 1: Optimizar capa de salida                    │
│ Fase 2: Congelar salida, optimizar capa anterior    │
│ Fase 3: Continuar hacia la entrada                  │
│ Fase 4: Fine-tuning de toda la red                  │
└─────────────────────────────────────────────────────┘
```

---

## Estrategia Forward

### Descripcion

La estrategia **Forward** es el metodo estandar usado en QPSO. Optimiza todos los parametros de la red simultaneamente.

### Funcionamiento

1. Cada particula representa **todos** los pesos de la red
2. El fitness es el loss de clasificacion
3. QPSO actualiza todas las particulas en cada iteracion

### Ecuacion de Fitness

```
fitness(particula) = CrossEntropyLoss(model(X_train), y_train)
```

### Ventajas

- Simple y directo
- Rapido (una sola fase de optimizacion)
- No requiere configuracion adicional

### Desventajas

- Puede tener dificultades en redes muy profundas
- Todas las capas compiten por atencion igualmente

### Codigo

```python
from ann.optimizers import create_training_strategy, StrategyConfig

config = StrategyConfig(n_particles=50, max_iters=100)
strategy = create_training_strategy(model, 'forward', config)
strategy.set_data(X_train, y_train, X_val, y_val)
result = strategy.train()
```

---

## Estrategia Weighted

### Descripcion

La estrategia **Weighted** usa un forward pass pero con pesos diferentes por capa. Las capas cercanas a la salida tienen mayor influencia en la funcion de fitness.

### Funcionamiento

1. Calcula pesos por capa: `peso = decay^(distancia_a_salida)`
2. Captura activaciones de cada capa durante forward pass
3. Aplica regularizacion ponderada por capa
4. Fitness = Loss + regularizacion

### Ecuacion de Fitness

```
fitness = main_loss + α × regularization

Donde:
  main_loss = CrossEntropyLoss(output, y_train)
  regularization = Σ (1 - peso_capa) × variance(activacion_capa)
```

### Pesos por Capa

Para una red con 3 capas y `layer_decay=0.7`:

| Capa | Distancia a Output | Peso |
|------|-------------------|------|
| Output (3) | 0 | 1.0 × 0.7^0 = 1.00 |
| Hidden2 (2) | 1 | 1.0 × 0.7^1 = 0.70 |
| Hidden1 (1) | 2 | 1.0 × 0.7^2 = 0.49 |

### Parametros

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `layer_decay` | 0.8 | Factor de decaimiento entre capas |
| `output_weight` | 1.0 | Peso de la capa de salida |
| `regularization` | 0.01 | Factor de regularizacion |

### Ventajas

- Prioriza el ajuste de capas de salida
- Estabiliza capas tempranas
- Una sola fase de optimizacion (rapido)

### Desventajas

- Requiere ajustar `layer_decay` y `regularization`
- Efectos pueden ser sutiles en redes pequeñas

### Codigo

```python
config = StrategyConfig(
    n_particles=50,
    max_iters=100,
    layer_decay=0.7,       # Capas output tienen mas peso
    regularization=0.01    # Penalizacion por varianza
)
strategy = create_training_strategy(model, 'weighted', config)
strategy.set_data(X_train, y_train, X_val, y_val)
result = strategy.train()
```

---

## Estrategia Layerwise

### Descripcion

La estrategia **Layerwise** entrena las capas secuencialmente desde la salida hacia la entrada. Cada capa se optimiza individualmente mientras las demas permanecen congeladas.

### Funcionamiento

1. **Fase 1**: Optimizar solo la capa de salida
2. **Fase 2**: Congelar salida, optimizar capa anterior
3. **Fase 3**: Continuar hacia la entrada
4. **Fase 4**: Fine-tuning de toda la red (opcional)

### Diagrama

```
Red: Input -> H1 -> H2 -> Output

Fase 1: [congelado] -> [congelado] -> [OPTIMIZAR]
        Input       -> H1          -> H2          -> Output

Fase 2: [congelado] -> [OPTIMIZAR] -> [congelado]
        Input       -> H1          -> H2          -> Output

Fase 3: [OPTIMIZAR] -> [congelado] -> [congelado]
        Input       -> H1          -> H2          -> Output

Fine-tune: [OPTIMIZAR TODOS]
```

### Parametros

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `iters_per_layer` | `max_iters/n_layers` | Iteraciones por capa |
| `fine_tune_iters` | 50 | Iteraciones de fine-tuning |
| `freeze_trained` | True | Congelar capas ya entrenadas |

### Ventajas

- Entrenamiento mas controlado
- Puede lograr mejores resultados en redes profundas
- Cada capa recibe atencion individual

### Desventajas

- Mas lento (multiples fases)
- Puede requerir mas iteraciones totales

### Codigo

```python
config = StrategyConfig(
    n_particles=50,
    max_iters=100,
    iters_per_layer=30,    # 30 iteraciones por capa
    fine_tune_iters=30     # 30 iteraciones finales
)
strategy = create_training_strategy(model, 'layerwise', config)
strategy.set_data(X_train, y_train, X_val, y_val)
result = strategy.train()
```

---

## Configuracion

### StrategyConfig

```python
@dataclass
class StrategyConfig:
    # Parametros comunes
    n_particles: int = 50           # Particulas del enjambre
    max_iters: int = 100            # Iteraciones maximas
    alpha: Tuple[float, float] = (1.0, 0.5)  # QPSO alpha
    g: float = 0.96                 # QDPSO g
    weight_bound: float = 1.0       # Limite de pesos
    patience: int = 50              # Early stopping
    tol: float = 1e-12              # Tolerancia
    seed: Optional[int] = None      # Semilla

    # Parametros para weighted
    layer_decay: float = 0.8        # Decaimiento entre capas
    output_weight: float = 1.0      # Peso de salida
    regularization: float = 0.01    # Factor regularizacion

    # Parametros para layerwise
    iters_per_layer: Optional[int] = None  # None = auto
    fine_tune_iters: int = 50       # Fine-tuning final
    freeze_trained: bool = True     # Congelar entrenadas
```

### Configuracion del Script

```python
# main_training_type.py

GENERAL_CONFIG = {
    'n_particles': 30,
    'max_iters': 100,
    'patience': 30,
    'alpha': (1.0, 0.5),
    'g': 0.96,
}

STRATEGY_CONFIGS = {
    'forward': {},
    'weighted': {
        'layer_decay': 0.7,
        'regularization': 0.01,
    },
    'layerwise': {
        'iters_per_layer': 30,
        'fine_tune_iters': 30,
    }
}

STRATEGIES = ['forward', 'weighted', 'layerwise']
OPTIMIZERS = ['QPSO', 'QDPSO']
DATASETS = ['iris', 'wine', 'breast_cancer']
```

---

## Uso del Script

### Ejecucion Basica

```bash
conda activate pytorch_qpso_gpu
python ann/main_training_type.py
```

### Salida del Script

```
======================================================================
 INFORMACION DEL SISTEMA
======================================================================
PyTorch version: 2.5.1
CUDA disponible: True
GPU: NVIDIA GeForce GTX 1050 Ti
Dispositivo: cuda

======================================================================
 BENCHMARK: IRIS
======================================================================

--- QPSO + FORWARD en iris ---
  Dataset: iris
  Arquitectura: 4 -> [12, 8] -> 3
  Train/Val/Test: 89/26/35
  Parametros: 143

  Resultados:
    Train Acc: 0.9663
    Test Acc:  0.9714
    F1 (macro): 0.9714
    Tiempo: 1.23s

--- QPSO + WEIGHTED en iris ---
...

======================================================================
 RESUMEN FINAL - COMPARATIVA DE ESTRATEGIAS
======================================================================

--------------------------------------------------------------------------------
 Dataset: IRIS
--------------------------------------------------------------------------------
Optimizer  Strategy     Train Acc    Test Acc     F1         Time
--------------------------------------------------------------------------------
QPSO       forward      0.9663       0.9714       0.9714     1.23s
QPSO       weighted     0.9551       0.9429       0.9429     1.45s
QPSO       layerwise    0.9775       0.9714       0.9714     2.34s
QDPSO      forward      0.9663       0.9714       0.9714     1.18s
...
```

---

## Uso Programatico

### Ejemplo Completo

```python
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ann.models import QPSOCompatibleANN
from ann.optimizers import (
    create_training_strategy,
    StrategyConfig,
    ForwardStrategy,
    WeightedStrategy,
    LayerwiseStrategy
)

# 1. Preparar datos
iris = datasets.load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 2. Crear modelo
model = QPSOCompatibleANN(
    input_dim=4,
    output_dim=3,
    hidden_layers=[12, 8],
    activation='tanh'
)

# 3. Crear estrategia
config = StrategyConfig(
    n_particles=30,
    max_iters=100,
    layer_decay=0.7  # Para weighted
)

strategy = create_training_strategy(
    model=model,
    strategy='weighted',  # 'forward', 'weighted', 'layerwise'
    config=config,
    use_qdpso=False  # True para QDPSO
)

# 4. Entrenar
strategy.set_data(X_train, y_train, X_test, y_test)
result = strategy.train(verbose=True)

# 5. Evaluar
print(f"Train Accuracy: {result.best_accuracy:.4f}")
print(f"Iterations: {result.iterations}")
print(f"Time: {result.elapsed_time:.2f}s")

# 6. Predecir
with torch.no_grad():
    predictions = model(X_test).argmax(dim=1)
    accuracy = (predictions == y_test).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")
```

### Usar Estrategias Directamente

```python
from ann.optimizers import WeightedStrategy, StrategyConfig

# Crear estrategia directamente
strategy = WeightedStrategy(
    model=model,
    config=StrategyConfig(layer_decay=0.7),
    use_qdpso=True
)

strategy.set_data(X_train, y_train, X_val, y_val)
result = strategy.train()
```

---

## Resultados Tipicos

### Dataset Iris

| Optimizer | Strategy | Train Acc | Test Acc | Time |
|-----------|----------|-----------|----------|------|
| QPSO | Forward | 0.9429 | 0.9556 | 0.38s |
| QPSO | Weighted | 0.9143 | 0.9556 | 0.45s |
| QPSO | Layerwise | 0.9619 | **1.0000** | 0.65s |
| QDPSO | Forward | 0.9524 | 0.9556 | 0.41s |
| QDPSO | Weighted | 0.9333 | 0.9333 | 0.48s |
| QDPSO | Layerwise | 0.9524 | 0.9778 | 0.72s |

### Observaciones

1. **Layerwise** suele lograr mejor accuracy pero es mas lento
2. **Forward** es el mas rapido y funciona bien en la mayoria de casos
3. **Weighted** puede ayudar en redes mas profundas
4. **QDPSO** tiende a ser mas estable que QPSO

---

## Comparativa de Estrategias

### Cuando usar cada estrategia

| Escenario | Estrategia Recomendada |
|-----------|----------------------|
| Red pequeña (< 500 params) | Forward |
| Red mediana (500-2000 params) | Forward o Weighted |
| Red profunda (> 3 capas ocultas) | Layerwise o Weighted |
| Tiempo limitado | Forward |
| Maxima precision | Layerwise |
| Entrenamiento estable | Weighted |

### Trade-offs

| Aspecto | Forward | Weighted | Layerwise |
|---------|---------|----------|-----------|
| Velocidad | Rapido | Medio | Lento |
| Precision | Buena | Buena | Mejor |
| Estabilidad | Media | Alta | Alta |
| Configuracion | Minima | Media | Media |
| Redes profundas | Regular | Bueno | Excelente |

---

## Referencias

1. **Layerwise Training**: Bengio et al. (2007). "Greedy Layer-Wise Training of Deep Networks"

2. **Weighted Training**: Conceptos inspirados en curriculum learning y attention mechanisms

3. **QPSO**: Sun et al. (2004). "Particle swarm optimization with particles having quantum behavior"
