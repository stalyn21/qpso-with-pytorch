# Modulo ANN - Entrenamiento de Redes Neuronales con QPSO

[🏠 README Principal](../../README_ES.md) | [🔧 Algoritmos QPSO](../../QPSO-PyTorch/docs/index_es.md) | **Modulo ANN** | [🇬🇧 English](index.md)

---

> **Version**: 2.0.0
> **Nota**: Este modulo es parte del proyecto [PyTorch QPSO Suite](../../README_ES.md).
> Para una vision general del repositorio completo, consulta el [README principal](../../README_ES.md).

---

## Descripcion General

Este paquete proporciona una implementacion modular y optimizada para entrenar redes neuronales artificiales utilizando **QPSO (Quantum Particle Swarm Optimization)** como algoritmo de optimizacion, en lugar del tradicional backpropagation con gradiente descendente.

### Concepto Fundamental

En el enfoque tradicional, las redes neuronales se entrenan calculando gradientes y actualizando pesos mediante algoritmos como SGD o Adam. En contraste, QPSO trata los pesos de la red como **posiciones de particulas** en un espacio de busqueda multidimensional, donde cada particula representa una configuracion completa de pesos.

```
Enfoque Tradicional:          Enfoque QPSO:

  Forward Pass                  Particula 1: [w1, w2, ..., wn]
       |                        Particula 2: [w1, w2, ..., wn]
  Calcular Loss                 ...
       |                        Particula N: [w1, w2, ..., wn]
  Backward Pass                        |
       |                        Evaluar fitness (loss)
  Actualizar Pesos                     |
                                Actualizar posiciones (QPSO)
                                       |
                                Mejor particula = Mejores pesos
```

---

## Tabla de Contenidos

### Documentacion de Modulos

| Documento | Descripcion |
|-----------|-------------|
| [models.md](models.md) | Modelos de redes neuronales compatibles con QPSO |
| [optimizers.md](optimizers.md) | Optimizadores QPSO/QDPSO para redes neuronales |
| [trainers.md](trainers.md) | Trainer de alto nivel con validacion cruzada |
| [utils.md](utils.md) | Utilidades para datos y metricas |
| [examples.md](examples.md) | Ejemplos completos y casos de uso |

### Documentacion de Scripts Ejecutables

| Documento | Descripcion |
|-----------|-------------|
| [main_qpso.md](main_qpso.md) | Benchmark de entrenamiento con QPSO |
| [main_qdpso.md](main_qdpso.md) | Benchmark de entrenamiento con QDPSO |
| [main_mcw.md](main_mcw.md) | Benchmark MCW de clasificacion de imagenes (QPSO vs QDPSO) |
| [main_training_type.md](main_training_type.md) | Benchmark de estrategias de entrenamiento (Forward, Weighted, Layerwise) |
| [main_hyperparameter_search.md](main_hyperparameter_search.md) | Busqueda automatica de hiperparametros con Optuna |
| [usage_cases.md](usage_cases.md) | 8 ejemplos de uso |

---

## Arquitectura del Paquete

```
ann/
├── __init__.py                 # Exports principales del paquete
├── main_qpso.py                # Benchmark con QPSO (Iris, Wine, Breast Cancer)
├── main_qdpso.py               # Benchmark con QDPSO (Iris, Wine, Breast Cancer)
├── main_mcw.py                 # Benchmark MCW con QPSO y QDPSO (imagenes clima)
├── main_training_type.py       # Benchmark de estrategias de entrenamiento
├── main_hyperparameter_search.py  # Busqueda de hiperparametros con Optuna
├── start_hyperparameter_search.py # Script de configuracion HPO
├── usage_cases.py              # 8 ejemplos de uso
│
├── tensor_qpso/                # Modulo de optimizacion QPSO (17 mejoras)
│   ├── __init__.py             # Exports del modulo
│   └── qpso_tensor_optimized.py  # Implementacion optimizada
│
├── data/                       # Modulos de carga de datos
│   ├── __init__.py             # Exports del modulo
│   └── mcw.py                  # MCWDataset, load_mcw (Multi-Class Weather)
│
├── models/                     # Modelos de redes neuronales
│   ├── __init__.py
│   └── ann.py                  # QPSOCompatibleANN
│
├── optimizers/                 # Optimizadores basados en QPSO
│   ├── __init__.py
│   ├── qpso_nn.py              # QPSONNOptimizer, QDPSONNOptimizer
│   └── training_strategies.py  # Estrategias: Forward, Weighted, Layerwise
│
├── trainers/                   # Logica de entrenamiento
│   ├── __init__.py
│   └── trainer.py              # Trainer, TrainingConfig, TrainingResult
│
├── utils/                      # Utilidades
│   ├── __init__.py
│   ├── data.py                 # Funciones de carga y preprocesamiento
│   └── metrics.py              # Metricas de evaluacion y visualizacion
│
├── results/                    # Resultados de experimentos
│   └── hyperparameter_search/  # Resultados de HPO
│
└── docs/                       # Documentacion
    ├── index.md                # Indice en ingles
    ├── index_es.md             # Este archivo (espanol)
    └── *.md                    # Documentacion de modulos
```

---

## Instalacion y Requisitos

### Requisitos

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- scikit-learn >= 0.24.0
- OpenCV >= 4.0.0 (para dataset MCW)
- mahotas >= 1.4.0 (para features de imagenes)
- matplotlib (opcional, para graficas)

### Instalacion

```bash
# Activar entorno conda
conda activate pytorch_qpso

# Verificar instalacion
python -c "from ann import QPSOCompatibleANN, Trainer; print('OK')"
```

---

## Inicio Rapido

### Ejemplo Minimo

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset

# 1. Cargar datos
X_train, X_test, y_train, y_test = load_dataset('iris')

# 2. Configurar
config = TrainingConfig(
    hidden_layers=[16, 8],
    n_particles=30,
    max_iters=100
)

# 3. Crear trainer
trainer = Trainer(
    input_dim=X_train.shape[1],
    output_dim=3,
    config=config
)

# 4. Entrenar
result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

# 5. Resultados
print(f"Test Accuracy: {result.test_accuracy:.4f}")
```

### Uso con Optimizador Directo

```python
import torch
from ann.models import QPSOCompatibleANN
from ann.optimizers import QPSONNOptimizer

# Crear modelo
model = QPSOCompatibleANN(
    input_dim=4,
    output_dim=3,
    hidden_layers=[16, 8]
)

# Crear optimizador
optimizer = QPSONNOptimizer(model)

# Entrenar
X = torch.randn(100, 4)
y = torch.randint(0, 3, (100,))
result = optimizer.fit(X, y)
```

---

## Algoritmos Soportados

### QPSO (Quantum Particle Swarm Optimization)

Algoritmo original de Sun et al. (2004). Usa el concepto de **mean best (mbest)**.

**Ecuacion de actualizacion:**
```
mbest = (1/N) * sum(pbest_i)
c = phi * pbest + (1-phi) * gbest
L = alpha * |mbest - x|
x_new = c +/- L * ln(1/u)
```

**Parametros:**
- `alpha`: Factor de contraccion-expansion (tipico: 0.75 o (1.0, 0.5) para decay)

### QDPSO (Quantum Delta PSO)

Variante que usa delta en lugar de mbest.

**Ecuacion de actualizacion:**
```
c = (u1*pbest + u2*gbest) / (u1+u2)
L = (1/g) * |x - c|
x_new = c +/- L * ln(1/u)
```

**Parametros:**
- `g`: Factor de control (tipico: 0.96)

---

## Estrategias de Entrenamiento

El modulo soporta tres estrategias de entrenamiento:

| Estrategia | Descripcion | Velocidad | Precision | Mejor Para |
|------------|-------------|-----------|-----------|------------|
| **Forward** | Optimiza todos los pesos simultaneamente | Rapida | Buena | Redes pequenas |
| **Weighted** | Fitness ponderado por capa con decay | Media | Buena | Redes medianas |
| **Layerwise** | Entrenamiento secuencial capa por capa | Lenta | Mejor | Redes profundas |

### Estrategia Forward
Enfoque estandar - todos los pesos de la red se optimizan simultaneamente usando QPSO.

### Estrategia Weighted
Aplica diferentes pesos a la contribucion de cada capa en la funcion de fitness, priorizando las capas de salida.

### Estrategia Layerwise
Entrena las capas secuencialmente desde la salida hacia la entrada, con fine-tuning opcional de todas las capas al final.

---

## Configuracion Recomendada

### Para Datasets Pequenos (< 1000 muestras)

```python
config = TrainingConfig(
    hidden_layers=[32, 16],
    n_particles=30,
    max_iters=100,
    alpha=(1.0, 0.5),
    patience=30
)
```

### Para Datasets Medianos (1000-10000 muestras)

```python
config = TrainingConfig(
    hidden_layers=[64, 32, 16],
    n_particles=50,
    max_iters=200,
    alpha=(1.0, 0.5),
    patience=50
)
```

### Para Datasets Grandes (> 10000 muestras)

```python
config = TrainingConfig(
    hidden_layers=[128, 64, 32],
    n_particles=100,
    max_iters=500,
    alpha=(1.0, 0.5),
    patience=100,
    use_qdpso=True  # QDPSO puede ser mas estable
)
```

---

## Scripts Ejecutables

### Descripcion de Scripts

| Script | Proposito | Dataset | Uso |
|--------|-----------|---------|-----|
| `main_qpso.py` | Benchmark QPSO | Iris, Wine, Breast Cancer | `python ann/main_qpso.py` |
| `main_qdpso.py` | Benchmark QDPSO | Iris, Wine, Breast Cancer | `python ann/main_qdpso.py` |
| `main_mcw.py` | Comparacion QPSO vs QDPSO | MCW (imagenes clima) | `python ann/main_mcw.py` |
| `main_training_type.py` | Benchmark estrategias | Iris, Wine, Breast Cancer | `python ann/main_training_type.py` |
| `main_hyperparameter_search.py` | HPO automatico (Optuna) | MCW | `python ann/main_hyperparameter_search.py` |
| `usage_cases.py` | Ejemplos educativos | Varios | `python ann/usage_cases.py` |

### Comparativa QPSO vs QDPSO

| Aspecto | QPSO | QDPSO |
|---------|------|-------|
| **Optimizador** | `QPSONNOptimizer` | `QDPSONNOptimizer` |
| **Parametro clave** | `alpha: (1.0, 0.5)` | `g: 0.96` |
| **Algoritmo base** | QPSO con mbest | QDPSO con factor g |
| **Ecuacion de L** | `L = alpha * |mbest - x|` | `L = (1/g) * |x - c|` |
| **Adaptabilidad** | Alpha con decay lineal | Factor g constante |
| **Complejidad** | Mayor (calcula mbest) | Menor |

---

## Referencias

1. Sun, J., Feng, B., & Xu, W. (2004). *Particle swarm optimization with particles having quantum behavior*. Congress on Evolutionary Computation.

2. Sun, J., Xu, W., & Feng, B. (2004). *A global search strategy of quantum-behaved particle swarm optimization*. IEEE Conference on Cybernetics and Intelligent Systems.

3. Sun, J., Fang, W., Wu, X., Palade, V., & Xu, W. (2012). *Quantum-behaved particle swarm optimization: Analysis of individual particle behavior and parameter selection*. Evolutionary Computation.

---

## Licencia

Este proyecto esta bajo la licencia MIT.

---

## Siguientes Pasos

Continua con la documentacion detallada de cada modulo:

### Documentacion de Modulos
- [models.md](models.md) - Modelos de redes neuronales
- [optimizers.md](optimizers.md) - Optimizadores QPSO
- [trainers.md](trainers.md) - Trainer de alto nivel
- [utils.md](utils.md) - Utilidades
- [examples.md](examples.md) - Ejemplos completos

### Documentacion de Scripts
- [main_qpso.md](main_qpso.md) - Benchmark QPSO
- [main_qdpso.md](main_qdpso.md) - Benchmark QDPSO
- [main_mcw.md](main_mcw.md) - Benchmark MCW (QPSO vs QDPSO)
- [main_training_type.md](main_training_type.md) - Estrategias de entrenamiento (Forward, Weighted, Layerwise)
- [main_hyperparameter_search.md](main_hyperparameter_search.md) - Busqueda de hiperparametros con Optuna
- [usage_cases.md](usage_cases.md) - Ejemplos de uso

---

<div align="center">

**[⬆️ Volver Arriba](#modulo-ann---entrenamiento-de-redes-neuronales-con-qpso)** | **[🏠 README Principal](../../README_ES.md)** | **[🔧 Algoritmos QPSO](../../QPSO-PyTorch/docs/index_es.md)**

</div>
