# QPSO with PyTorch

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**[English Version](README.md)**

Suite de aplicaciones basadas en **Quantum Particle Swarm Optimization (QPSO)** implementadas en PyTorch con aceleracion GPU completa.

---

## Tabla de Contenidos

- [Descripcion](#descripcion)
- [Que es QPSO?](#que-es-qpso)
- [Arquitectura del Repositorio](#arquitectura-del-repositorio)
- [Resumen de Modulos](#resumen-de-modulos)
- [Instalacion](#instalacion)
- [Datasets](#datasets)
- [Inicio Rapido](#inicio-rapido)
- [Documentacion Detallada](#documentacion-detallada)
- [Referencia de API](#referencia-de-api)
- [Resultados de Ejemplo](#resultados-de-ejemplo)
- [Contribuir](#contribuir)
- [Citacion](#citacion)
- [Licencia](#licencia)
- [Registro de Cambios](#registro-de-cambios)

---

## Descripcion

Este repositorio contiene implementaciones optimizadas del algoritmo QPSO y sus variantes, junto con aplicaciones practicas para entrenamiento de redes neuronales y optimizacion de hiperparametros.

### Caracteristicas Principales

- **Optimizacion Acelerada por GPU**: Soporte completo de tensores PyTorch para aceleracion CUDA
- **Multiples Variantes de Algoritmo**: QPSO (original) y QDPSO (variante delta)
- **Entrenamiento de Redes Neuronales**: Entrena redes neuronales usando QPSO y QDPSO en lugar de las implementaciones clasicas basadas en gradientes
- **Multiples Estrategias de Entrenamiento**: Optimizacion Forward, Weighted y Layerwise
- **Optimizacion de Hiperparametros**: Soporte integrado de Optuna para HPO automatizado
- **Metricas Completas**: Validacion cruzada, reportes de clasificacion detallados, visualizaciones

---

## Que es QPSO?

**Quantum Particle Swarm Optimization (QPSO)** es un algoritmo de optimizacion metaheuristico inspirado en el comportamiento cuantico de las particulas. A diferencia del PSO clasico, QPSO no requiere parametros de velocidad, lo que simplifica su implementacion y mejora la convergencia global.

### Ecuacion Fundamental

```
x_new = c +/- L * ln(1/u)

donde:
  c = punto atractor (combinacion de pbest y gbest)
  L = longitud caracteristica
  u ~ U(0,1)
```

### Variantes Implementadas

| Algoritmo | Descripcion | Parametro Clave | Formula para L |
|-----------|-------------|-----------------|----------------|
| **QPSO** | Original (Sun et al., 2004). Usa posicion media mejor (mbest) | `alpha` (0.5-1.0) | L = alpha * \|mbest - x\| |
| **QDPSO** | Variante Delta. Usa distancia al punto atractor | `g` (~0.96) | L = (1/g) * \|x - c\| |

---

## Arquitectura del Repositorio

```
qdpso/
├── README.md                    # Version en ingles (principal)
├── README_ES.md                 # Este archivo (espanol)
│
├── QPSO-PyTorch/                # Implementaciones base del algoritmo QPSO
│   ├── tensor_qpso/             # Modulo principal
│   │   ├── __init__.py          # Exportaciones del modulo
│   │   ├── qpso.py              # Implementacion NumPy (referencia, basada en pypi qpso 0.0.1)
│   │   ├── qpso_tensor.py       # Implementacion basica con tensores PyTorch
│   │   └── qpso_tensor_optimized.py  # Implementacion optimizada (17 mejoras)
│   ├── docs/                    # Documentacion del algoritmo
│   │   ├── docs_qpso.md         # Docs implementacion NumPy
│   │   ├── docs_qpso_tensor.md  # Docs tensores basicos
│   │   └── docs_qpso_tensor_optimized.md  # Docs optimizados
│   ├── main_pypi.py             # Ejemplo QPSO original (basado en pypi)
│   ├── main_qpso.py             # Ejemplo wrapper QPSO
│   ├── main_qpso_tensor.py      # Ejemplo QPSO con tensores
│   ├── main_qpso_tensor_optimized.py  # Ejemplo tensores optimizados
│   └── get_device.py            # Utilidad deteccion de dispositivo
│
└── ann/                         # Entrenamiento de Redes Neuronales con QPSO
    ├── __init__.py              # Exportaciones del paquete
    ├── tensor_qpso/             # Modulo QPSO optimizado local
    │   ├── __init__.py
    │   └── qpso_tensor_optimized.py
    │
    ├── models/                  # Arquitecturas de redes neuronales
    │   ├── __init__.py
    │   └── ann.py               # Modelo QPSOCompatibleANN
    │
    ├── optimizers/              # Optimizadores QPSO para redes neuronales
    │   ├── __init__.py
    │   ├── qpso_nn.py           # QPSONNOptimizer, QDPSONNOptimizer
    │   └── training_strategies.py  # Estrategias Forward, Weighted, Layerwise
    │
    ├── trainers/                # Interfaz de entrenamiento de alto nivel
    │   ├── __init__.py
    │   └── trainer.py           # Clase Trainer con soporte CV
    │
    ├── utils/                   # Utilidades
    │   ├── __init__.py
    │   ├── data.py              # Carga y preprocesamiento de datos
    │   └── metrics.py           # Metricas de clasificacion y visualizacion
    │
    ├── data/                    # Cargadores de datasets
    │   ├── __init__.py
    │   └── mcw.py               # Dataset MCW (Multi-Class Weather)
    │
    ├── docs/                    # Documentacion detallada de modulos
    │   ├── index.md             # Indice de documentacion
    │   ├── models.md            # Documentacion de modelos
    │   ├── optimizers.md        # Documentacion de optimizadores
    │   ├── trainers.md          # Documentacion de trainers
    │   └── *.md                 # Documentacion especifica de scripts
    │
    ├── main_qpso.py             # Script benchmark QPSO
    ├── main_qdpso.py            # Script benchmark QDPSO
    ├── main_mcw.py              # Benchmark clasificacion imagenes MCW
    ├── main_training_type.py    # Comparacion de estrategias de entrenamiento
    ├── main_hyperparameter_search.py  # HPO con Optuna
    ├── start_hyperparameter_search.py # Script configuracion HPO
    └── usage_cases.py           # Ejemplos de uso
```

---

## Resumen de Modulos

### 1. Modulo QPSO-PyTorch

Las implementaciones base del algoritmo QPSO con optimizaciones progresivas:

| Archivo | Descripcion | Caso de Uso |
|---------|-------------|-------------|
| `qpso.py` | Implementacion NumPy basada en pypi qpso 0.0.1 | Referencia/aprendizaje |
| `qpso_tensor.py` | Implementacion basica con tensores PyTorch | Optimizacion GPU simple |
| `qpso_tensor_optimized.py` | Implementacion completamente optimizada | Uso en produccion |

**Optimizaciones en qpso_tensor_optimized.py (17 mejoras):**

- **Rendimiento**: Generacion eficiente de signos, memory pooling, `torch.no_grad()`
- **Estabilidad**: Division segura, dtype configurable, manejo de epsilon
- **Funcionalidad**: Manejo de limites (clamp/reflect/wrap/random), convergencia temprana, historial
- **Robustez**: Validacion de parametros, manejo de NaN/Inf
- **Usabilidad**: Dataclass `OptimizationResult`, soporte context manager
- **Extensibilidad**: Sistema de callbacks basado en eventos (ON_INIT, ON_ITERATION_START, ON_NEW_BEST, etc.)

### 2. Modulo ANN

Framework de entrenamiento de redes neuronales usando QPSO:

| Submodulo | Descripcion | Clases Principales |
|-----------|-------------|-------------------|
| `models/` | Redes neuronales compatibles con QPSO | `QPSOCompatibleANN` |
| `optimizers/` | Optimizadores para redes neuronales | `QPSONNOptimizer`, `QDPSONNOptimizer` |
| `trainers/` | Interfaz de entrenamiento de alto nivel | `Trainer`, `TrainingConfig` |
| `utils/` | Utilidades de datos y metricas | `load_dataset`, `MulticlassMetrics` |
| `data/` | Cargadores de datasets | `MCWDataset`, `load_mcw` |

**Estrategias de Entrenamiento:**

| Estrategia | Descripcion | Mejor Para |
|------------|-------------|------------|
| **Forward** | Optimiza todos los pesos simultaneamente | Redes pequenas, entrenamiento rapido |
| **Weighted** | Fitness ponderado por capa con decaimiento | Redes medianas, balanceado |
| **Layerwise** | Entrenamiento secuencial capa por capa | Redes profundas, mejor convergencia |

---

## Instalacion

> **Guia Detallada**: Ver [docs/installation_es.md](docs/installation_es.md) para instrucciones completas de instalacion.

### Requisitos

| Componente | Minimo | Recomendado |
|------------|--------|-------------|
| **Python** | >= 3.10 | 3.12 |
| **PyTorch** | >= 2.0.0 | 2.5.1 |
| **CUDA** | - | 12.4 |
| **cuDNN** | - | 9.1.0 |

### Instalacion Rapida con Conda (Recomendado)

```bash
# Clonar el repositorio
git clone https://github.com/stalyn21/qpso-with-pytorch.git
cd pytorch-qpso-suite

# Crear entorno desde archivo
conda env create -f environment.yml

# Activar entorno
conda activate pytorch_qpso_gpu
```

### Instalacion con Pip

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Instalar PyTorch con CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Instalar dependencias del proyecto
pip install -r requirements.txt
```

### Instalacion Solo CPU

```bash
# Instalar version CPU de PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar dependencias del proyecto
pip install -r requirements.txt
```

### Verificacion

```bash
# Verificar PyTorch y CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Verificar modulo ann
python -c "from ann import QPSOCompatibleANN, Trainer; print('modulo ann OK')"
```

---

## Datasets

### Dataset MCW (Multi-Class Weather)

El dataset MCW **no está incluido** en este repositorio debido a su tamaño. Debes descargarlo por separado para ejecutar los ejemplos de clasificación de imágenes de clima.

#### Descarga

Descarga el dataset desde Kaggle:

**[Multi-Class Weather Dataset](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset)**

#### Instalación

1. Descarga y extrae el dataset
2. Coloca las imágenes en la siguiente estructura:

```
ann/data/img/mcw/
├── cloudy/
│   ├── cloudy1.jpg
│   ├── cloudy2.jpg
│   └── ...
├── rain/
│   ├── rain1.jpg
│   ├── rain2.jpg
│   └── ...
├── shine/
│   ├── shine1.jpg
│   ├── shine2.jpg
│   └── ...
└── sunrise/
    ├── sunrise1.jpg
    ├── sunrise2.jpg
    └── ...
```

#### Solución de Problemas

Algunas imágenes del dataset pueden causar problemas de carga debido a corrupción o formatos no soportados. Si encuentras errores durante la carga de datos:

1. **Identificar imágenes problemáticas**: El mensaje de error mostrará qué archivo causó el problema
2. **Eliminar o reemplazar**: Borra la imagen problemática o reemplázala con una válida
3. **Problemas comunes**:
   - Archivos JPEG truncados/corruptos
   - Imágenes con espacios de color inusuales
   - Archivos con extensiones incorrectas

```python
# Ejemplo: Verificar si una imagen carga correctamente
import cv2
img = cv2.imread('ruta/a/imagen.jpg')
if img is None:
    print("La imagen falló al cargar - elimínala o reemplázala")
```

### Otros Datasets

Los siguientes datasets se descargan automáticamente via scikit-learn:
- **Iris**: 150 muestras, 4 features, 3 clases
- **Wine**: 178 muestras, 13 features, 3 clases
- **Breast Cancer**: 569 muestras, 30 features, 2 clases
- **Digits**: 1797 muestras, 64 features, 10 clases

No se requiere configuración adicional para estos datasets.

---

## Inicio Rapido

### 1. Optimizacion Basica con QPSO

```python
from QPSO_PyTorch.tensor_qpso import QPSOTensorOptimized

# Definir funcion de costo
def sphere(x):
    return (x ** 2).sum()

# Crear optimizador
optimizer = QPSOTensorOptimized(
    cf=sphere,
    size=50,                    # Numero de particulas
    dim=10,                     # Dimensiones
    bounds=[(-5, 5)] * 10,      # Limites de busqueda
    maxIters=1000,              # Iteraciones maximas
    alpha=(1.0, 0.5),           # Decaimiento lineal de 1.0 a 0.5
    device='cuda',              # Usar GPU
    track_history=True          # Registrar historial de optimizacion
)

# Ejecutar optimizacion
result = optimizer.optimize()
print(f"Mejor valor: {result.best_value:.6e}")
print(f"Iteraciones: {result.iterations}")
print(f"Tiempo: {result.elapsed_time:.2f}s")
```

### 2. Entrenamiento de Redes Neuronales con QPSO

```python
from ann.trainers import Trainer, TrainingConfig
from ann.utils import load_dataset

# Cargar dataset
X_train, X_test, y_train, y_test = load_dataset('iris')

# Configurar entrenamiento
config = TrainingConfig(
    hidden_layers=[16, 8],      # Arquitectura de la red
    activation='tanh',          # Funcion de activacion
    n_particles=30,             # Tamano del enjambre
    max_iters=100,              # Iteraciones maximas
    alpha=(1.0, 0.5),           # Parametro alpha de QPSO
    patience=30,                # Paciencia para early stopping
    random_state=42             # Reproducibilidad
)

# Crear trainer y entrenar
trainer = Trainer(input_dim=4, output_dim=3, config=config)
result = trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)

print(f"Test Accuracy: {result.test_accuracy:.4f}")
```

### 3. Comparar Estrategias de Entrenamiento

```python
from ann.optimizers.training_strategies import create_training_strategy, StrategyConfig
from ann.models import QPSOCompatibleANN

# Crear modelo
model = QPSOCompatibleANN(
    input_dim=4,
    output_dim=3,
    hidden_layers=[16, 8]
)

# Probar diferentes estrategias
for strategy_name in ['forward', 'weighted', 'layerwise']:
    config = StrategyConfig(
        strategy=strategy_name,
        n_particles=30,
        max_iters=100
    )

    strategy = create_training_strategy(model, config)
    result = strategy.train(X_train, y_train, X_val, y_val)
    print(f"{strategy_name}: {result['val_accuracy']:.4f}")
```

### 4. Optimizacion de Hiperparametros

```bash
# Editar start_hyperparameter_search.py para configurar espacio de busqueda
cd ann
python start_hyperparameter_search.py
```

---

## Documentacion Detallada

### Indices de Documentacion

| Modulo | Indice | Descripcion |
|--------|--------|-------------|
| **QPSO-PyTorch** | [📚 index_es.md](QPSO-PyTorch/docs/index_es.md) | Implementaciones del algoritmo QPSO |
| **ANN** | [📚 index_es.md](ann/docs/index_es.md) | Entrenamiento de redes neuronales con QPSO |

### Modulo QPSO-PyTorch

| Documento | Descripcion |
|-----------|-------------|
| [docs_qpso.md](QPSO-PyTorch/docs/docs_qpso.md) | Implementacion de referencia NumPy (basada en pypi) |
| [docs_qpso_tensor.md](QPSO-PyTorch/docs/docs_qpso_tensor.md) | Implementacion basica con tensores PyTorch |
| [docs_qpso_tensor_optimized.md](QPSO-PyTorch/docs/docs_qpso_tensor_optimized.md) | Implementacion optimizada (17 mejoras) |
| [implementation_comparison_es.md](QPSO-PyTorch/docs/implementation_comparison_es.md) | **Comparacion de implementaciones**: enfoque de codificacion, rendimiento, caracteristicas |

### Modulo ANN

| Documento | Descripcion |
|-----------|-------------|
| [index_es.md](ann/docs/index_es.md) | Vista general y arquitectura del modulo |
| [models.md](ann/docs/models.md) | Documentacion del modelo QPSOCompatibleANN |
| [optimizers.md](ann/docs/optimizers.md) | QPSONNOptimizer, estrategias de entrenamiento |
| [trainers.md](ann/docs/trainers.md) | Clase Trainer y configuracion |
| [utils.md](ann/docs/utils.md) | Carga de datos y metricas |

### Scripts Ejecutables

| Script | Documento | Descripcion |
|--------|-----------|-------------|
| `main_qpso.py` | [main_qpso.md](ann/docs/main_qpso.md) | Benchmark QPSO en datasets clasicos |
| `main_qdpso.py` | [main_qdpso.md](ann/docs/main_qdpso.md) | Benchmark QDPSO |
| `main_mcw.py` | [main_mcw.md](ann/docs/main_mcw.md) | Clasificacion de imagenes MCW |
| `main_training_type.py` | [main_training_type.md](ann/docs/main_training_type.md) | Comparacion de estrategias |
| `main_hyperparameter_search.py` | [main_hyperparameter_search.md](ann/docs/main_hyperparameter_search.md) | HPO con Optuna |
| `usage_cases.py` | [usage_cases.md](ann/docs/usage_cases.md) | Ejemplos educativos |

---

## Referencia de API

### QPSOTensorOptimized

```python
QPSOTensorOptimized(
    cf: Callable,                       # Funcion de costo a minimizar/maximizar
    size: int,                          # Numero de particulas
    dim: int,                           # Dimensiones del problema
    bounds: List[Tuple[float, float]],  # Limites por dimension [(min, max), ...]
    maxIters: int,                      # Iteraciones maximas
    alpha: Union[float, Tuple] = 0.75,  # Coeficiente de contraccion-expansion
                                        # float: valor fijo
                                        # tuple: (max, min) para decaimiento lineal
    device: str = 'auto',               # 'cpu', 'cuda', 'cuda:N', 'mps', 'auto'
    dtype: torch.dtype = torch.float32, # Tipo de dato del tensor
    seed: Optional[int] = None,         # Semilla aleatoria para reproducibilidad
    boundary_strategy: str = 'clamp',   # 'none', 'clamp', 'reflect', 'wrap', 'random'
    tol: float = 1e-12,                 # Tolerancia de convergencia
    patience: int = 100,                # Iteraciones sin mejora antes de parar
    track_history: bool = False,        # Registrar historial de optimizacion
    minimize: bool = True               # True para minimizar, False para maximizar
)

# Metodos
result = optimizer.optimize(callback=None, interval=None)  # Retorna OptimizationResult
```

### QDPSOTensorOptimized

```python
QDPSOTensorOptimized(
    # Mismos parametros que QPSOTensorOptimized, excepto:
    g: float = 0.96,                    # Parametro delta en lugar de alpha
)
```

### QPSOCompatibleANN

```python
QPSOCompatibleANN(
    input_dim: int,                     # Dimension de entrada
    output_dim: int,                    # Dimension de salida (clases)
    hidden_layers: List[int],           # Neuronas por capa oculta [64, 32, 16]
    activation: str = 'relu',           # 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'gelu'
    output_activation: str = 'softmax', # 'softmax', 'sigmoid', 'log_softmax', None
    dropout: float = 0.0,               # Probabilidad de dropout
    use_batch_norm: bool = False,       # Usar batch normalization
    device: str = 'auto'                # Dispositivo
)

# Propiedades
model.num_params                        # Numero total de parametros

# Metodos
params = model.get_flat_params()        # Obtener todos los params como tensor 1D
model.set_flat_params(params)           # Establecer params desde tensor 1D
bounds = model.get_param_bounds(1.0)    # Obtener limites para QPSO
```

### Trainer

```python
Trainer(
    input_dim: int,
    output_dim: int,
    config: TrainingConfig
)

# Metodos
result = trainer.fit(X_train, y_train, X_test=None, y_test=None)
result = trainer.fit_cv(X, y, X_test=None, y_test=None)  # Validacion cruzada
predictions = trainer.predict(X)
trainer.save_model(path)
trainer.load_model(path)
```

### TrainingConfig

```python
TrainingConfig(
    # Arquitectura
    hidden_layers: List[int] = [32, 16],
    activation: str = 'tanh',

    # Parametros QPSO
    n_particles: int = 50,
    max_iters: int = 100,
    alpha: Tuple[float, float] = (1.0, 0.5),
    g: float = 0.96,                    # Para QDPSO
    use_qdpso: bool = False,

    # Entrenamiento
    weight_bound: float = 1.0,
    patience: int = 50,

    # Validacion cruzada
    n_folds: int = 5,

    # Otros
    random_state: int = 42,
    verbose: bool = True,
    save_best_model: bool = False,
    output_dir: str = './output'
)
```

---

## Resultados de Ejemplo

### Benchmarks en Datasets Clasicos

| Dataset | Clases | Features | QPSO Acc | QDPSO Acc |
|---------|--------|----------|----------|-----------|
| Iris | 3 | 4 | 96.7% | 97.3% |
| Wine | 3 | 13 | 94.4% | 95.6% |
| Breast Cancer | 2 | 30 | 96.5% | 97.1% |
| Digits | 10 | 64 | 92.3% | 93.8% |

### MCW (Clasificacion de Imagenes de Clima)

| Metrica | QPSO | QDPSO |
|---------|------|-------|
| Accuracy | 82.5% | 85.0% |
| F1-Score | 0.81 | 0.84 |
| Precision | 0.83 | 0.86 |
| Recall | 0.80 | 0.83 |
| Cohen's Kappa | 0.76 | 0.80 |

### Comparacion de Estrategias de Entrenamiento

| Estrategia | Accuracy | Tiempo de Entrenamiento | Mejor Para |
|------------|----------|------------------------|------------|
| Forward | 94.2% | Rapido | Redes pequenas |
| Weighted | 95.1% | Medio | Balanceado |
| Layerwise | 96.3% | Lento | Redes profundas |

### Mejor Configuracion HPO

```
Optimizador: QDPSO
Estrategia: Layerwise
g: 0.9534
Particulas: 52
Arquitectura: [28, 18, 11]
Iteraciones por capa: 45
Iteraciones fine-tune: 30

Resultados:
  CV F1-Score: 0.865
  Test Accuracy: 87.5%
  Test F1-Score: 0.871
```

---

## Contribuir

Las contribuciones son bienvenidas! Por favor:

1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/FuncionalidadIncreible`)
3. Haz commit de tus cambios (`git commit -m 'Agregar FuncionalidadIncreible'`)
4. Push a la rama (`git push origin feature/FuncionalidadIncreible`)
5. Abre un Pull Request

---

## Referencias

### Papers Fundamentales

1. **QPSO Original**: Sun, J., Feng, B., & Xu, W. (2004). *Particle swarm optimization with particles having quantum behavior*. Congress on Evolutionary Computation.

2. **Analisis QPSO**: Sun, J., Fang, W., Wu, X., Palade, V., & Xu, W. (2012). *Quantum-behaved particle swarm optimization: Analysis of individual particle behavior and parameter selection*. Evolutionary Computation.

3. **Optuna**: Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD.

### Implementaciones de Referencia

- [QPSO PyPI Package](https://pypi.org/project/qpso/) - Paquete Python original (v0.0.1)

---

## Citacion

Si usas este software en tu investigacion o proyecto, por favor citalo:

### Formato BibTeX

```bibtex
@software{chancay2024pytorchqpso,
  author       = {Chancay Moreira, Stalyn Javier},
  title        = {PyTorch QPSO Suite: Quantum Particle Swarm Optimization for Neural Network Training},
  year         = {2024},
  version      = {2.0.0},
  url          = {https://github.com/stalyn21/qpso-with-pytorch}
}
```

### Formato APA

> Chancay Moreira, S. J. (2024). *PyTorch QPSO Suite: Quantum Particle Swarm Optimization for Neural Network Training* (Version 2.0.0) [Software]. https://github.com/stalyn21/qpso-with-pytorch

---

## Licencia

Este proyecto esta bajo la licencia MIT. Ver [LICENSE](LICENSE) para mas detalles.

**Copyright (c) 2024 Stalyn Javier Chancay Moreira**

Se permite el uso, copia, modificacion y distribucion de este software siempre que:
- Se mantenga el aviso de copyright original
- Se incluya la licencia MIT en cualquier redistribucion

---

## Autor

**Stalyn Javier Chancay Moreira**

- GitHub: [@stalyn21](https://github.com/stalyn21)

---

## Agradecimientos

Este proyecto esta basado en los trabajos fundamentales de:
- Sun, J., Feng, B., & Xu, W. - Creadores del algoritmo QPSO original
- La comunidad de PyTorch por el excelente framework de deep learning
- El equipo de Optuna por el framework de optimizacion de hiperparametros

---

## Registro de Cambios

### v2.0.0 (2024)
- **Refactorizacion completa** del codigo
- Renombrado del modulo `src/` a `ann/` para mayor claridad
- Organizacion mejorada de modulos e imports
- Documentacion mejorada con referencia de API detallada
- Type hints completos en todo el codigo
- Optimizaciones de rendimiento en operaciones tensoriales
- Manejo de errores y validacion mejorados
- Soporte agregado para Apple Silicon (MPS)
- Actualizacion de toda la documentacion para reflejar la nueva estructura

### v1.0.0 (2024)
- Implementacion inicial de QPSO/QDPSO con tensores PyTorch
- Framework modular para entrenamiento de redes neuronales sin backpropagation
- Tres estrategias de entrenamiento: Forward, Weighted, Layerwise
- Soporte HPO (Hyperparameter Optimization) con Optuna
- Aceleracion GPU completa
- Documentacion completa en espanol e ingles
