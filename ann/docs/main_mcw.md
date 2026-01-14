# Documentacion: main_mcw.py

[🏠 README](../../README.md) | [📚 Index](index.md) | [⬅️ QDPSO](main_qdpso.md) | **MCW Script** | [Next: Training Type ➡️](main_training_type.md)

---

## Descripcion General

Script de benchmark para entrenar redes neuronales en el dataset **MCW (Multi-Class Weather)** usando optimizadores QPSO y QDPSO. Permite comparar el rendimiento de ambos optimizadores en clasificacion de imagenes meteorologicas, con soporte para reduccion de dimensionalidad.

### Caracteristicas Principales

- **Dual Optimizer**: Ejecuta QPSO y QDPSO en paralelo para comparacion
- **Feature Extraction**: Extrae 84 features de imagenes (histogram, haralick, hu moments)
- **Reduccion de Dimensionalidad**: Soporta PCA, Isomap y MDS
- **Cross-Validation**: Validacion cruzada de k-folds
- **Visualizacion Completa**: Genera 5 tipos de graficas por optimizador
- **Metricas Detalladas**: Accuracy, Precision, Recall, F1, Kappa, MCC por clase

---

## Tabla de Contenidos

1. [Dataset MCW](#dataset-mcw)
2. [Extraccion de Features](#extraccion-de-features)
3. [Reduccion de Dimensionalidad](#reduccion-de-dimensionalidad)
4. [Configuracion del Script](#configuracion-del-script)
5. [Arquitectura de Red](#arquitectura-de-red)
6. [Uso del Script](#uso-del-script)
7. [Modulo de Datos](#modulo-de-datos)
8. [Metricas de Evaluacion](#metricas-de-evaluacion)
9. [Salidas y Graficas](#salidas-y-graficas)
10. [Ejemplo de Ejecucion](#ejemplo-de-ejecucion)
11. [Dependencias](#dependencias)
12. [Notas y Consideraciones](#notas-y-consideraciones)

---

## Dataset MCW

### Descripcion

El dataset **Multi-Class Weather (MCW)** contiene imagenes de diferentes condiciones climaticas, organizadas en 4 categorias:

| Clase | Indice | Descripcion | Caracteristicas Tipicas |
|-------|--------|-------------|------------------------|
| **cloudy** | 0 | Cielo nublado | Tonos grises, textura uniforme, baja saturacion |
| **rain** | 1 | Lluvia | Gotas visibles, cielo oscuro, patron de lluvia |
| **shine** | 2 | Soleado/brillante | Alta luminosidad, cielo azul, sombras marcadas |
| **sunrise** | 3 | Amanecer | Tonos calidos, gradiente de color, horizonte visible |

### Estructura de Directorios

```
data/img/mcw/
├── cloudy/           # ~300 imagenes de cielo nublado
│   ├── cloudy001.jpg
│   ├── cloudy002.jpg
│   └── ...
├── rain/             # ~214 imagenes de lluvia
│   ├── rain001.jpg
│   └── ...
├── shine/            # ~252 imagenes de cielo soleado
│   ├── shine001.jpg
│   └── ...
└── sunrise/          # ~357 imagenes de amanecer
    ├── sunrise001.jpg
    └── ...
```

### Estadisticas del Dataset

| Metrica | Valor |
|---------|-------|
| Total imagenes | ~1,123 |
| Clases | 4 |
| Tamaño procesamiento | 150x150 pixels |
| Formato soportado | JPG, PNG, BMP |

---

## Extraccion de Features

El modulo extrae tres tipos de caracteristicas de cada imagen, totalizando **84 features**:

### 1. Histograma de Color HSV (64 features)

**Que es**: Distribucion estadistica de colores en el espacio HSV (Hue, Saturation, Value).

**Como funciona**:
1. Convierte la imagen de BGR a HSV
2. Calcula histograma 3D con `bins=4` por canal
3. Genera `4^3 = 64` features

**Por que es util**:
- Captura la distribucion de colores independiente de la posicion
- HSV separa el color (H) de la intensidad (V), siendo mas robusto a cambios de iluminacion
- Distingue cielos azules (shine) de cielos grises (cloudy) o rojizos (sunrise)

```python
# Implementacion interna
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0, 1, 2], None,
                     [bins, bins, bins],
                     [0, 256, 0, 256, 0, 256])
cv2.normalize(hist, hist)
features = hist.flatten()  # 64 valores
```

### 2. Caracteristicas Haralick (13 features)

**Que es**: Medidas estadisticas de textura basadas en la matriz GLCM (Gray Level Co-occurrence Matrix).

**Como funciona**:
1. Convierte a escala de grises
2. Calcula la matriz GLCM en 4 direcciones (0°, 45°, 90°, 135°)
3. Extrae 13 estadisticas de Haralick, promediando las 4 direcciones

**Las 13 metricas Haralick**:

| # | Nombre | Descripcion |
|---|--------|-------------|
| 1 | Angular Second Moment | Uniformidad/homogeneidad de la textura |
| 2 | Contrast | Variacion local de intensidad |
| 3 | Correlation | Correlacion lineal entre pixeles vecinos |
| 4 | Sum of Squares: Variance | Varianza de la distribucion |
| 5 | Inverse Difference Moment | Homogeneidad local |
| 6 | Sum Average | Promedio de sumas |
| 7 | Sum Variance | Varianza de sumas |
| 8 | Sum Entropy | Entropia de sumas |
| 9 | Entropy | Aleatoriedad de la textura |
| 10 | Difference Variance | Varianza de diferencias |
| 11 | Difference Entropy | Entropia de diferencias |
| 12-13 | Information Measures | Medidas de correlacion de informacion |

**Por que es util**:
- Captura patrones de textura (gotas de lluvia vs cielo liso)
- Diferencia nubes densas de cielos despejados
- Detecta patrones repetitivos en imagenes

```python
# Implementacion interna
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
haralick = mahotas.features.haralick(gray).mean(axis=0)  # 13 valores
```

### 3. Momentos de Hu (7 features)

**Que es**: Momentos invariantes que describen la forma/estructura de la imagen.

**Como funciona**:
1. Convierte a escala de grises
2. Calcula momentos geometricos de la imagen
3. Deriva los 7 momentos invariantes de Hu

**Los 7 momentos de Hu**:

| # | Formula | Propiedad |
|---|---------|-----------|
| 1 | `η20 + η02` | Inercia alrededor del centro |
| 2 | `(η20 - η02)² + 4η11²` | Excentricidad |
| 3-4 | Combinaciones de η30, η12, η21, η03 | Orientacion y simetria |
| 5-6 | Productos cruzados | Invariantes de orden mayor |
| 7 | Combinacion especial | Distingue imagenes espejo |

**Por que es util**:
- Invariantes a traslacion, escala y rotacion
- Capturan la "firma" estructural de la imagen
- Utiles para distinguir formas de nubes y patrones de luz

```python
# Implementacion interna
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
moments = cv2.moments(gray)
hu = cv2.HuMoments(moments).flatten()  # 7 valores
```

### Resumen de Features

| Tipo | Cantidad | Informacion Capturada |
|------|----------|----------------------|
| Histogram HSV | 64 | Distribucion de color |
| Haralick | 13 | Textura y patrones |
| Hu Moments | 7 | Forma y estructura |
| **Total** | **84** | |

---

## Reduccion de Dimensionalidad

El script soporta tres metodos para reducir las 84 features a un espacio de menor dimension:

### 1. PCA (Principal Component Analysis)

**Que es**: Transformacion lineal que proyecta los datos en direcciones de maxima varianza.

**Como funciona**:
1. Centra los datos (resta la media)
2. Calcula la matriz de covarianza
3. Encuentra autovectores (componentes principales)
4. Proyecta datos en los primeros k componentes

**Ecuacion**:
```
Z = X · W
```
Donde:
- X: datos originales (n_samples × 84)
- W: matriz de autovectores (84 × k)
- Z: datos reducidos (n_samples × k)

**Ventajas**:
- Muy rapido (O(n²d) donde d=features)
- Preserva maxima varianza
- Soporta `transform()` para nuevos datos

**Desventajas**:
- Solo captura relaciones lineales
- Puede perder estructuras no lineales

**Uso recomendado**: Primera opcion para explorar, datos con estructura lineal.

### 2. Isomap (Isometric Mapping)

**Que es**: Reduccion no lineal que preserva distancias geodesicas en un manifold.

**Como funciona**:
1. Construye grafo k-NN (k vecinos mas cercanos)
2. Calcula distancias geodesicas (camino mas corto en el grafo)
3. Aplica MDS a las distancias geodesicas

**Ecuacion**:
```
D_G[i,j] = shortest_path(i, j)   # Distancia geodesica
Z = MDS(D_G)                      # Proyeccion que preserva D_G
```

**Ventajas**:
- Captura estructuras no lineales (manifolds)
- Preserva la geometria intrinseca de los datos
- Soporta `transform()` para nuevos datos

**Desventajas**:
- Mas lento que PCA
- Sensible a ruido y outliers
- Requiere elegir n_neighbors

**Uso recomendado**: Datos con estructuras curvas o manifolds.

### 3. MDS (Multidimensional Scaling)

**Que es**: Busca configuracion de puntos que preserve distancias originales.

**Como funciona**:
1. Calcula matriz de distancias par a par
2. Optimiza posiciones para minimizar stress:
   ```
   stress = sqrt(Σ(d_ij - δ_ij)² / Σd_ij²)
   ```
   Donde d_ij = distancia original, δ_ij = distancia en espacio reducido

**Ventajas**:
- Preserva distancias directamente
- No asume linealidad

**Desventajas**:
- **No soporta `transform()`**: Cada conjunto requiere fit separado
- Muy lento para datasets grandes
- Puede dar resultados inconsistentes entre train/test

**Uso recomendado**: Solo para analisis exploratorio, no para produccion.

### Comparativa de Metodos

| Aspecto | PCA | Isomap | MDS |
|---------|-----|--------|-----|
| Tipo | Lineal | No lineal | No lineal |
| Velocidad | Muy rapida | Media | Lenta |
| `transform()` | Si | Si | No |
| Memoria | Baja | Media | Alta |
| Consistencia train/test | Alta | Alta | Baja |
| Recomendacion | Default | Datos complejos | Evitar |

### Configuracion

```python
MCW_CONFIG = {
    'reduction_method': 'pca',    # 'pca', 'isomap', 'mds', o None
    'n_components': 20,           # Dimensiones finales
}
```

---

## Configuracion del Script

### Estructura de Configuracion

El script usa tres diccionarios de configuracion principales:

### 1. MCW_CONFIG - Dataset

```python
MCW_CONFIG = {
    'root_path': './data/img/mcw',  # Ruta al directorio de imagenes
    'reduction_method': 'isomap',    # Metodo de reduccion
    'n_components': 7,               # Dimensiones finales
    'img_size': (150, 150),          # Tamaño de redimension
    'bins': 4,                       # Bins para histograma HSV
}
```

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `root_path` | str | `'./data/img/mcw'` | Directorio con subdirectorios de clases |
| `reduction_method` | str/None | `'pca'` | `'pca'`, `'isomap'`, `'mds'`, o `None` |
| `n_components` | int/None | `20` | Dimensiones finales (None = n_features/6) |
| `img_size` | tuple | `(150, 150)` | Ancho x Alto para redimension |
| `bins` | int | `4` | Bins por canal para histograma (genera bins³ features) |

### 2. TRAINING_CONFIG - Entrenamiento

```python
TRAINING_CONFIG = {
    'activation': 'tanh',            # Activacion capas ocultas
    'output_activation': 'softmax',  # Activacion capa salida
    'n_particles': 50,               # Particulas del enjambre
    'max_iters': 1000,               # Iteraciones maximas
    'n_folds': 4,                    # Folds para CV
    'train_size': 0.70,              # Proporcion entrenamiento
    'val_size': 0.10,                # Proporcion validacion
    'test_size': 0.20,               # Proporcion test
    'random_state': 42,              # Semilla aleatoria
    'patience': 50,                  # Early stopping
}
```

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `activation` | str | `'tanh'` | Funcion de activacion ocultas (`'tanh'`, `'relu'`, `'sigmoid'`) |
| `output_activation` | str | `'softmax'` | Activacion de salida (`'softmax'` para multiclase) |
| `n_particles` | int | `50` | Numero de particulas en el enjambre |
| `max_iters` | int | `1000` | Iteraciones maximas de optimizacion |
| `n_folds` | int | `4` | Numero de folds para cross-validation |
| `train_size` | float | `0.70` | Proporcion de datos para entrenamiento |
| `val_size` | float | `0.10` | Proporcion para validacion |
| `test_size` | float | `0.20` | Proporcion para test |
| `random_state` | int | `42` | Semilla para reproducibilidad |
| `patience` | int | `50` | Iteraciones sin mejora antes de parar |

### 3. Configuracion de Optimizadores

```python
QPSO_CONFIG = {
    'alpha': (1.0, 0.5),  # Alpha inicial y final (decay lineal)
}

QDPSO_CONFIG = {
    'g': 0.96,            # Factor g constante
}

OPTIMIZERS = ['QPSO', 'QDPSO']  # Optimizadores a ejecutar
```

| Optimizador | Parametro | Default | Descripcion |
|-------------|-----------|---------|-------------|
| QPSO | `alpha` | `(1.0, 0.5)` | Factor contraccion-expansion con decay |
| QDPSO | `g` | `0.96` | Factor de control constante |

---

## Arquitectura de Red

### Formula de Arquitectura

```
Entrada (n_features) → Oculta1 (n_features × 3) → Oculta2 (n_features × 2) → Salida (4)
```

### Ejemplo con Reduccion

Con `n_components=7`:

```
7 → 21 → 14 → 4
```

**Calculo de parametros**:
```
Capa 1: (7 + 1) × 21 = 168 pesos
Capa 2: (21 + 1) × 14 = 308 pesos
Capa 3: (14 + 1) × 4 = 60 pesos
─────────────────────────────
Total: 536 parametros
```

### Formula General de Parametros

```
n_params = (input_dim + 1) × hidden1 + (hidden1 + 1) × hidden2 + (hidden2 + 1) × output_dim
```

Donde `+1` representa los bias de cada capa.

### Activaciones

| Capa | Activacion | Formula | Rango |
|------|------------|---------|-------|
| Oculta 1 | tanh | `(e^x - e^-x) / (e^x + e^-x)` | [-1, 1] |
| Oculta 2 | tanh | `(e^x - e^-x) / (e^x + e^-x)` | [-1, 1] |
| Salida | softmax | `e^xi / Σe^xj` | [0, 1], suma=1 |

---

## Uso del Script

### Ejecucion Basica

```bash
# Activar entorno
conda activate pytorch_qpso_gpu

# Ejecutar benchmark
python ann/main_mcw.py
```

### Modificar Configuracion

Edita las constantes al inicio del script:

```python
# Cambiar a PCA con 20 componentes
MCW_CONFIG = {
    'root_path': './data/img/mcw',
    'reduction_method': 'pca',
    'n_components': 20,
    ...
}

# Solo ejecutar QPSO
OPTIMIZERS = ['QPSO']

# Aumentar particulas
TRAINING_CONFIG = {
    ...
    'n_particles': 100,
    'max_iters': 500,
    ...
}
```

### Uso Programatico

```python
# Importar modulo de datos
from ann.data import load_mcw, MCW_CLASSES

# Cargar dataset con reduccion
data = load_mcw(
    root_path='./data/img/mcw',
    reduction_method='isomap',
    n_components=15,
    train_size=0.7,
    val_size=0.1,
    test_size=0.2
)

# Acceder a los datos
print(f"Train shape: {data.X_train.shape}")
print(f"Classes: {data.class_names}")
print(f"Features: {data.n_features}")
```

---

## Modulo de Datos

### Ubicacion

```
src/data/
├── __init__.py    # Exports
└── mcw.py         # Implementacion
```

### Clase MCWDataset

```python
from ann.data import MCWDataset

dataset = MCWDataset(
    root_path='./data/img/mcw',
    img_size=(150, 150),
    bins=4
)

result = dataset.load(
    train_size=0.70,
    val_size=0.10,
    test_size=0.20,
    reduction_method='pca',
    n_components=20,
    random_state=42,
    verbose=True
)
```

### Funcion load_mcw

```python
from ann.data import load_mcw

data = load_mcw(
    root_path='./data/img/mcw',
    train_size=0.70,
    val_size=0.10,
    test_size=0.20,
    reduction_method='isomap',
    n_components=15,
    random_state=42,
    img_size=(150, 150),
    bins=4,
    verbose=True
)
```

### MCWDataResult (Dataclass)

```python
@dataclass
class MCWDataResult:
    X_train: np.ndarray       # Features de entrenamiento
    X_val: np.ndarray         # Features de validacion
    X_test: np.ndarray        # Features de test
    y_train: np.ndarray       # Labels de entrenamiento
    y_val: np.ndarray         # Labels de validacion
    y_test: np.ndarray        # Labels de test
    class_names: List[str]    # ['cloudy', 'rain', 'shine', 'sunrise']
    n_features: int           # Numero de features (post-reduccion)
    n_classes: int            # Numero de clases (4)
    reduction_method: str     # Metodo usado o None
    n_components: int         # Componentes usados o None
    original_features: int    # Features originales (84)
```

---

## Metricas de Evaluacion

El script calcula metricas comprehensivas para evaluar el rendimiento. A continuacion se detalla cada una:

### 1. Accuracy (Exactitud)

**Definicion**: Proporcion de predicciones correctas sobre el total.

**Formula**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Para multiclase:
Accuracy = Correctas / Total = Σ(y_pred == y_true) / n_samples
```

**Interpretacion**:
- Rango: [0, 1] donde 1 es perfecto
- Facil de entender
- **Limitacion**: Puede ser enganosa en clases desbalanceadas

**Ejemplo**:
```
y_true = [0, 0, 1, 1, 2, 2]
y_pred = [0, 0, 1, 2, 2, 2]
Accuracy = 5/6 = 0.8333
```

### 2. Precision

**Definicion**: De todas las predicciones positivas, cuantas fueron correctas.

**Formula**:
```
Precision_clase_i = TP_i / (TP_i + FP_i)
```

**Macro Precision** (promedio simple):
```
Precision_macro = (1/k) × Σ Precision_i
```

**Weighted Precision** (ponderado por soporte):
```
Precision_weighted = Σ (n_i / n_total) × Precision_i
```

**Interpretacion**:
- Alta precision = pocas falsas alarmas
- Importante cuando el costo de FP es alto
- Ejemplo: En diagnostico medico, evitar declarar enfermo a alguien sano

**Ejemplo por clase**:
```
Clase 0 (cloudy): De 50 predicciones como cloudy, 46 eran correctas
Precision = 46/50 = 0.92
```

### 3. Recall (Sensibilidad / TPR)

**Definicion**: De todos los positivos reales, cuantos fueron detectados.

**Formula**:
```
Recall_clase_i = TP_i / (TP_i + FN_i)
```

**Macro Recall**:
```
Recall_macro = (1/k) × Σ Recall_i
```

**Interpretacion**:
- Alto recall = pocos casos perdidos
- Importante cuando el costo de FN es alto
- Ejemplo: En deteccion de cancer, no perder casos positivos

**Ejemplo por clase**:
```
Clase 0 (cloudy): De 60 imagenes reales de cloudy, 46 fueron correctamente identificadas
Recall = 46/60 = 0.7667
```

### 4. F1-Score

**Definicion**: Media armonica de Precision y Recall.

**Formula**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Macro F1**:
```
F1_macro = (1/k) × Σ F1_i
```

**Por que media armonica**:
- Penaliza valores extremos
- Si Precision=1.0 y Recall=0.0, F1=0 (no 0.5)
- Requiere buen balance entre ambas metricas

**Interpretacion**:
- Rango: [0, 1]
- Util cuando hay desbalance de clases
- Metrica preferida para comparar modelos

**Ejemplo**:
```
Precision = 0.92, Recall = 0.77
F1 = 2 × (0.92 × 0.77) / (0.92 + 0.77) = 0.8384
```

### 5. Cohen's Kappa

**Definicion**: Mide el acuerdo entre predicciones y realidad, ajustando por azar.

**Formula**:
```
κ = (p_o - p_e) / (1 - p_e)
```

Donde:
- `p_o` = proporcion de acuerdo observado (accuracy)
- `p_e` = proporcion de acuerdo esperado por azar

**Calculo de p_e**:
```
p_e = Σ (n_pred_i × n_true_i) / n_total²
```

**Interpretacion**:

| Kappa | Interpretacion |
|-------|----------------|
| < 0 | Peor que azar |
| 0.00 - 0.20 | Leve |
| 0.21 - 0.40 | Regular |
| 0.41 - 0.60 | Moderado |
| 0.61 - 0.80 | Sustancial |
| 0.81 - 1.00 | Casi perfecto |

**Por que usarlo**:
- Corrige el sesgo de accuracy en clases desbalanceadas
- Un clasificador aleatorio tiene Kappa ≈ 0
- Mas informativo que accuracy para evaluacion real

**Ejemplo**:
```
Accuracy = 0.8756, pero...
Si clases estuvieran balanceadas al 25% cada una:
p_e = 4 × (0.25 × 0.25) = 0.25

Kappa = (0.8756 - 0.25) / (1 - 0.25) = 0.8341
```

### 6. Matriz de Confusion

**Definicion**: Tabla que muestra predicciones vs valores reales por clase.

**Estructura (4 clases)**:
```
                    Predicho
                    cloudy  rain  shine  sunrise
           cloudy     46     6      7       1     → 60 reales
Real       rain        5    36      1       1     → 43 reales
           shine       0     0     50       0     → 50 reales
           sunrise     3     0      4      65     → 72 reales
                      ↓     ↓      ↓       ↓
                      54    42     62      67   = 225 predicciones
```

**Como leerla**:
- **Diagonal**: Predicciones correctas (TP de cada clase)
- **Fila**: Distribucion de predicciones para esa clase real
- **Columna**: De que clases reales vienen las predicciones

**Metricas derivadas**:
```
Para clase "cloudy" (i=0):
- TP = 46 (diagonal)
- FP = 5 + 0 + 3 = 8 (columna sin diagonal)
- FN = 6 + 7 + 1 = 14 (fila sin diagonal)
- TN = resto = 225 - 46 - 8 - 14 = 157
```

### 7. Cross-Validation Metrics

**Definicion**: Metricas calculadas sobre k particiones del dataset.

**Proceso**:
1. Dividir datos en k folds
2. Para cada fold i:
   - Entrenar con k-1 folds
   - Validar con fold i
   - Registrar accuracy
3. Calcular media y desviacion estandar

**Metricas reportadas**:
- **CV Mean**: Promedio de accuracies de los k folds
- **CV Std**: Desviacion estandar (mide estabilidad)

**Interpretacion**:
```
CV Accuracy: 0.8497 +/- 0.0094
```
- Modelo logra ~85% en promedio
- Std bajo (0.0094) indica alta estabilidad
- Std alto indicaria overfitting o variabilidad

### Resumen de Metricas

| Metrica | Formula | Rango | Mejor es |
|---------|---------|-------|----------|
| Accuracy | Correctas / Total | [0, 1] | Mayor |
| Precision | TP / (TP + FP) | [0, 1] | Mayor |
| Recall | TP / (TP + FN) | [0, 1] | Mayor |
| F1-Score | 2PR / (P + R) | [0, 1] | Mayor |
| Cohen's Kappa | (p_o - p_e) / (1 - p_e) | [-1, 1] | Mayor (>0.8 excelente) |
| CV Std | σ de k-fold accuracies | [0, ∞) | Menor |

---

## Salidas y Graficas

### Estructura de Directorios

```
./img/metric/MCW/
├── QPSO_MCW_confusion_matrix_*.png
├── QPSO_MCW_loss_curves_*.png
├── QPSO_MCW_accuracy_curves_*.png
├── QPSO_MCW_training_summary_*.png
├── QPSO_MCW_cv_summary_*.png
├── QDPSO_MCW_confusion_matrix_*.png
├── QDPSO_MCW_loss_curves_*.png
├── QDPSO_MCW_accuracy_curves_*.png
├── QDPSO_MCW_training_summary_*.png
└── QDPSO_MCW_cv_summary_*.png
```

### Nomenclatura de Archivos

```
{OPTIMIZER}_MCW_{tipo}_{reduccion}_c{componentes}_{params}_{particles}_{iters}_{timestamp}.png
```

**Ejemplo**:
```
QPSO_MCW_confusion_matrix_isomap_c7_alpha_1.0-0.5_p50_i1000_20260111_175153.png
```

| Componente | Significado |
|------------|-------------|
| `QPSO` | Optimizador usado |
| `MCW` | Dataset |
| `confusion_matrix` | Tipo de grafica |
| `isomap_c7` | Reduccion con 7 componentes |
| `alpha_1.0-0.5` | Parametros QPSO |
| `p50` | 50 particulas |
| `i1000` | 1000 iteraciones |
| `20260111_175153` | Timestamp |

### Tipos de Graficas

#### 1. Matriz de Confusion (`confusion_matrix`)

Heatmap con:
- Filas: Clases reales
- Columnas: Predicciones
- Colores: Intensidad proporcional al conteo
- Anotaciones: Numero de muestras

#### 2. Curvas de Loss (`loss_curves`)

Grafico de lineas con:
- Eje X: Iteraciones
- Eje Y: Loss (Cross-Entropy)
- Lineas: Train (azul), Validation (naranja)

#### 3. Curvas de Accuracy (`accuracy_curves`)

Grafico de lineas con:
- Eje X: Iteraciones
- Eje Y: Accuracy [0, 1]
- Lineas: Train, Validation

#### 4. Resumen de Entrenamiento (`training_summary`)

Panel 2x2 con:
- Superior izquierda: Curva de loss
- Superior derecha: Curva de accuracy
- Inferior izquierda: Barras de loss final (train/val/test)
- Inferior derecha: Barras de accuracy final (train/val/test)

#### 5. Resumen Cross-Validation (`cv_summary`)

Grafico de barras con:
- Accuracy por fold
- Linea horizontal: Media
- Banda: Desviacion estandar

### Salida de Consola

```
======================================================================
 RESUMEN FINAL - COMPARATIVA QPSO vs QDPSO
======================================================================

Dataset: MCW (Multi-Class Weather)
Reduccion: isomap
Componentes: 7 (de 84 originales)
Fecha: 2026-01-11 17:53:23
Dispositivo: cuda
Tiempo total: 146.98s

----------------------------------------------------------------------------------------------------
Optimizador  Params       Test Acc     CV Acc               F1         Kappa      Time
----------------------------------------------------------------------------------------------------
QPSO         536          0.8756       0.8162 +/- 0.0446    0.8705     0.8327     57.62s
QDPSO        536          0.8667       0.8497 +/- 0.0094    0.8652     0.8206     86.04s
----------------------------------------------------------------------------------------------------

Mejor optimizador: QPSO con 0.8756 accuracy

Metricas por clase (QPSO):
  cloudy    : Precision=0.8519, Recall=0.7667, F1=0.8070
  rain      : Precision=0.8571, Recall=0.8372, F1=0.8471
  shine     : Precision=0.8065, Recall=1.0000, F1=0.8929
  sunrise   : Precision=0.9559, Recall=0.9028, F1=0.9286
```

---

## Ejemplo de Ejecucion

### Output Completo Tipico

```bash
$ python ann/main_mcw.py

======================================================================
 INFORMACION DEL SISTEMA
======================================================================
PyTorch version: 2.5.1
CUDA disponible: True
GPU: NVIDIA GeForce GTX 1050 Ti
CUDA version: 12.4
Dispositivo a usar: cuda

======================================================================
 CARGANDO DATASET MCW
======================================================================

1. Extrayendo features de imagenes...
Procesando cloudy: 300 imagenes
Procesando rain: 214 imagenes
Procesando shine: 252 imagenes
Procesando sunrise: 357 imagenes

   Total imagenes: 1123
   Features originales: 84

2. Dividiendo dataset...
   Train: 785 (70%)
   Val: 113 (10%)
   Test: 225 (20%)

3. Normalizando features por tipo...

4. Aplicando reduccion de dimensionalidad...
   Metodo: isomap
   Componentes: 7
   Shape final: 7 features

======================================================================
 BENCHMARK QPSO: MCW
======================================================================

--- 4. Configurando optimizador QPSO ---
  Alpha: (1.0, 0.5)
  Particulas: 50
  Iteraciones max: 1000

--- 5. Entrenando modelo con QPSO ---
Iter    0: loss=1.265888, acc=0.4535 | val_loss=1.307530, val_acc=0.4159
Iter  100: loss=0.981905, acc=0.8013 | val_loss=1.016994, val_acc=0.7434
...
Iter  980: loss=0.824820, acc=0.9287 | val_loss=0.913951, val_acc=0.8319

Entrenamiento completado en 18.18s
  Mejor loss: 0.824820

--- 6. Evaluando modelo ---
  Train - Loss: 0.824820, Acc: 0.9287
  Val   - Loss: 0.913951, Acc: 0.8319
  Test  - Loss: 0.869870, Acc: 0.8756

--- 7. Metricas detalladas (Test) ---
  Accuracy: 0.8756
  Precision (macro): 0.8714
  Recall (macro): 0.8767
  F1-Score (macro): 0.8705
  Cohen's Kappa: 0.8327

[... Similar para QDPSO ...]

======================================================================
 BENCHMARK MCW COMPLETADO
======================================================================
```

---

## Dependencias

### Requerimientos Python

```
torch>=1.9.0
numpy>=1.19.0
opencv-python>=4.5.0
mahotas>=1.4.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

### Instalacion

```bash
# Activar entorno
conda activate pytorch_qpso_gpu

# Instalar dependencias especificas MCW
pip install opencv-python mahotas

# Verificar instalacion
python -c "import cv2; import mahotas; print('OK')"
```

---

## Notas y Consideraciones

### 1. Normalizacion por Tipo de Feature

Las 84 features se normalizan por separado segun su tipo:
- StandardScaler para histogram (64)
- StandardScaler para haralick (13)
- StandardScaler para hu_moments (7)

Esto evita que features de diferente escala dominen unas sobre otras.

### 2. Limitacion de MDS

MDS no soporta `transform()`, por lo que:
- Se ajusta un modelo separado para train, val y test
- Puede causar inconsistencias entre conjuntos
- **Recomendacion**: Usar PCA o Isomap en su lugar

### 3. Isomap n_neighbors

Se usa `n_neighbors = min(100, n_samples - 1)` para:
- Evitar errores con datasets pequeños
- Mantener conectividad del grafo

### 4. GPU Acceleration

El entrenamiento QPSO/QDPSO utiliza GPU si esta disponible:
- Las operaciones tensoriales se ejecutan en CUDA
- La extraccion de features (OpenCV, mahotas) es CPU

### 5. Reproducibilidad

Para resultados reproducibles, `random_state=42` se aplica a:
- Split de datos (train/val/test)
- Reduccion de dimensionalidad
- Inicializacion de particulas QPSO

### 6. Early Stopping

Con `patience=50`, el entrenamiento para si:
- No hay mejora en loss por 50 iteraciones
- La mejora es menor a 1e-12

### 7. Interpretacion de Resultados MCW

Resultados tipicos esperados:
- **Accuracy**: 0.80 - 0.90
- **Cohen's Kappa**: 0.75 - 0.85
- Clase `shine` suele tener mejor recall (cielo azul distintivo)
- Clases `cloudy` y `rain` pueden confundirse

---

## Referencias

1. **QPSO**: Sun, J., Feng, B., & Xu, W. (2004). *Particle swarm optimization with particles having quantum behavior*.

2. **Haralick Features**: Haralick, R.M., Shanmugam, K., & Dinstein, I. (1973). *Textural features for image classification*. IEEE Transactions on Systems, Man, and Cybernetics.

3. **Hu Moments**: Hu, M.K. (1962). *Visual pattern recognition by moment invariants*. IRE Transactions on Information Theory.

4. **Isomap**: Tenenbaum, J.B., de Silva, V., & Langford, J.C. (2000). *A global geometric framework for nonlinear dimensionality reduction*. Science.

5. **Cohen's Kappa**: Cohen, J. (1960). *A coefficient of agreement for nominal scales*. Educational and Psychological Measurement.
