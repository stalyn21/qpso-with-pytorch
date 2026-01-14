# Documentacion Completa: QPSO y QDPSO

[🏠 README](../../README.md) | [📚 Index](index.md) | **NumPy Reference** | [Next: Tensor ➡️](docs_qpso_tensor.md)

---

## Nota Importante sobre esta Implementacion

> **La version original del paquete PyPI `tensor_qpso` solo incluia la variante QDPSO.**
>
> Esta version extendida incluye:
> - **QPSO**: Implementacion original del paper de Sun et al. (2004)
> - **QDPSO**: Variante Delta (implementacion original de PyPI)

---

## Tabla de Contenidos

1. [Introduccion](#introduccion)
2. [Diferencias Clave: QPSO vs QDPSO](#diferencias-clave-qpso-vs-qdpso)
3. [Arquitectura de Clases](#arquitectura-de-clases)
4. [QPSO Original (Paper)](#qpso-original-paper)
5. [QDPSO (Variante Delta)](#qdpso-variante-delta)
6. [Comparativa de Formulas](#comparativa-de-formulas)
7. [Parametros y Configuracion](#parametros-y-configuracion)
8. [Ejemplos de Uso](#ejemplos-de-uso)
9. [Guia de Seleccion](#guia-de-seleccion)
10. [Referencias](#referencias)

---

## Introduccion

**QPSO (Quantum Particle Swarm Optimization)** es una metaheuristica de optimizacion propuesta por Sun, Feng y Xu en 2004. Combina conceptos de mecanica cuantica con la optimizacion por enjambre de particulas (PSO) para mejorar la capacidad de exploracion global.

A diferencia del PSO clasico que usa posicion y velocidad, en QPSO las particulas se modelan como particulas cuanticas sin velocidad determinista, donde su posicion sigue una distribucion de probabilidad.

---

## Diferencias Clave: QPSO vs QDPSO

| Aspecto | QPSO (Original Paper) | QDPSO (Variante PyPI) |
|---------|----------------------|----------------------|
| **Origen** | Paper de Sun et al., 2004 | Variante de implementacion |
| **Punto atractor** | `c = phi*pbest + (1-phi)*gbest` | `c = (u1*pbest + u2*gbest)/(u1+u2)` |
| **Calculo de L** | `L = alpha * \|mbest - x\|` | `L = (1/g) * \|x - c\|` |
| **Usa mbest** | Si (caracteristica distintiva) | No |
| **Parametro** | `alpha` (0.5 - 1.0) | `g` (~0.96) |
| **Alpha adaptativo** | Si (decrecimiento lineal) | No |

### Diferencia Conceptual Principal

- **QPSO**: La longitud caracteristica `L` depende de la distancia entre la particula y el **centro del enjambre** (mbest = promedio de todos los pbest). Esto promueve convergencia hacia el centro colectivo.

- **QDPSO**: La longitud caracteristica `L` depende de la distancia entre la particula y su **punto atractor individual** (c). Esto da mas independencia a cada particula.

---

## Arquitectura de Clases

```
Particle           <- Particula individual
    |
    v
Swarm              <- Coleccion de particulas + gbest
    |
    v
QPSOBase           <- Clase base abstracta (logica comun)
    |
    +-------+-------+
    |               |
    v               v
  QPSO           QDPSO
(Original)     (Variante)
```

### Descripcion de Clases

| Clase | Descripcion |
|-------|-------------|
| `Particle` | Representa una particula con posicion `x`, mejor personal `pbest` y su valor |
| `Swarm` | Gestiona el enjambre, calcula `mbest` y `gbest` |
| `QPSOBase` | Clase base abstracta con `init_eval()`, `update_best()`, `update()` |
| `QPSO` | Implementacion original del paper usando `mbest` |
| `QDPSO` | Variante Delta usando `\|x - c\|` |

---

## QPSO Original (Paper)

### Formula Matematica

```
1. mbest = (1/N) * SUM(pbest_j) para j = 1..N

2. phi ~ U(0, 1)

3. c = phi * pbest + (1 - phi) * gbest

4. alpha = alpha_max - (alpha_max - alpha_min) * t / T
   (o valor fijo)

5. L = alpha * |mbest - x|

6. x_nuevo = c +/- L * ln(1/u), donde u ~ U(0,1)
```

### Significado de Cada Termino

| Termino | Significado |
|---------|-------------|
| `mbest` | Mean Best: promedio de todas las mejores posiciones personales |
| `phi` | Peso aleatorio para combinacion convexa |
| `c` | Punto atractor local (entre pbest y gbest) |
| `alpha` | Coeficiente de contraccion-expansion |
| `L` | Longitud caracteristica (rango de exploracion) |
| `ln(1/u)` | Genera distribucion exponencial |
| `+/-` | Signo aleatorio (50% cada uno) |

### Parametro Alpha

El parametro `alpha` controla el balance exploracion/explotacion:

| Configuracion | Descripcion | Uso |
|--------------|-------------|-----|
| `alpha = 0.75` | Valor fijo balanceado | Simple, robusto |
| `alpha = (1.0, 0.5)` | Decrecimiento lineal | Recomendado en paper |
| `alpha > 1.0` | Mas exploracion | Funciones multimodales |
| `alpha < 0.5` | Mas explotacion | Refinamiento final |

### Decrecimiento Lineal de Alpha: Variables `t` y `T`

Cuando se usa alpha con decrecimiento lineal, la formula es:

```
alpha(t) = alpha_max - (alpha_max - alpha_min) * (t / T)
```

#### Significado de las Variables

| Variable | Significado | Origen en el Codigo |
|----------|-------------|---------------------|
| `t` | Iteracion actual | `self._iters` - contador interno que incrementa en cada iteracion |
| `T` | Iteraciones maximas | `self._maxIters` - parametro del constructor |
| `alpha_max` | Valor inicial de alpha | Primer elemento de la tupla, ej: `1.0` en `(1.0, 0.5)` |
| `alpha_min` | Valor final de alpha | Segundo elemento de la tupla, ej: `0.5` en `(1.0, 0.5)` |

#### Comportamiento del Decrecimiento

- **Cuando `t = 0`** (inicio): `alpha = alpha_max` → Mayor exploracion
- **Cuando `t = T`** (final): `alpha = alpha_min` → Mayor explotacion

#### Ejemplo Visual

Con `alpha = (1.0, 0.5)` y `maxIters = 1000`:

```
Iteracion (t)    t/T      alpha(t)    Comportamiento
-------------  ------    ----------  ----------------
      0         0.00       1.00      Exploracion maxima
    200         0.20       0.90      |
    400         0.40       0.80      |  Transicion
    600         0.60       0.70      |  gradual
    800         0.80       0.60      v
   1000         1.00       0.50      Explotacion maxima
```

#### Razon del Decrecimiento (segun el paper)

El paper de Sun et al. recomienda este comportamiento porque:

1. **Fase inicial** (`alpha` alto):
   - Las particulas exploran ampliamente el espacio de busqueda
   - Mayor probabilidad de encontrar la region del optimo global
   - Evita quedar atrapado en optimos locales

2. **Fase final** (`alpha` bajo):
   - Las particulas refinan la solucion cerca del optimo encontrado
   - Movimientos mas pequenos para mayor precision
   - Convergencia hacia la solucion final

Este esquema balancea **exploracion global** con **explotacion local**.

#### Implementacion en el Codigo

```python
def _get_alpha(self):
    """Calcula alpha actual (fijo o con decrecimiento lineal)."""
    if isinstance(self._alpha, tuple):
        t = self._iters      # Iteracion actual
        T = self._maxIters   # Maximo de iteraciones
        return self._alpha_max - (self._alpha_max - self._alpha_min) * t / T
    return self._alpha  # Valor fijo
```

### Codigo de Implementacion

```python
def kernel_update(self, **kwargs):
    mbest = self.mean_best()  # Calcula promedio de pbest
    alpha = self._get_alpha()  # Obtiene alpha actual

    for p in self._particles:
        for i in range(0, self._dim):
            phi = random.uniform(0., 1.)
            u = random.uniform(0., 1.)
            rand_sign = 1 if random.random() > 0.5 else -1

            # Punto atractor: combinacion convexa
            c = phi * p.best[i] + (1 - phi) * self._gbest[i]

            # L usa mbest (diferencia clave)
            L = alpha * abs(mbest[i] - p[i])

            # Nueva posicion
            p[i] = c + rand_sign * L * np.log(1. / u)
```

---

## QDPSO (Variante Delta)

### Formula Matematica

```
1. u1, u2, u3 ~ U(0, 1)

2. c = (u1 * pbest + u2 * gbest) / (u1 + u2)

3. L = (1/g) * |x - c|

4. x_nuevo = c +/- L * ln(1/u3)
```

### Significado de Cada Termino

| Termino | Significado |
|---------|-------------|
| `u1, u2` | Pesos aleatorios para el punto atractor |
| `c` | Punto atractor (promedio ponderado estocastico) |
| `g` | Coeficiente de contraccion (~0.96) |
| `L` | Longitud caracteristica basada en distancia a `c` |

### Parametro g

| Valor de g | Efecto |
|------------|--------|
| `g < 1.0` | Mayor exploracion (L mas grande) |
| `g = 0.96` | Valor tipico balanceado |
| `g > 1.0` | Mayor explotacion (L mas pequeno) |

### Codigo de Implementacion

```python
def kernel_update(self, **kwargs):
    for p in self._particles:
        for i in range(0, self._dim):
            u1 = random.uniform(0., 1.)
            u2 = random.uniform(0., 1.)
            u3 = random.uniform(0., 1.)
            rand_sign = 1 if random.random() > 0.5 else -1

            # Punto atractor: promedio ponderado
            c = (u1 * p.best[i] + u2 * self._gbest[i]) / (u1 + u2)

            # L usa distancia a punto atractor (no mbest)
            L = (1 / self._g) * abs(p[i] - c)

            # Nueva posicion
            p[i] = c + rand_sign * L * np.log(1. / u3)
```

---

## Comparativa de Formulas

### Punto Atractor (c)

| QPSO | QDPSO |
|------|-------|
| `c = phi * pbest + (1-phi) * gbest` | `c = (u1*pbest + u2*gbest) / (u1+u2)` |
| Combinacion convexa | Promedio ponderado |
| Siempre entre pbest y gbest | Siempre entre pbest y gbest |
| Un solo numero aleatorio (phi) | Dos numeros aleatorios (u1, u2) |

### Longitud Caracteristica (L)

| QPSO | QDPSO |
|------|-------|
| `L = alpha * \|mbest - x\|` | `L = (1/g) * \|x - c\|` |
| Depende del centro colectivo | Depende del atractor individual |
| Particulas influenciadas por el enjambre | Particulas mas independientes |
| Converge hacia mbest | Converge hacia c individual |

### Diagrama Visual

```
QPSO:
                    mbest (centro del enjambre)
                       *
                      /|\
                     / | \
                    /  |  \
         particula x   |   L = alpha * |mbest - x|
                   *---+
                       c (atractor local)

QDPSO:
         particula x
                   *
                   |
                   | L = (1/g) * |x - c|
                   |
                   *
                   c (atractor local)
```

---

## Parametros y Configuracion

### Parametros QPSO

| Parametro | Tipo | Descripcion | Valor Recomendado |
|-----------|------|-------------|-------------------|
| `cf` | callable | Funcion de costo | - |
| `size` | int | Numero de particulas | 20-50 |
| `dim` | int | Dimensionalidad | Depende del problema |
| `bounds` | list[tuple] | Limites [(min, max), ...] | Depende del problema |
| `maxIters` | int | Iteraciones maximas | 500-2000 |
| `alpha` | float o tuple | Coeficiente alpha | 0.75 o (1.0, 0.5) |

### Parametros QDPSO

| Parametro | Tipo | Descripcion | Valor Recomendado |
|-----------|------|-------------|-------------------|
| `cf` | callable | Funcion de costo | - |
| `size` | int | Numero de particulas | 20-50 |
| `dim` | int | Dimensionalidad | Depende del problema |
| `bounds` | list[tuple] | Limites [(min, max), ...] | Depende del problema |
| `maxIters` | int | Iteraciones maximas | 500-2000 |
| `g` | float | Coeficiente g | 0.96 |

---

## Ejemplos de Uso

### QPSO Original

```python
import numpy as np
from tensor_qpso.qpso import QPSO

# Funcion objetivo
def sphere(args):
    return sum([x**2 for x in args])

# Configuracion
NParticle = 40
MaxIters = 1000
NDim = 10
bounds = [(-5.12, 5.12) for _ in range(NDim)]

# Opcion 1: Alpha fijo
qpso = QPSO(sphere, NParticle, NDim, bounds, MaxIters, alpha=0.75)
qpso.update()

# Opcion 2: Alpha con decrecimiento lineal (recomendado)
qpso = QPSO(sphere, NParticle, NDim, bounds, MaxIters, alpha=(1.0, 0.5))
qpso.update()

print(f"Mejor valor: {qpso.gbest_value}")
print(f"Mejor posicion: {qpso.gbest}")
```

### QDPSO (Variante)

```python
import numpy as np
from tensor_qpso.qpso import QDPSO

# Funcion objetivo
def sphere(args):
    return sum([x**2 for x in args])

# Configuracion
NParticle = 40
MaxIters = 1000
NDim = 10
bounds = [(-5.12, 5.12) for _ in range(NDim)]
g = 0.96

# Crear y ejecutar
qdpso = QDPSO(sphere, NParticle, NDim, bounds, MaxIters, g=g)
qdpso.update()

print(f"Mejor valor: {qdpso.gbest_value}")
print(f"Mejor posicion: {qdpso.gbest}")
```

### Con Callback para Monitoreo

```python
def log(optimizer):
    print(f"Iter {optimizer.iters}: Best = {optimizer.gbest_value:.6E}")

# Ejecutar con callback cada 100 iteraciones
qpso.update(callback=log, interval=100)
```

---

## Guia de Seleccion

### Cuando usar QPSO (Original)

- Problemas donde el centro del enjambre es informativo
- Funciones unimodales o con pocos optimos locales
- Cuando se quiere convergencia coordinada del enjambre
- Para comparacion con literatura cientifica

### Cuando usar QDPSO (Variante)

- Problemas altamente multimodales
- Cuando se necesita mayor diversidad de busqueda
- Particulas explorando independientemente
- Mantener compatibilidad con codigo existente de PyPI

### Recomendacion General

Para **problemas nuevos**, probar ambos algoritmos con sus parametros recomendados:

```python
# QPSO con alpha decreciente
qpso = QPSO(cf, size, dim, bounds, maxIters, alpha=(1.0, 0.5))

# QDPSO con g estandar
qdpso = QDPSO(cf, size, dim, bounds, maxIters, g=0.96)
```

Ejecutar multiples veces (10-30 ejecuciones) y comparar:
- Mejor valor encontrado (promedio y desviacion)
- Tasa de convergencia
- Robustez entre ejecuciones

---

## Flujo de Ejecucion

```
INICIALIZACION
     |
     +-- Crear particulas con posiciones aleatorias
     +-- Evaluar fitness inicial
     +-- Establecer pbest = posicion_inicial
     +-- Determinar gbest
     |
LOOP PRINCIPAL (t = 0 hasta maxIters)
     |
     +-- [QPSO] Calcular mbest = promedio(pbest)
     +-- [QPSO] Calcular alpha actual (si adaptativo)
     |
     +-- Para cada particula, cada dimension:
     |       |
     |       +-- Calcular punto atractor c
     |       +-- Calcular longitud L
     |       +-- Actualizar posicion
     |
     +-- Evaluar nuevas posiciones
     +-- Actualizar pbest si mejora
     +-- Actualizar gbest si mejora
     |
     +-- t = t + 1
     |
RESULTADO
     |
     +-- Retornar gbest y gbest_value
```

---

## Referencias

1. **Paper Original QPSO**:
   Sun, J., Feng, B., & Xu, W. (2004). "Particle swarm optimization with particles having quantum behavior". *Proceedings of the 2004 Congress on Evolutionary Computation*, vol. 1, pp. 325-331.

2. **Analisis de Parametros**:
   Sun, J., Fang, W., Wu, X., Palade, V., & Xu, W. (2012). "Quantum-behaved particle swarm optimization: Analysis of individual particle behavior and parameter selection". *Evolutionary Computation*, 20(3), pp. 349-393.

3. **Survey QPSO**:
   Sun, J., Xu, W., & Feng, B. (2004). "A global search strategy of quantum-behaved particle swarm optimization". *IEEE Conference on Cybernetics and Intelligent Systems*, vol. 1, pp. 111-116.

---

## Historial de Cambios

| Version | Cambio |
|---------|--------|
| Original (PyPI) | Solo QDPSO disponible |
| Actual | Agregado QPSO original del paper |
| Actual | Clase base QPSOBase para extensibilidad |
| Actual | Alpha adaptativo con decrecimiento lineal |
| Actual | Documentacion completa de ambas variantes |

---

## Related Documents

- [📊 Implementation Comparison](implementation_comparison.md) - Compare all three implementations
- [📗 Tensor Implementation](docs_qpso_tensor.md) - PyTorch tensor version
- [📙 Optimized Implementation](docs_qpso_tensor_optimized.md) - Production-ready version

---

<div align="center">

**[⬆️ Back to Top](#documentacion-completa-qpso-y-qdpso)** | **[📚 Index](index.md)** | **[Next: Tensor ➡️](docs_qpso_tensor.md)**

</div>
