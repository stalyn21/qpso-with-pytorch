# Origen de los nombres de las técnicas

> **Audiencia:** alguien que ve por primera vez `T1`, `T2`, `T2-jacobi`, `T3`, `T4` y quiere entender **por qué se llaman así** y de qué literatura clásica vienen los términos *Gauss-Seidel*, *Jacobi* y *Hogwild*.

Todos estos nombres son **etiquetas estándar** del análisis numérico y la optimización paralela. Aquí solo los aplicamos al esquema "una capa de la red = un bloque de variables".

---

## 1. Punto de partida: ¿qué optimizamos?

En todas las técnicas dividimos los pesos de la MLP en **bloques por capa**:

```
W = [ W_1 | W_2 | ... | W_L ]
        ↑     ↑           ↑
     capa 1 capa 2 ...  capa L
```

Cada capa tiene su **propio enjambre QDPSO**. La pregunta diseño es:

> **¿qué valores de las otras capas ve cada enjambre mientras optimiza la suya?**

La respuesta define el método.

---

## 2. Gauss-Seidel y Jacobi — métodos clásicos (s. XIX)

Ambos nombres vienen de los **métodos iterativos para resolver sistemas `Ax = b`** y, por extensión, de las reglas estándar de **Block Coordinate Descent (BCD)** en optimización.

### Jacobi (Carl Jacobi, ~1845)

> "Actualizo **todas** las variables usando la **foto anterior**, después intercambio."

```
x_1^(t+1) = f( x_1^(t), x_2^(t), x_3^(t) )    ← usa valores de t
x_2^(t+1) = f( x_1^(t), x_2^(t), x_3^(t) )    ← usa valores de t (NO el x_1 nuevo)
x_3^(t+1) = f( x_1^(t), x_2^(t), x_3^(t) )    ← usa valores de t
```

- **Todos leen del mismo snapshot.**
- **Paralelizable trivialmente** (las 3 actualizaciones son independientes dentro de la iteración).
- **Determinista**, pero suele necesitar más iteraciones globales.

### Gauss-Seidel (Gauss + Seidel, ~1874)

> "Actualizo en orden, y cada variable **ya ve** los cambios de las anteriores en esta misma iteración."

```
x_1^(t+1) = f( x_1^(t),   x_2^(t),   x_3^(t) )   ← usa t
x_2^(t+1) = f( x_1^(t+1), x_2^(t),   x_3^(t) )   ← YA ve x_1 nuevo
x_3^(t+1) = f( x_1^(t+1), x_2^(t+1), x_3^(t) )   ← YA ve x_1 y x_2 nuevos
```

- **Propagación inmediata** dentro de la iteración.
- Suele converger más rápido que Jacobi.
- **Inherentemente secuencial** (cada paso depende del anterior).

---

## 3. Cómo se traduce al multi-swarm MLP

En nuestro caso, **cada "variable" es una capa completa de pesos**, y "actualizar" significa "correr un QDPSO sobre esa capa". La elección de **qué ve cada enjambre** define la técnica.

### T1 = BCD **Gauss-Seidel** (granularidad capa)

```
[ QDPSO capa 1: 500 iters ]
        └─→ produce W_1_new
              ↓
[ QDPSO capa 2: 500 iters, usando W_1_new ]
        └─→ produce W_2_new
              ↓
[ QDPSO capa 3: 500 iters, usando W_1_new y W_2_new ]
```

- Propagación inmediata **a nivel de capa**: la siguiente capa **ya ve** el resultado final de la anterior.
- Es **BCD Gauss-Seidel** clásico, aplicado a bloques = capas.

### T2-jacobi = BCD **Jacobi**

```
foto = ( W_1^(t), W_2^(t), W_3^(t) )           ← snapshot

en paralelo:
   QDPSO capa 1 contra (foto)  → W_1^(t+1)
   QDPSO capa 2 contra (foto)  → W_2^(t+1)
   QDPSO capa 3 contra (foto)  → W_3^(t+1)

sincronizar y repetir round
```

- Todos los enjambres leen del **mismo snapshot** anterior.
- **Determinista, paralelizable**.
- Converge más lento → por eso necesita varios rounds (`--n-rounds 5`).

---

## 4. Hogwild — concepto distinto (paper de 2011)

**Hogwild!** viene del paper:

> Niu, Recht, Ré, Wright.
> *"HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent."*
> NeurIPS 2011.

La idea radical:

> "Varios threads escriben sobre el **mismo vector de parámetros** **sin candados** (lock-free).
> Las colisiones (race conditions) ocurren pero son raras o el ruido es benigno, así que simplemente las ignoras."

Es decir: se **rompe la pureza** Jacobi/Gauss-Seidel a propósito, porque el costo de sincronizar > beneficio de un orden estricto.

### T2 (async) = **Hogwild-style**

```
threads concurrentes, sin locks, sobre el mismo `current`:

  thread 1: QDPSO capa 1 ─┐
  thread 2: QDPSO capa 2 ─┼─→ leen/escriben en `current` (vector compartido)
  thread 3: QDPSO capa 3 ─┘
                          sin barreras, sin snapshot, sin orden garantizado
```

- **Lock-free**, asíncrono, race conditions toleradas.
- No es Jacobi puro (no hay snapshot) ni Gauss-Seidel puro (no hay orden).
- "Quién ve qué" depende del scheduler del SO.

---

## 5. T3 — "Gauss-Seidel iter-por-iter" (lockstep)

Aquí el matiz crítico es **la granularidad** del Gauss-Seidel.

- **T1** hace GS a nivel **"toda la optimización de la capa"**: termina el QDPSO de la capa 1 entero (ej. 500 iters), y **recién entonces** la capa 2 lo ve.
- **T3** hace GS a nivel **"una iteración del QDPSO"**: la capa 1 hace **UNA** iter de QDPSO → la capa 2 hace UNA iter viéndola → la capa 3 hace UNA iter viéndolas → **sincronizan** → repiten.

```
T1 (GS por capa):
   [ capa1: 500 iters ] → [ capa2: 500 iters viendo capa1 final ] → ...

T3 (GS por iter, lockstep):
   for t in 1..500:
       capa1: 1 iter
       capa2: 1 iter (viendo el resultado de capa1 en este t)
       capa3: 1 iter (viendo capa1 y capa2 en este t)
       sincronizar
```

- Misma **regla GS** (propagación inmediata, orden de capas) que T1.
- Pero **acoplado en cada paso del QDPSO**, no esperando a que la capa termine entera.
- En la práctica suele dar **mejor accuracy** porque las capas **co-evolucionan** sin "fosilizarse" como en T1.

---

## 6. T4 — Single swarm (no aplica BCD)

T4 **no es** un método BCD: vuelve a un **solo enjambre monolítico** sobre todos los pesos `W = [W_1 | ... | W_L]` concatenados.

Lo que lo hace competitivo es el **warm-start cluster**: K partículas centradas alrededor de una buena solución previa + el resto aleatorias.

No tiene etiqueta Gauss-Seidel/Jacobi/Hogwild porque no hay descomposición por bloques.

---

## 7. Tabla resumen

| Técnica | Etiqueta clásica | Qué ve cada enjambre | Sincronización | Paralelo |
|---|---|---|---|---|
| **T1** | BCD **Gauss-Seidel** (granularidad capa) | Valores **nuevos** de capas anteriores | Al terminar QDPSO entero de cada capa | No |
| **T2 async** | **Hogwild** (Niu et al. 2011) | Lo que esté en memoria (race) | **Ninguna** (lock-free) | Sí, ~1.3× |
| **T2-jacobi** | BCD **Jacobi** | Snapshot **viejo** de todas las capas | Al final de cada round | Sí, determinista |
| **T3** | BCD **Gauss-Seidel iter-por-iter** | Valores **nuevos** de capas anteriores | Cada **iteración** del QDPSO | Sí, lockstep |
| **T4** | (single swarm) | Todos los pesos juntos | — | — |

---

## 8. Cómo explicar cada técnica con analogías (presentación informal)

> **Idea común a todas:** en lugar de un solo enjambre QDPSO que optimiza todos los pesos de golpe, **partimos la red por capas** y le damos a cada capa **su propio enjambre**. Lo que cambia entre técnicas es **quién espera a quién**, **quién ve los cambios del otro** y **cuándo**.
>
> **Analogía base:** imagina varios equipos trabajando sobre un mismo edificio (la red). Cada técnica define cómo se coordinan.

---

### T1 — Secuencial (Gauss-Seidel a nivel capa)

**Idea:** entrena la **capa 1**, congela el resultado; luego entrena la **capa 2** viendo ya el cambio anterior; y así.

- **Analogía:** el equipo A pinta la pared, **termina**, y le pasa la llave al equipo B, que pinta la siguiente viendo lo que hizo A.
- **Ventaja:** propagación inmediata de información, muy estable.
- **Costo:** no aprovecha paralelismo, es la **línea base** secuencial.
- **Frase clave:** *"Block Coordinate Descent clásico, una capa a la vez."*

---

### T2 — Asíncrono lock-free (Hogwild-style)

**Idea:** los enjambres de **todas las capas corren al mismo tiempo**, leyendo y escribiendo sobre un mismo vector de pesos compartido **sin candados**.

- **Analogía:** **todos los equipos pintan a la vez** sobre el mismo muro. A veces se pisan, pero el avance global es más rápido.
- **Ventaja:** paralelismo **real**, ~1.3× speedup vs T1.
- **Costo:** sin garantía de orden → más varianza, sensible a cold-start.
- **Frase clave:** *"T1 paralelizada estilo Hogwild, sin barreras."*

---

### T2-jacobi — Paralelo determinista (Jacobi + warm-start)

**Idea:** todos los enjambres corren a la vez **pero con una foto fija** del estado, luego **sincronizan** todos juntos al final de cada round.

- **Analogía:** cada equipo trabaja con el **plano de ayer**; al final del día se reúnen y actualizan el plano único para mañana.
- **Ventaja:** **determinista** y reproducible.
- **Costo:** **4–5× más lento** (varios rounds); colapsa en redes pequeñas (`d_total < 50`, ej. circle).
- **Frase clave:** *"Jacobi paralelo con warm-start vía `reuse_with`."*

---

### T3 — Concurrente lockstep (Gauss-Seidel iter-por-iter)

**Idea:** los enjambres avanzan **una iteración cada uno, sincronizan, otra iteración, sincronizan…** Es Gauss-Seidel **pero a nivel de iteración**, no de capa.

- **Analogía:** los equipos pintan **una brochada cada uno por turno**, y todos ven el avance de los demás antes de la siguiente brochada.
- **Ventaja:** **mejor accuracy promedio** (campeón en los 4 datasets en el benchmark CV con 1 seed).
- **Costo:** sincronización fina → un poco más lento que T2 async.
- **Frase clave:** *"Concurrencia Gauss-Seidel iter-por-iter con warm-start cluster."*

---

### T4 — Single swarm con warm-start cluster

**Idea:** vuelve a **un solo enjambre monolítico** que optimiza todos los pesos juntos, pero **inicializado de forma inteligente**: K partículas cerca de una buena solución previa + el resto aleatorias.

- **Analogía:** **un solo equipo grande**, pero al que le das **el plano del ganador anterior** para que arranque casi terminado.
- **Ventaja:** **el más rápido** y simple conceptualmente.
- **Costo:** menos exploración por capa, pierde la modularidad multi-swarm.
- **Frase clave:** *"Baseline single-swarm potenciado con warm-start cluster."*

---

### Orden recomendado para presentar (guion para charla / reunión)

1. **Motivación.** "Una MLP grande es difícil para un solo QDPSO → partamos por capas."
2. **T1 primero** (base secuencial, fácil de entender).
3. **T2 y T2-jacobi** como **dos formas de paralelizar T1**:
   - una **rápida pero ruidosa** (async, Hogwild),
   - otra **determinista pero costosa** (jacobi).
4. **T3** como **híbrido**: paralelismo coordinado iter-por-iter → mejor accuracy.
5. **T4** como **contraste**: ¿realmente necesitamos multi-swarm? El single-swarm con warm-start es un **baseline duro**.
6. **Cierre.** Trade-off **accuracy ↔ tiempo ↔ reproducibilidad** → la figura **Pareto** lo resume visualmente.

---

## 9. Pitch de 60 segundos

1. **Premisa:** en lugar de un solo QDPSO para toda la red, **un enjambre por capa**.
2. **La pregunta es:** ¿cómo coordinas los enjambres?
3. **Dos extremos clásicos** del análisis numérico:
   - **Gauss-Seidel** → secuencial, ves los cambios al instante. → **T1**
   - **Jacobi** → paralelo, todos leen la foto anterior. → **T2-jacobi**
4. **Una alternativa moderna (Hogwild, 2011):** rompe la sincronización a propósito por velocidad. → **T2 async**
5. **Un híbrido fino:** Gauss-Seidel pero a nivel de **iteración**, no de capa entera. → **T3**
6. **Un baseline duro:** olvídate de los bloques, un solo enjambre con buen warm-start. → **T4**

El trade-off central que estamos midiendo: **accuracy ↔ tiempo ↔ reproducibilidad**.

---

## 10. Referencias para el paper

- **BCD / Gauss-Seidel / Jacobi:**
  - Bertsekas, D. *Nonlinear Programming.* Athena Scientific, 1999. Cap. 2.7.
  - Wright, S. J. *"Coordinate descent algorithms."* Mathematical Programming, 2015.
- **Hogwild:**
  - Niu, F.; Recht, B.; Ré, C.; Wright, S. J. *"HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent."* NeurIPS 2011.
- **QDPSO base:**
  - Sun, J.; Feng, B.; Xu, W. *"Particle swarm optimization with particles having quantum behavior."* CEC 2004.

---

## 11. Ver también

- [`README.md`](./README.md) — docs principales del re-design.
- [`re-design.md`](../../../MEMORY/nn_qdpso/re-design.md) — overview completo (memoria del proyecto).
- [`bcd-warmstart.md`](./bcd-warmstart.md) — detalle técnico del cambio BCD + warm-start (2026-04-26).
- [`compare.md`](./compare.md) — tablas de resultados actuales.
