# Re-design multi-swarm MLP (PyTorch GPU)

> **Última verificación:** 2026-05-05

Re-implementación limpia del subsistema multi-swarm de `nn_qdpso/`, sobre **PyTorch GPU** usando el QDPSO tensorizado de `tensor_qpso/`. Reemplaza la implementación NumPy original (`parallel_e.py`, `concurrencia_e.py`, etc.), que tenía fugas críticas de test y bugs metodológicos.

Documentación complementaria:
- [`technique-names.md`](./technique-names.md) — **por qué se llaman Gauss-Seidel, Jacobi, Hogwild** (origen de los nombres y mapeo a T1/T2/T2-jacobi/T3/T4)
- [`bcd-warmstart.md`](./bcd-warmstart.md) — detalle técnico del cambio BCD + warm-start (2026-04-26)
- `MEMORY/nn_qdpso/re-design.md` — overview completo (fuera del repo)
- `MEMORY/nn_qdpso/re-design-no-test-in-sessions.md` — por qué nunca seleccionar por test

---

## Estructura

```
re-design/
├── runner.py                          # CLI principal
├── compare.py                         # Comparador + plots de learning curves
├── command.sh                         # Script de corridas
├── bcd-warmstart.md                   # Detalle del cambio 2026-04-26
├── tensor_qpso/                       # QDPSO PyTorch (importado)
│   ├── qpso.py
│   ├── qpso_tensor.py
│   └── qpso_tensor_optimized.py
├── core/
│   ├── data.py                        # Carga + 70/15/15 estratificado
│   ├── mlp.py                         # MLP configurable + cost_fn vectorizada
│   ├── loss.py                        # CE estable + L2 real
│   └── metrics.py                     # acc, mse_int, confusion_matrix
├── techniques/
│   ├── base.py                        # Loop de sesiones + selección + persistencia
│   ├── warm_start.py                  # Helper de warm-start cluster (T3/T4)
│   ├── t1_sequential.py               # T1: BCD Gauss-Seidel a nivel de capa
│   ├── t2_async.py                    # T2: BCD asíncrono lock-free (Hogwild-style)
│   ├── t2_jacobi.py                   # T2-jacobi: Jacobi paralelo + warm-start QDPSO (variante)
│   ├── t3_concurrency.py              # T3: lockstep Gauss-Seidel iter-por-iter
│   └── t4_single_swarm.py             # T4: un swarm con warm-start cluster
└── output/                            # JSONs + npy + PNGs por corrida
```

---

## Las técnicas

| Tag | Nombre | Cómo corre los swarms | Contrapartes ven... |
|-----|--------|----------------------|---------------------|
| **T1** | Secuencial (BCD Gauss-Seidel a nivel de capa) | Una capa después de la otra (salida → entrada); `n_rounds` sweeps | `current` evolutivo: la capa siguiente ve la actualización inmediata de la anterior dentro del round |
| **T2** | BCD asíncrono lock-free (Hogwild-style) | Threads concurrentes (con CUDA streams), cada capa avanza independientemente sobre el `current` compartido sin barreras | `current` compartido leído ad-hoc en cada step: cada thread ve los updates más recientes de los demás (relajación asíncrona de GS) |
| **T2-jacobi** | Jacobi paralelo con warm-start (variante de T2) | Threads (con CUDA streams), todas a la vez contra el mismo snapshot; sync al final del round; `n_rounds` rounds | `frozen_for_round` del inicio del round; entre rounds las partículas del QDPSO MANTIENEN su estado (warm-start) |
| **T3** | Concurrencia (Gauss-Seidel iter-por-iter) | Lockstep: cada iter avanza todos los swarms en orden salida→entrada, intercambiando gBests | Counterparts dinámicos: cada swarm ve los gBests más recientes de los otros |
| **T4** | Single swarm | Un swarm sobre el vector plano completo de pesos | N/A (no hay descomposición); warm-start cluster con `params_init` |

### Notas semánticas

- **Back-to-front**: orden de entrenamiento de capas. Salida primero, entrada al final. **No es backprop** — es scheduling. Backprop calcula gradientes; nosotros corremos QDPSOs separados.
- **`n_rounds`**: sub-rondas de block coordinate descent dentro de cada sesión (T1: GS, T2-jacobi: Jacobi). Default `1`. T2 (async), T3 y T4 lo ignoran. **Recomendado:** T1 con `n_rounds=1` ya converge; T2-jacobi necesita `n_rounds=5` para igualar a T1.
- **Concurrencia (T3)**: en cada iteración del lockstep, después de que un swarm avanza, su gBest se vuelve **inmediatamente disponible** para el siguiente swarm en la misma iteración (Gauss-Seidel iter-por-iter, no Jacobi).
- **T2 vs T2-jacobi**: ambas paralelizan capas en threads. **T2 (async)** nunca congela `current`: cada step lee un snapshot ad-hoc del estado global y escribe su slice — la información fluye continuamente entre capas (Gauss-Seidel relajado). **T2-jacobi** congela `current` durante todo el round y sincroniza al final — más simple pero pierde la propagación inmediata.
- **Lock-free en T2 (async)**: las escrituras de threads distintos van a slices DISJUNTOS de `current` (cada capa escribe su offset). Las lecturas pueden traer un estado intermedio (algunas capas recientes, otras un step atrás), pero ningún byte es garbage. Es el patrón Hogwild! aplicado a QDPSO.
- **Warm-start cluster (TODAS las técnicas, post 2026-05-05)**: cada QDPSO se inicializa con K partículas concentradas alrededor de un punto semilla (`current[layer_k]` en T1; `params_init[layer_k]` en T2/T2-jacobi/T3; `params_init` completo en T4) y P-K random para preservar exploración. Reduce dramáticamente la varianza por cold start. Configurable via `--warm-particles` y `--warm-noise`. Helper compartido en `techniques/warm_start.py`.
- **Warm-start adicional en T2-jacobi**: además del cluster inicial, el QDPSO de cada capa persiste durante TODA la sesión y entre rounds llama `reuse_with(new_cf)` que mantiene `positions` (la geometría espacial explorada) y re-evalúa `pbest_values` con la nueva cf. Es lo que hace que Jacobi converja en lugar de oscilar.

---

## Definición y descripción detallada por técnica

### T1 — Secuencial / Block Coordinate Descent (Gauss-Seidel a nivel de capa)

**Definición.** Block coordinate descent (BCD) sobre el vector de pesos del MLP, donde cada bloque es una capa. Las capas se optimizan **una a la vez** en orden back-to-front (salida → entrada). Cada capa corre un QDPSO completo de `max_iter` pasos contra un `current` que ya contiene los updates más recientes de las capas posteriores. Es **Gauss-Seidel a nivel de capa**: la información fluye inmediatamente entre capas dentro del mismo sweep.

**Por qué back-to-front.** La capa de salida define el espacio de error inmediato (logits → cross-entropy con las etiquetas). Fijarla primero le da a las capas anteriores un objetivo más estable y mejor calibrado. Optimizar la entrada antes que la salida tendría el problema inverso: la entrada se ajustaría a un mapeo (capas posteriores) que aún no se ha definido bien.

**Estructura por sesión.**
```
for round in 0..n_rounds-1:
    for k in [N-1, N-2, ..., 0]:        # back-to-front
        cost_fn_k = cf vs current        # capas != k congeladas a current
        QDPSO_k = nuevo (max_iter pasos)
        callback ON_NEW_BEST → eval val con (current con candidato en slot k)
        current[layer_k] = mejor segmento por val
```
- El QDPSO de cada (round, capa) se crea fresco con seed determinista. No hay persistencia entre capas — cada uno explora desde random init.
- Después de cerrar un QDPSO, su mejor segmento (por val, no por train_loss) se inserta en `current` y se descarta el resto del estado.
- La capa siguiente (`k-1`) construye su `cost_fn` contra el `current` ya actualizado: ve la capa `k` "fresca", las capas posteriores `k+1..N-1` también frescas (del round actual o anterior), y las capas anteriores `0..k-2` viejas (las verá refinar en el próximo round).

**Por qué se llama Gauss-Seidel.** En álgebra numérica, Gauss-Seidel resuelve sistemas lineales bloque por bloque, donde cada bloque usa los valores **ya actualizados** de los bloques anteriores en la misma iteración. Aquí: cada capa se optimiza contra un `current` que ya incluye los updates de los bloques anteriores en el sweep actual. Es el mismo patrón aplicado a optimización no-lineal estocástica.

**Por qué `n_rounds=1` suele bastar.** En redes pequeñas (3 capas como iris con hidden=12), un solo sweep back-to-front es suficiente para que las capas converjan al óptimo del modelo combinado. Con más rounds los retornos son decrecientes: la primera ronda hace la mayor parte del trabajo, las siguientes refinan marginalmente. En redes más profundas o con `n_features` alto sí puede haber valor en `n_rounds > 1`.

**Trade-off.** Secuencial puro, sin paralelismo. Es el baseline de **calidad** — la combinación más limpia entre exploración (un QDPSO completo por capa) y propagación inmediata (GS). Su velocidad es proporcional a `N × max_iter`, donde N = num_layers.

---

### T2 — BCD asíncrono lock-free (Hogwild-style)

**Definición.** Variante paralela y asíncrona de T1. **N threads concurrentes** (uno por capa), cada uno corriendo su propio QDPSO persistente sobre un tensor compartido `current`. **Sin barreras** durante toda la sesión: cada thread lee, computa y escribe a su ritmo. Es una **relajación asíncrona** del Gauss-Seidel de T1 — preserva la idea de propagar información continuamente entre capas, pero saca la sincronización del camino crítico.

**Inspiración.** El paper *"Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent"* (Niu, Recht, Ré, Wright, 2011) demostró que para SGD con updates "sparse" (que tocan pocas coordenadas), eliminar locks acelera el entrenamiento sin degradar significativamente la convergencia. Aquí aplicamos el mismo principio a QDPSO multi-capa: las escrituras de threads distintos van a **slices disjuntos** de `current` (cada capa tiene su offset), así que el conflicto write-write no existe a nivel de coordenadas. Las lecturas pueden capturar un estado mixto (algunas capas recientes, otras un step atrás), pero ningún byte es garbage — todos son floats válidos.

**Por step de cada thread.**
1. **Snapshot ad-hoc**: `frozen = current.clone()` — lectura lock-free del estado global en este momento.
2. **Construir cost_fn fresca** vs `frozen`. Inyectar en el opt: `opt._cf = fresh_cf; opt._vectorized_cf = None`.
3. **Un paso del QDPSO**: `kernel_update + apply_boundary + update_best`, incrementar `_iters`.
4. **Escribir slice**: `current[layer_k] = opt.gbest` — visible para los otros threads en su próxima lectura ad-hoc.
5. **Si `update_best` retornó mejora**, evaluar val con un segundo snapshot ad-hoc + el nuevo gbest, y guardar el mejor segmento por val.

**Es un Gauss-Seidel relajado, no Jacobi.** En GS estricto, la capa `k-1` ve siempre la versión más reciente de la capa `k`. En T2 (async), la capa `k-1` ve la versión que estaba en `current` cuando hizo su `clone()` — puede ser de "hace 1 step" si la capa `k` justo terminó de escribir, o más antigua si la capa `k` está corriendo un step más lento. Lo importante: la información **fluye** entre capas continuamente. Vs Jacobi puro (T2-jacobi), donde durante todo el round nada de lo que hace una capa es visible para las otras hasta la sincronización.

**Paralelización en GPU.** Cada thread corre dentro de su propio `torch.cuda.Stream`. Los kernels de capas distintas se pueden solapar en el scheduler de CUDA (mientras la capa `k` evalúa su cost_fn, la capa `k-1` puede estar en `kernel_update`). En CPU, el GIL serializa el código Python, pero las operaciones tensoriales liberan el GIL durante kernels nativos, así que sigue habiendo solapamiento parcial.

**Por qué no usa `n_rounds`.** El round-by-round explícito de T1 se reemplaza por la concurrencia continua de threads. La unidad temporal pasa a ser el **step individual** del QDPSO, no el sweep completo. Cada thread hace `max_iter` steps, todos en paralelo, intercambiando información en cada step a través de `current`.

**Trade-offs.**
- **vs T1**: paralelización real, speedup teórico hasta `~N × ` (limitado por scheduling y memory bandwidth). Costo: ligero **no-determinismo** entre corridas (el orden de scheduling de threads cambia el snapshot que cada uno lee).
- **vs T2-jacobi**: T2 nunca pierde información durante el round (como sí pasa con Jacobi); pero a cambio asume el riesgo lock-free (que el paper Hogwild! mostró que es benigno).
- **vs T3**: T3 sincroniza al final de cada iter (todos avanzan 1 step, luego intercambian); T2 no sincroniza nunca. T3 es determinista; T2 acepta no-determinismo a cambio de paralelización real.

---

### T2-jacobi — Variante: paralelización Jacobi + warm-start QDPSO

**Definición.** Variante alternativa de T2 donde las capas se entrenan en paralelo contra un **snapshot CONGELADO** de `current` durante todo el round. Sincronización determinista al final de cada round; warm-start del QDPSO entre rounds para que Jacobi no diverja. Vive como variante para comparar el costo de la oscilación de Jacobi puro vs la propagación continua del async.

**Por qué Jacobi puro (sin warm-start) no funciona para QDPSO.** En Jacobi clásico (álgebra lineal), cada bloque resuelve su sub-problema asumiendo que los demás bloques están fijos en su valor inicial del round; al final del round se actualiza todo a la vez. Para sistemas lineales bien condicionados, esto converge — más lento que GS, pero converge. Para PSO/QDPSO sin warm-start, **cada round es una lotería independiente**: el QDPSO arranca con partículas random, no recuerda búsquedas previas, y la solución que encuentra para la capa `k` puede ser inconsistente con lo que las otras capas encontraron en paralelo. Resultado empírico: oscila o diverge incluso en iris.

**Solución `reuse_with(new_cf)`.** El QDPSO de cada capa **persiste durante toda la sesión**. Entre rounds, sólo se reemplaza la cost function (porque `frozen_for_round` cambió), pero las `positions` (la nube de partículas) y los `pbest` se mantienen. Las partículas siguen explorando desde donde estaban en el round anterior; al re-evaluar `pbest_values` con la nueva cf, mantienen la **geometría espacial** pero adaptan los valores. Esto da continuidad entre rounds y permite que Jacobi converja.

**Estructura por sesión.**
```
opts = [QDPSO_0, ..., QDPSO_{N-1}]   # 1 persistente por capa, sobreviven toda la sesión
for round in 0..n_rounds-1:
    frozen_for_round = current.clone()                     # snapshot único del round
    ThreadPoolExecutor(max_workers=N) lanza N tareas:
        si round == 0:  opts[k] usa cost_fn vs params_init
        si round  > 0:  opts[k].reuse_with(new_cf vs frozen_for_round)
        opts[k].callbacks.register(ON_NEW_BEST, eval val con frozen_for_round + gbest)
        opts[k].optimize()                                # max_iter pasos completos
    cuda.synchronize()                                    # barrera de fin de round
    for r in results: current[layer_k] = r.best_segment   # sincronización Jacobi
```

**Por qué necesita más rounds que T1.** Resultado clásico de álgebra numérica: **Jacobi converge más lento que Gauss-Seidel**, típicamente con un factor constante extra de iteraciones para alcanzar el mismo error. Empíricamente en iris, T2-jacobi con `n_rounds=5` empata a T1 con `n_rounds=1`. El precio del paralelismo determinista de Jacobi es ese factor.

**Por qué se mantiene como variante (no se eliminó).** Es académicamente interesante para comparar:
- Paralelización con sincronización determinista (T2-jacobi) vs asíncrona (T2 async).
- En condiciones de hardware donde lock-free no se aprovecha (CPU sin solapamiento real, redes muy profundas con cost_fn lenta), la versión Jacobi puede ser preferible porque su comportamiento es predecible y reproducible.
- Sirve como ablation: muestra el costo (en convergencia) de eliminar la propagación continua que sí tiene T2.

---

### T3 — Concurrencia: lockstep Gauss-Seidel iter-por-iter

**Definición.** Algoritmo cooperativo donde N swarms (uno por capa) avanzan en **sincronía paso a paso**. En cada iteración del bucle externo, los swarms se actualizan en orden back-to-front intercambiando sus gBests **dentro de la misma iteración** (Gauss-Seidel iter-por-iter, no Jacobi). El bucle se ejecuta serial en Python (lockstep) pero la cooperación entre swarms es muy fina-granular.

**Diferencia conceptual con T1.**
- **T1** hace BCD a nivel de **capa**: un QDPSO completo (`max_iter` steps) por capa, antes de pasar a la siguiente. La capa `k-1` no ve cambios en `k` hasta que `k` completó su entrenamiento.
- **T3** hace GS a nivel de **iteración**: un solo step de cada QDPSO, en orden back-to-front, intercambiando gBests entre cada step. La capa `k-1` ve el nuevo gBest de `k` después de UN solo step de `k`.

T3 es **mucho más fino-granular** que T1 en cuanto al intercambio de información entre capas.

**Estructura por sesión.**
```
swarms = [QDPSO_0, ..., QDPSO_{N-1}]   # N persistentes
para cada k: warm_start_cluster(swarms[k], params_init[layer_k], K, σ)
best_combined = combine_gbests(params_init, swarms)
best_val_cost = eval val (best_combined)

for it in 0..max_iter-1:               # BUCLE LOCKSTEP
    current = combine_gbests(params_init, swarms)   # snapshot inicial del iter
    for k in [N-1, ..., 0]:            # back-to-front, dentro del iter
        opt_k._cf = build_layer_cost_fn(k, frozen=current)   # cf fresca
        opt_k.kernel_update()          # 1 paso del QDPSO
        opt_k._positions = apply_boundary(...)
        opt_k.update_best()
        opt_k._iters += 1
        current[layer_k] = opt_k.gbest # ◄── el siguiente swarm en ESTE iter ya lo ve

    # Snapshot del modelo combinado por iter:
    val_cost = eval(current); train_loss = CE+L2(current)
    if (val_cost <  best_val_cost) or
       (val_cost == best_val_cost and train_loss < best_train_loss):
        best_combined = current.clone()
```

**Por qué "iter-por-iter".** Es el patrón más fuerte de Gauss-Seidel posible aplicado a multi-swarm: cada step intermedio del QDPSO de la capa `k` es inmediatamente visible para la capa `k-1` en el mismo iter. La información se difunde tan rápido como puede.

**Trade-off de implementación lockstep.** Python no permite paralelizar este patrón sin perder la garantía GS (lockstep estricto requiere serializar). Sin embargo, el costo computacional dominante (`kernel_update` y `update_best`) ya es **vectorizado en GPU** sobre las P partículas del swarm. La pérdida de paralelismo entre threads se compensa con la rapidez de información compartida y la compactación de operaciones GPU.

**Warm-start cluster (compartido con T4).** Cada swarm de capa `k` se inicializa con K partículas alrededor de `params_init[layer_k]` + (P-K) random. La cadena monótona inter-sesión llena `params_init` con el mejor global hasta el momento; el cluster aprovecha esa semilla y reduce significativamente la varianza entre sesiones (la cadena se estabiliza en lugar de "olvidar" lo aprendido).

**Selección dentro de la sesión.** Snapshot del `current` combinado al final de cada iter; si mejora `val_cost`, se guarda. Tie-breaker `train_loss` (con `<=` en val_cost) para preferir el snapshot **más maduro** en empates — el QDPSO sigue convergiendo en train aunque val haya saturado, así que entre snapshots con el mismo val el de menor train_loss es el más generalizable.

---

### T4 — Single swarm con warm-start cluster

**Definición.** Baseline clásico: un único QDPSO sobre el **vector plano completo** de todos los pesos del MLP. **Sin descomposición** por capa, sin paralelización entre swarms, sin concurrencia. Es el approach "naive" contra el que se comparan T1, T2, T2-jacobi y T3.

**Estructura.**
```
d_total = Σ (n_in_k × n_out_k + n_out_k)   # toda la red
cost_fn = build_full_cost_fn(...)
QDPSO(dim = d_total)
warm_start_cluster(opt, params_init, K, σ):
    positions[0]      = params_init             # seed exacto
    positions[1..K-1] = params_init + N(0, σ·rango)   # K-1 perturbaciones
    positions[K..P-1] = random init             # exploración
    re-evalúa pbest con cf actual
opt.callbacks.register(ON_NEW_BEST, eval val con opt.gbest)
opt.optimize()                                  # max_iter pasos
```

**Dimensionalidad.** Para iris con `hidden=12`: `d_total = (4×12 + 12) + (12×3 + 3) = 60 + 39 = 99` dimensiones en un solo enjambre. Para breast con `hidden=15`: `d_total = (30×15 + 15) + (15×2 + 2) = 465 + 32 = 497`. Crece rápido con la profundidad y el ancho.

**Cadena inter-sesión vía warm-start cluster.** No hay "swap de pesos por capa" como en las otras técnicas. En su lugar, K partículas se inicializan como perturbaciones gaussianas alrededor de `params_init` (el mejor global hasta ahora), y P-K quedan random para preservar exploración. Mantiene la mejor solución conocida como semilla mientras evita la convergencia prematura.

**Callback con `<=` (no `<`).** Al evaluar val cada vez que `gbest` mejora (en train+L2), si el nuevo `val_cost` es **igual** al mejor previo, se acepta. Razón: `opt.gbest_value` decrece monótonamente entre disparos del callback (el QDPSO es minimizante en train), así que entre dos snapshots con el mismo val el segundo es el más **maduro** — más convergido en train, mejor candidato a generalizar.

**Cuándo gana T4.** En redes muy pequeñas (d_total bajo), un single swarm con suficientes partículas puede ser más eficiente que cualquier descomposición: no paga overhead de coordinación, y la cost_fn evaluada toda-de-una vez es más rápida que N cost_fns por capa. En iris (d_total=99) T4 frecuentemente está cerca de T1/T3.

**Cuándo pierde T4.** En redes grandes (d_total ≫ n_particles), el espacio de búsqueda se vuelve intratable para un solo enjambre: las partículas no exploran lo suficiente y el QDPSO converge a óptimos locales muy lejos del óptimo global. Las técnicas multi-swarm dividen el problema en sub-espacios manejables (cada capa k tiene `dim_k ≪ d_total`) y por eso escalan mejor.

**Rol como baseline.** T4 es el "control" experimental: si una técnica multi-swarm no supera a T4, no justifica su complejidad. Cuanto más profunda/ancha la red, más amplio es el margen esperado entre T1/T2/T3 y T4.

---

## Flujo de cada técnica (ASCII)

Diagramas del cuerpo de cada `run_session_tX` en `techniques/tX_*.py`. Toda la lógica de selección por val, cadena monótona y guard A vive en `techniques/base.py:run_sessions()` y se aplica igual a las cuatro.

### T1 — Secuencial / Block Coordinate Descent (Gauss-Seidel a nivel de capa)

```
                                  ┌─────────────────────────────────────┐
                                  │  params_init (mejor global hasta k) │
                                  └─────────────────┬───────────────────┘
                                                    ▼
                                          current = params_init.clone()
                                                    │
        ┌───────────────────────────────────────────┴───────────────────────────────┐
        │                              for round in 0..n_rounds-1                   │
        │  ┌─────────────────────────────────────────────────────────────────────┐  │
        │  │           for k in [num_layers-1, ..., 1, 0]   (back-to-front)      │  │
        │  │                                                                     │  │
        │  │   ┌───────────────────────────────────────────────────────────┐     │  │
        │  │   │  cost_fn_k = build_layer_cost_fn(k, frozen=current)       │     │  │
        │  │   │             (capas != k congeladas a `current` actual)    │     │  │
        │  │   └───────────────────────┬───────────────────────────────────┘     │  │
        │  │                           ▼                                         │  │
        │  │   ┌───────────────────────────────────────────────────────────┐     │  │
        │  │   │  QDPSO_k = NUEVO  (seed = sseed*10000 + round*100 + k)    │     │  │
        │  │   │  callback ON_NEW_BEST → eval val con `current`+gbest      │     │  │
        │  │   │                       → guarda val_best.params si mejora  │     │  │
        │  │   │  opt.optimize()  (max_iter pasos)                         │     │  │
        │  │   └───────────────────────┬───────────────────────────────────┘     │  │
        │  │                           ▼                                         │  │
        │  │      current[offset_k : offset_k+size_k] = val_best.params          │  │
        │  │                  ▲                                                  │  │
        │  │                  │ propagación inmediata: la siguiente              │  │
        │  │                  │ capa (k-1) ya ve esta actualización              │  │
        │  └──────────────────┴──────────────────────────────────────────────────┘  │
        └───────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                                       return params_best = current
```

### T2 — BCD asíncrono lock-free (Hogwild-style)

```
   params_init ──► current   ◄── tensor COMPARTIDO entre threads
        │             (lecturas/escrituras lock-free a slices disjuntos)
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Crear UN QDPSO PERSISTENTE por capa                         │
   │  opts = [QDPSO_0, QDPSO_1, ..., QDPSO_{N-1}]                 │
   │  streams = [cuda.Stream() per capa]   (sólo si CUDA)         │
   └────────────────────────────┬─────────────────────────────────┘
                                │
   ┌────────────────────────────┴─────────────────────────────────┐
   │      ThreadPoolExecutor (max_workers = N)   ◄── 1 sola fase  │
   │   ┌──────────┬──────────┬──────────┬──────────┐              │
   │   │ thread 0 │ thread 1 │ thread 2 │   ...    │              │
   │   │ capa N-1 │ capa N-2 │ capa N-3 │  capa 0  │              │
   │   ▼          ▼          ▼          ▼                         │
   │  ┌─────────────────────────────────────────────────┐         │
   │  │  for step in 0..max_iter-1:    (sin barreras)   │         │
   │  │                                                 │         │
   │  │    frozen = current.clone()  ◄── lectura ad-hoc │         │
   │  │             (puede traer updates parciales      │         │
   │  │              de otras capas — eso es CORRECTO)  │         │
   │  │                                                 │         │
   │  │    cost_fn = build_layer_cost_fn(k, frozen)     │         │
   │  │    opt._cf = cost_fn;  opt._vectorized_cf = None│         │
   │  │                                                 │         │
   │  │    opt.kernel_update()       ◄── 1 paso QDPSO   │         │
   │  │    opt._positions = apply_boundary(...)         │         │
   │  │    improved = opt.update_best()                 │         │
   │  │    opt._iters += 1                              │         │
   │  │                                                 │         │
   │  │    current[layer_k] = opt.gbest                 │         │
   │  │      ▲                                          │         │
   │  │      │ escritura LOCK-FREE a slice disjunto     │         │
   │  │      │ (visible para los otros threads en su    │         │
   │  │      │  próxima lectura ad-hoc)                 │         │
   │  │                                                 │         │
   │  │    if improved:                                 │         │
   │  │       candidate = current.clone()               │         │
   │  │       candidate[layer_k] = opt.gbest            │         │
   │  │       val_cost = MSE(y_val, predict(candidate)) │         │
   │  │       if val_cost < val_best.val_cost:          │         │
   │  │           val_best.params = opt.gbest.clone()   │         │
   │  └─────────────────────┬───────────────────────────┘         │
   │                        ▼                                     │
   │   torch.cuda.synchronize()   ◄── única barrera (al final)    │
   └──────────────────────────────────────────────────────────────┘
                                │
                                ▼
   params_best = combine(val_best.params por capa, sobre params_init)
```

### T2-jacobi — Variante: paralelización Jacobi + warm-start QDPSO

```
   params_init ──► current
        │
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Crear UN QDPSO PERSISTENTE por capa (con cost_fn vs init)   │
   │  opts = [QDPSO_0, QDPSO_1, ..., QDPSO_{N-1}]                 │
   └────────────────────────────┬─────────────────────────────────┘
                                │
   ┌────────────────────────────┴─────────────────────────────────┐
   │              for round in 0..n_rounds-1                      │
   │                                                              │
   │   frozen_for_round = current.clone()   ◄── snapshot Jacobi   │
   │   streams = [cuda.Stream() per capa]   (sólo si CUDA)        │
   │                                                              │
   │       ThreadPoolExecutor (max_workers = N)                   │
   │   ┌──────────┬──────────┬──────────┬──────────┐              │
   │   │ thread 0 │ thread 1 │ thread 2 │   ...    │              │
   │   │ capa N-1 │ capa N-2 │ capa N-3 │  capa 0  │              │
   │   ▼          ▼          ▼          ▼                         │
   │  ┌─────────────────────────────────────────────────┐         │
   │  │ if round == 0:                                  │         │
   │  │     opt usa cost_fn inicial (vs params_init)    │         │
   │  │ else:                                           │         │
   │  │     opt.reuse_with(new_cost_fn(frozen_for_round)│         │
   │  │       └─ MANTIENE positions (geometría)         │         │
   │  │       └─ re-evalúa pbest con nueva cf           │         │
   │  │   ◄── WARM-START del QDPSO entre rounds         │         │
   │  │                                                 │         │
   │  │ opt.callbacks.register(ON_NEW_BEST, ...)        │         │
   │  │   (eval val con frozen_for_round + gbest)       │         │
   │  │ opt.optimize() en su cuda.Stream                │         │
   │  └─────────────────────┬───────────────────────────┘         │
   │                        ▼                                     │
   │   torch.cuda.synchronize()  ◄── barrera de fin de round      │
   │                                                              │
   │   ── SINCRONIZACIÓN JACOBI ──                                │
   │   for r in results:                                          │
   │       current[layer_k] = r.best_segment                      │
   │       (todas las capas se vieron `frozen_for_round`,         │
   │        no la versión updated por su vecina del mismo round)  │
   └──────────────────────────────────────────────────────────────┘
                                │
                                ▼
                      params_best = current
```

### T3 — Concurrencia lockstep Gauss-Seidel iter-por-iter

```
   params_init
        │
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Crear N swarms (uno por capa) + warm_start_cluster          │
   │  para cada uno alrededor de params_init[layer_k]:            │
   │     partícula 0 = seed exacto                                │
   │     partículas 1..K-1 = seed + N(0, noise_scale·rango)       │
   │     partículas K..P-1 = random                               │
   └────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
   best_combined = combine_gbests(params_init, swarms)
   best_val_cost, best_train_loss ← eval(best_combined)

   ┌──────────────────────────────────────────────────────────────┐
   │           for it in 0..max_iter-1   (LOCKSTEP)               │
   │                                                              │
   │   current = combine_gbests(params_init, swarms)              │
   │             ▲ snapshot al inicio de la iteración             │
   │                                                              │
   │   for k in [N-1, ..., 1, 0]:    (back-to-front)              │
   │     ┌────────────────────────────────────────────────────┐   │
   │     │ opt_k._cf = build_layer_cost_fn(k, frozen=current) │   │
   │     │ opt_k.kernel_update()       ◄── 1 paso QDPSO       │   │
   │     │ opt_k._positions = apply_boundary(...)             │   │
   │     │ opt_k.update_best()                                │   │
   │     │ opt_k._iters += 1                                  │   │
   │     │                                                    │   │
   │     │ current[layer_k] = opt_k.gbest                     │   │
   │     │   ▲                                                │   │
   │     │   │ GAUSS-SEIDEL: el siguiente swarm en ESTA       │   │
   │     │   │ misma iteración ya ve el gBest fresco          │   │
   │     └───┴────────────────────────────────────────────────┘   │
   │                                                              │
   │   val_cost = predict(X_val, current)                         │
   │   if val_cost <= best_val_cost:                              │
   │       train_loss_now = CE+L2(current, X_train)               │
   │       if (val_cost <  best_val_cost) or                      │
   │          (val_cost == best_val_cost and                      │
   │           train_loss_now < best_train_loss):                 │
   │           best_combined = current.clone()                    │
   │           iter_at_best = it+1                                │
   └──────────────────────────────────────────────────────────────┘
                                │
                                ▼
                  params_best = best_combined
```

### T4 — Single swarm con warm-start cluster

```
   params_init
        │
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  cost_fn = build_full_cost_fn(...)                           │
   │  dim = d_total = Σ(weights+biases de todas las capas)        │
   └────────────────────────────┬─────────────────────────────────┘
                                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  QDPSOTensorOptimized(dim = d_total, ...)                    │
   │                                                              │
   │  warm_start_cluster(opt, params_init, K, noise):             │
   │     positions[0]      = params_init                          │
   │     positions[1..K-1] = params_init + N(0, noise·rango)      │
   │     positions[K..P-1] = random (los del init original)       │
   │     re-evalúa pbest con cf actual                            │
   │                                                              │
   │  callback ON_NEW_BEST:                                       │
   │     y_pred = predict(X_val, opt.gbest)                       │
   │     val_cost = MSE(y_val, y_pred)                            │
   │     if val_cost <= val_best.val_cost:    ◄── `<=`            │
   │        val_best.params = opt.gbest.clone()                   │
   │     (`<=` favorece el snapshot MÁS MADURO ante empates,      │
   │      porque opt.gbest_value decrece monótonamente)           │
   │                                                              │
   │  opt.optimize()    ◄── max_iter pasos sobre el vector plano  │
   └────────────────────────────┬─────────────────────────────────┘
                                ▼
                  params_best = val_best.params
```

### Comparación de flujo: cuándo intercambian información las capas

```
   T1 (BCD GS)         T2 (async)         T2-jacobi          T3 (lockstep GS)     T4 (single)
   ───────────         ──────────         ─────────          ────────────────     ───────────
   round r:            (sin rounds)       round r:           iter i:              (no aplica
   ┌─────────┐         ┌────────────┐     ┌─────────┐        ┌───────────────┐     un solo
   │ capa N-1│ ──┐     │thread N-1  │     │capa N-1 │┐       │ capa N-1 step │     vector
   │ optimiza│   │     │ step ▲ step│     │capa N-2 ││       │  ↓ updates    │     plano)
   │ completa│   │     │ step │ step│ ──► │  ...    ││ todas │ capa N-2 step │
   └────┬────┘   │     │ step │ step│     │capa  0  ││ vs    │  ↓ updates    │
        │        │     │  ▲   │     │     └─────────┘│ frozen│     ...       │
        ▼        │     │  │   ▼     │        │       │_for_  │ capa  0  step │
   ┌─────────┐   │     │  └─ current◄┐       ▼       │ round │  ↓ updates    │
   │ capa N-2│ ◄─┘     │     compart.│  propagación ◄┘       └───────┬───────┘
   │ ve cap  │         │  (lock-free)│   sync                        │
   │ N-1 fresc│         └────────────┘   AL FINAL                    ▼
   └────┬────┘         lectura ad-hoc    DEL ROUND               siguiente iter:
        ...            cada step                                 combine current
                                                                  de gBests
```

| Aspecto | T1 (BCD GS) | T2 (async) | T2-jacobi | T3 (lockstep GS) | T4 (single) |
|---|---|---|---|---|---|
| Swarms | 1 nuevo por (round, capa) | 1 persistente por capa | 1 persistente por capa | 1 persistente por capa | 1 sobre `d_total` |
| Comparte info entre capas | round-to-round dentro del sweep | continuo (cada step lee snapshot ad-hoc) | sólo al final del round | cada iteración (in-iter) | N/A |
| Paralelización real | No (secuencial) | Sí (threads + CUDA streams, lock-free) | Sí (threads + CUDA streams, sync por round) | No (lockstep serial) | No |
| Sincronización | N/A | Única barrera al final de la sesión | Barrera al final de cada round | Barrera al final de cada iter | N/A |
| Warm-start | `warm_start_cluster` por (round, capa) | `warm_start_cluster` inicial | `warm_start_cluster` inicial + `reuse_with(new_cf)` entre rounds | `warm_start_cluster` inicial | `warm_start_cluster` inicial |
| `n_rounds` aplica | Sí | No (usa `max_iter`) | Sí | No (usa `max_iter`) | No |
| Selección dentro de la sesión | Callback `ON_NEW_BEST` por capa, val_cost | Por capa: si `update_best` mejora, eval val ad-hoc | Callback `ON_NEW_BEST` por capa, val_cost | Snapshot del combinado por iter, val + tie train_loss | Callback global, val + `<=` |

---

## Evaluación: K-fold CV vs single-split

El runner soporta dos modos de evaluación. **Para reportes serios usar K-fold CV**; el single-split sigue disponible para exploración rápida.

### Modo 1 — Single-split (default, legacy)

Un único split estratificado **70/15/15** + N sesiones internas con cadena monótona:
```bash
python runner.py --technique t2 --dataset iris --n-sessions 15 --seed 42
```
- **Ventaja**: rápido, mismo split entre seeds (solo cambia la inicialización del QDPSO).
- **Limitación**: en datasets pequeños (iris val=23, circle val=30), val satura a 1.0 fácil → cherry-pick por suerte de muestreo. La métrica reportada es UN punto, no una distribución.

### Modo 2 — K-fold CV (recomendado para reportes)

K splits estratificados con `StratifiedKFold`, una corrida por fold, métricas agregadas con media ± std:
```bash
python runner.py --technique t2 --dataset iris --cv-folds 5 --n-sessions 1 --seed 42
```

**Cómo funciona**:
- Cada fold tiene su propio split: `test = ~n_total/K` muestras, `train+val = ~(K-1)·n_total/K`. Dentro del train+val, sub-split estratificado para val (`val_frac_within_train=0.15` → ~12% del total).
- **Sin data leakage**: `MinMaxScaler` se ajusta SÓLO con el train del fold, se aplica a val y test.
- Cada muestra del dataset cae en test exactamente una vez a través de los K folds. La métrica agregada (mean ± std) es estadísticamente sólida.
- `--n-sessions N` se aplica POR FOLD. Recomendado `--n-sessions 1` en CV (la cadena monótona pierde sentido cuando el split cambia entre folds).

**Tamaños de splits con K=5**:

| Dataset | n_total | test/fold | train+val/fold | val (15% del 80%) |
|---|---:|---:|---:|---:|
| iris   | 150 | 30 | 120 | ~18 |
| wine   | 178 | 36 | 142 | ~21 |
| circle | 500 (made) | 100 | 400 | ~60 |
| breast | 569 | 114 | 455 | ~68 |

**Output estructurado**:
```
output/<tech>/<dataset>/cv5/<config_tag>/
├── fold0/
│   ├── run.json
│   └── gbest.npy
├── fold1/
│   ...
├── fold4/
│   ...
└── cv_summary.json   ← agregado: mean, std, min, max por fold
```

**Comparación de modos**:

| Aspecto | Single-split | K-fold CV |
|---|---|---|
| Splits | 1 | K (cada fold uno distinto) |
| Test = | 15% × 1 split | 100% del dataset (a través de K folds) |
| Métrica reportada | 1 valor | media ± std |
| Robustez en datasets pequeños | ⚠ baja | ✅ alta |
| Reproducibilidad | seed fija inicializaciones, no el split | seed fija ambos |
| Costo | N corridas | K × N corridas (N=1 por fold típicamente) |
| Cuándo usar | exploración rápida, debugging | reportes, paper, decisión final |

### Comandos típicos CV

```bash
# Una técnica × un dataset (5 folds, 1 corrida por fold)
python runner.py --technique t2 --dataset iris --cv-folds 5 --n-sessions 1 \
  --n-particles 100 --max-iter 100 --seed 42

# Benchmark CV completo: 5 técnicas × 4 datasets
for tech in t1 t2 t2-jacobi t3 t4; do
  for ds in iris wine circle breast; do
    extra=""
    [ "$tech" = "t2-jacobi" ] && extra="--n-rounds 5"
    python runner.py --technique $tech --dataset $ds --cv-folds 5 --n-sessions 1 \
      --n-particles 100 --max-iter 100 --seed 42 $extra
  done
done

# Comparar (compare.py muestra ambas tablas: single-split y CV)
python compare.py
```

---

## Decisiones metodológicas clave

- **Splits**: dos modos disponibles (ver "Evaluación: K-fold CV vs single-split" arriba).
  - **Single-split 70/15/15** (`prepare_dataset`): legacy, scaler ajustado sobre todo el dataset (potencial data leakage menor). Útil para iteración rápida.
  - **K-fold CV** (`prepare_dataset_kfold`): recomendado para reportes. StratifiedKFold + sub-split estratificado para val. **Sin data leakage** — scaler fit sólo en train del fold.
- **Score combinado**: `(1 - val_acc) + val_mse / mse_max` con `mse_max = (n_classes - 1)²`. Menor es mejor. Pondera error rate y error ordinal por igual.
- **Jerarquía de selección** (`techniques/base.py`):
  1. menor `val_score` (tol 1e-9)
  2. menor `train_loss` (tol relativa 5%)
  3. menor `||θ||²` (Occam)
  4. mayor `val_acc`
  5. menor `val_mse`
- **Cadena monótona inter-sesión**: `params_init = best_params global`, no la sesión inmediata anterior. Una sesión peor no envenena la cadena.
- **Guard A**: si la salida de la técnica es peor que `params_init` (por val_score), fallback a `params_init` y se marca `guard_triggered=True`.
- **CE estable**: `logsumexp` en `cross_entropy_from_logits` evita overflow numérico.
- **L2 normalizado** (`core/loss.py:l2_penalty`):
  ```
  l2 = (lambda / 2) · ||W||² / (d_total · n_train)
  ```
  Justificación de la doble normalización (convención Bishop / Goodfellow / MAP bayesiano):
  - **Por `d_total`**: hace `lambda` invariante al tamaño de la red. Sin esta normalización, una red grande (breast d_total≈3000) recibiría un L2 efectivo ~30× más fuerte que iris (d_total≈100) bajo el mismo `lambda`.
  - **Por `n_train`**: la cross-entropy se promedia sobre `n_train`, así que el L2 debe normalizarse igual para mantener un balance constante entre datos y prior. Bayesianamente, `lambda` corresponde al inverso de la varianza del prior gaussiano sobre los pesos.
  - **Default `--lambda-l2 = 10.0`** (con la nueva normalización). El default viejo (1e-3 sin normalización) NO es comparable: para igualar su efecto se necesitaría `lambda_new = 1e-3 · d_total · n_train`, que varía de ~10 (iris) a ~1160 (breast) entre datasets. **Sugerencia**: ajustar con grid search `λ ∈ {1, 10, 100}` por dataset si se quiere afinar.

---

## Comandos básicos

### Activación

```bash
conda activate pytorch_qpso_gpu
cd /home/schancay/Documentos/work/full-stack/repo/personal/nn_qdpso/src/multi-swarm_mlp/re-design
```

### Corrida individual

```bash
python runner.py --technique t3 --dataset iris \
  --n-sessions 15 --n-particles 100 --max-iter 500 \
  --hidden-sizes 12 --lambda-l2 10.0 --g 0.96 \
  --seed 42 --device auto
```

### Configs recomendados para iris

```bash
# T1: BCD ya converge en 1 round
python runner.py --technique t1 --dataset iris \
  --hidden-sizes 12 --n-sessions 5 --n-particles 50 --max-iter 100 --seed 42 --n-rounds 1

# T2 (async): paraleliza T1 sin barreras; n_rounds no aplica
python runner.py --technique t2 --dataset iris \
  --hidden-sizes 12 --n-sessions 5 --n-particles 50 --max-iter 100 --seed 42

# T2-jacobi (variante): Jacobi necesita más rounds para igualar a T1
python runner.py --technique t2-jacobi --dataset iris \
  --hidden-sizes 12 --n-sessions 5 --n-particles 50 --max-iter 100 --seed 42 --n-rounds 5

# T3 y T4 ignoran n-rounds
python runner.py --technique t3 --dataset iris \
  --hidden-sizes 12 --n-sessions 5 --n-particles 50 --max-iter 100 --seed 42

python runner.py --technique t4 --dataset iris \
  --hidden-sizes 12 --n-sessions 5 --n-particles 50 --max-iter 100 --seed 42
```

### Benchmarks (4 datasets × 5 técnicas)

```bash
for tech in t1 t2 t2-jacobi t3 t4; do
  for ds in iris wine circle breast; do
    python runner.py --technique $tech --dataset $ds \
      --n-sessions 15 --n-particles 100 --max-iter 500
  done
done
```

### Comparación + plots

```bash
python compare.py                                # tabla + CSV + MD
python compare.py --filter-dataset breast        # solo un dataset
python compare.py --sort-by test_acc             # ordenar
python compare.py --plot                         # learning_curve.png en cada output
python compare.py --plot --plot-comparison       # + comparison_by_dataset.png
```

---

## Output estructurado

**Single-split** (`output/<tech>/<dataset>/<hidden>_iters<N>_part<P>_seed<S>/`):
- **`run.json`** — config + sesiones + ganadora + curvas de aprendizaje
- **`gbest.npy`** — vector de pesos del modelo ganador
- **`learning_curve.png`** (con `compare.py --plot`) — val_acc + test_acc por sesión + barras de gap

**K-fold CV** (`output/<tech>/<dataset>/cv<K>/<hidden>_iters<N>_part<P>_seed<S>/`):
- **`fold<i>/run.json`** y **`fold<i>/gbest.npy`** — uno por fold
- **`fold<i>/learning_curve.png`** (con `compare.py --plot`)
- **`cv_summary.json`** — agregado: por-fold + mean / std / min / max sobre `test_acc`, `val_acc`, `train_loss`, `total_time`

`compare.py` detecta ambos automáticamente y produce **dos tablas** + `compare.csv` (single-split) + `compare_cv.csv` (CV).
