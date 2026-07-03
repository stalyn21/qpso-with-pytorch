# Block Coordinate Descent + Warm-start: arreglo estructural de T1 y T2

> **Fecha:** 2026-04-26
> **Aplica a:** `techniques/t1_sequential.py`, `techniques/t2_parallel.py`, `tensor_qpso/qpso_tensor_optimized.py`
> **Estado:** implementado y validado en iris

---

## TL;DR

T1 y T2 inicialmente tenían **contrapartes congeladas a `params_init` durante toda la sesión**. Esto las hacía estructuralmente incapaces de componer capas: cada capa optimizaba contra un contexto que no existía en el modelo combinado, y los resultados quedaban en accuracy random (~30-65% en iris).

Después de:
- **T1**: convertirla en **block coordinate descent (Gauss-Seidel a nivel de capa)** — la capa entrenada propaga su mejor segmento al `current` y la siguiente capa lo ve.
- **T2**: convertirla en **Jacobi con warm-start del QDPSO** — las N capas se entrenan en paralelo contra el mismo snapshot, sincronizan al final del round, y el QDPSO de cada capa MANTIENE sus partículas entre rounds para acumular progreso en lugar de hacer búsqueda random nueva en cada uno.

…ambas alcanzan **test_acc = 0.9565 en iris**, igualando o superando a T3/T4. T2 mantiene el speedup de paralelización (~1.94×) por round, a costa de necesitar más rounds que T1 (Jacobi converge más lento que Gauss-Seidel).

---

## Resultados (iris, hidden=12, seed=42, 5 sesiones, 50 partículas, 100 iters)

| Técnica | Config | Val acc | Test acc | Tiempo | Comentario |
|---------|--------|---------|----------|--------|-----------|
| **T1** | `n-rounds=1` | 1.0000 | **0.9565** | 1.25s | BCD: 1 sweep back-to-front basta |
| **T2** | `n-rounds=5` warm-start | 1.0000 | **0.9565** | 4.17s | Jacobi: 5 rounds para alcanzar a T1, ~1.94× speedup por round |
| T3 | lockstep iter | 1.0000 | 0.8696 | 1.36s | Concurrencia Gauss-Seidel iter-por-iter |
| T4 | single swarm | 1.0000 | 0.8261 | 0.81s | Optimiza todos los pesos juntos |

T1 y T2 igualados en calidad. T2 paga ~3× tiempo total para hacer paralelización, pero **sí logra paralelización efectiva** dentro de cada round.

---

## El problema original

### Antes del cambio

```python
# t1_sequential.py / t2_parallel.py viejo
for k in training_order:
    cost_fn = mlp.build_layer_cost_fn(
        k=k,
        frozen_flat=params_init,   # ← TODAS las otras capas congeladas a params_init
        ...
    )
```

Durante toda la sesión, cuando se entrenaba la capa k, las **otras N-1 capas** estaban en `params_init` (random en sesión 1; mejor global en sesiones siguientes). Esto significaba:

- En T1, cuando entrenaba la capa de entrada, la capa de salida era la del `params_init` random — no la que acababa de entrenarse.
- En T2, las N capas se entrenaban en paralelo, todas viendo el mismo `params_init`. Ninguna veía las actualizaciones de las otras.

**Síntoma típico:** `val_best=0.0000` en cada capa durante el callback (cada capa AISLADA logra 100% contra su contraparte random) pero `val=0.65` al evaluar el modelo combinado. Las capas no componían.

### Análisis cuantitativo (iris, T1 viejo)

Sesiones 0-4 (val_acc): `0.34, 0.34, 0.69, 0.26, 0.00`
- Sesiones aleatorias entre random (~33%) y un solo lucky strike (s2=0.69)
- Sesión 3 con val=0.26 = PEOR que random — la cadena propagó pesos malos

Y T2 viejo con cualquier `n-rounds`: nunca pasaba de 0.43.

---

## Cambio en T1: Block Coordinate Descent

`t1_sequential.py`:

```python
current = params_init.clone()  # ← contexto que evoluciona

for round_idx in range(n_rounds):                # nuevo: múltiples sweeps
    for k in training_order:                      # back-to-front
        cost_fn = mlp.build_layer_cost_fn(
            k=k, frozen_flat=current,              # ← `current` actualizado, no params_init
            ...
        )
        # ... entrenar capa k con QDPSO ...
        current[spec.offset:spec.offset+spec.size] = val_best["params"]  # ← propagar
```

**Diferencia operacional con T3:** T3 hace lockstep iter-por-iter (1 step de cada swarm por iter externa). T1 hace **QDPSO completo por capa antes de pasar a la siguiente** — Gauss-Seidel a nivel de capa, no de iter.

**Empíricamente en iris**, basta con `n-rounds=1` (un solo sweep back-to-front). Más rounds no mejoran y a veces empeoran por estocasticidad.

---

## Cambio en T2: Jacobi + warm-start del QDPSO

### Por qué Jacobi puro NO funciona

Jacobi puro = capas se entrenan en paralelo contra mismo snapshot, sincronizan al final del round, repeten. El problema observado:

- Round 0: las capas optimizan contra contexto random → no componen.
- Round 1: capas reinicializan QDPSO **random** otra vez → no acumulan progreso del round previo.
- Cada round es esencialmente una corrida independiente sobre un contexto cambiante.

**Síntoma:** `train_loss` por capa oscilaba entre rounds en lugar de bajar monótonamente. Con `n-rounds=10` no había convergencia.

### Solución: warm-start del QDPSO entre rounds

En `tensor_qpso/qpso_tensor_optimized.py` se agregó el método:

```python
def reuse_with(self, new_cf):
    """Re-arma el optimizador para una nueva cost function manteniendo positions."""
    self._cf = new_cf
    self._iters = 0
    self._no_improvement_count = 0
    # Mantiene positions (la diversidad espacial explorada).
    # Re-evalúa pbest_values con la nueva cf, recalcula gbest.
    self._pbest = self._positions.clone()
    self._pbest_values = self._evaluate(self._positions)
    self._gbest_value = float('inf')
    self.update_gbest(self._minimize)
```

Y en `t2_parallel.py`:

```python
# Crear UN QDPSO persistente por capa al inicio de la sesión
opts: List[QDPSOTensorOptimized] = [...]

for round_idx in range(n_rounds):
    frozen_for_round = current.clone()  # snapshot Jacobi

    with ThreadPoolExecutor(...) as pool:
        for k in submission_order:
            future_by_k[k] = pool.submit(
                _train_layer_round,
                opt=opts[k],                            # ← opt persistente
                is_first_round=(round_idx == 0),
                ...
            )

# Dentro del worker:
if not is_first_round:
    opt.reuse_with(cost_fn)   # ← warm-start: mantener partículas
result = opt.optimize()
```

**Qué se mantiene entre rounds:**
- `positions`: la geometría espacial explorada en el round previo (las partículas siguen donde estaban).
- `pbest` (geometry): se mantiene como `positions`, pero re-evaluado con la nueva cf.

**Qué se resetea:**
- `_iters`, `_no_improvement_count`, razón de convergencia.
- `_pbest_values`, `_gbest_value`: re-evaluados contra la nueva cf (los valores viejos no son comparables).

### Por qué funciona

Sin warm-start, cada round arrancaba random → las partículas pasaban iters re-explorando regiones ya vistas. Con warm-start, las partículas **adaptan** su búsqueda al nuevo objetivo manteniendo lo aprendido.

**Síntoma de éxito:** `train_loss` por capa baja monótonamente entre rounds (de 0.37 → 0.10 a través de los 5 rounds), y al final del último round las dos capas están "alineadas" porque cada una se ajustó iterativamente a la otra.

---

## Por qué T1 > T2 en redes pequeñas

Diferencia teórica clásica de iteración por bloques:

- **Gauss-Seidel (T1):** dentro de UN sweep, capa 0 ya ve la capa 1 RECIÉN actualizada. Información perfecta para componer en 1 round.
- **Jacobi (T2):** dentro de UN round, ambas capas optimizan contra el mismo `frozen_for_round`. La sincronización ocurre al final → cada capa "asume" que la otra es la del inicio del round.

Para sistemas lineales A·x=b, la convergencia de Jacobi tiene `ρ_J ≥ ρ_GS` (radio espectral). Jacobi necesita más iteraciones para llegar al mismo punto que GS. La mejora del warm-start no rompe esto — sólo evita que Jacobi diverja.

**En el paper:** "T2 alcanza calidad equivalente a T1 con ~5× más rounds, pero gana ~1.94× wall-clock por round → trade-off Jacobi vs GS clásico, viable gracias al warm-start de partículas."

---

## Guard A: no-regresión (en `base.py`)

Como protección adicional contra sesiones catastróficas, `run_sessions` evalúa al final de cada sesión:

```python
val_score_init = combined_score(val_acc_init, val_cost_init, mse_max)
if val_score_init < val_score:
    print("⚠ Guard A activado: técnica produjo peor que params_init. Fallback.")
    params_k = params_init.clone()
```

Si la técnica produce un `params_best` PEOR que el `params_init` con el que arrancó la sesión, se descarta y queda `params_init`. La cadena monótona ya protege el `best_global`, pero el guard limpia los logs por sesión y evita propagar basura aunque sea temporalmente.

Se observa activado ocasionalmente en T2 sesiones intermedias (cuando una sesión "se desalinea") y nunca en T1 (BCD es más estable).

---

## Reproducir

```bash
cd nn_qdpso/src/multi-swarm_mlp/re-design

rm -rf output/    # opcional — limpia corridas previas

python runner.py --technique t1 --dataset iris \
  --hidden-sizes 12 --n-sessions 5 --n-particles 50 --max-iter 100 \
  --seed 42 --n-rounds 1

python runner.py --technique t2 --dataset iris \
  --hidden-sizes 12 --n-sessions 5 --n-particles 50 --max-iter 100 \
  --seed 42 --n-rounds 5

python runner.py --technique t3 --dataset iris \
  --hidden-sizes 12 --n-sessions 5 --n-particles 50 --max-iter 100 --seed 42

python runner.py --technique t4 --dataset iris \
  --hidden-sizes 12 --n-sessions 5 --n-particles 50 --max-iter 100 --seed 42

python compare.py --plot
```

**Configs recomendados por técnica:**
- T1: `n-rounds=1` (basta con un sweep BCD)
- T2: `n-rounds=5` (Jacobi necesita más rounds para alcanzar a GS)
- T3, T4: ignoran `n-rounds`

---

## Archivos tocados

| Archivo | Cambio |
|---------|--------|
| `tensor_qpso/qpso_tensor_optimized.py` | + método `reuse_with(new_cf)` para warm-start |
| `techniques/t1_sequential.py` | Reescrito como BCD con `current` evolutivo y soporte de `n_rounds` |
| `techniques/t2_parallel.py` | Reescrito como Jacobi con N opts persistentes y warm-start entre rounds |
| `techniques/base.py` | + Guard A (fallback a params_init si la técnica regresiona) y campo `guard_triggered` en SessionResult |
| `runner.py` | + flag `--n-rounds` (default 1, sólo afecta T1/T2) |

---

## Ver también

- [`../../../MEMORY/nn_qdpso/re-design.md`](../../../../../MEMORY/nn_qdpso/re-design.md) — overview general del re-design
- [`../../../MEMORY/nn_qdpso/re-design-no-test-in-sessions.md`](../../../../../MEMORY/nn_qdpso/re-design-no-test-in-sessions.md) — por qué el test set no se usa para selección
