#!/usr/bin/env bash
# Barrido de partículas del paper 2 (MLP): T1, T2 y T3 con una sola sesión.
#
# Igualdad de condiciones entre las tres técnicas:
#   - ITERS iteraciones POR CAPA (enjambre); en total ITERS × n_capas en las tres.
#   - P partículas POR CAPA (enjambre); en total P × n_capas en las tres.
#   - Cluster inicial = WARM_PCT % de P (parte entera), no un número fijo.
#   - PATIENCE >= ITERS para que T1 no corte antes (T2 y T3 nunca cortan).
#   - T2 y T3 re-evalúan el pbest en cada paso (es parte de la técnica): gastan ~2×
#     las evaluaciones de T1. Queda registrado en cada run.json.
#
# Cada variable se puede cambiar sin editar el archivo, p. ej.:
#   ITERS=200 WARM_PCT=40 PARTICLES="25 50" SEEDS="42" ./sweep_particles.sh
# Corre en serie para que los tiempos de pared sean comparables.
set -u
OUT=${OUT:-output/paper2-1s}
ITERS=${ITERS:-100}
PATIENCE=${PATIENCE:-$ITERS}
WARM_PCT=${WARM_PCT:-20}
PARTICLES=${PARTICLES:-"25 50 75 100"}
SEEDS=${SEEDS:-"42 43 44"}
DATASETS=${DATASETS:-"iris breast segment satimage digits"}
TECHNIQUES=${TECHNIQUES:-"t1 t2 t3"}
FOLDS=${FOLDS:-5}

echo "OUT=$OUT ITERS=$ITERS PATIENCE=$PATIENCE WARM_PCT=$WARM_PCT PARTICLES=[$PARTICLES]"
echo "SEEDS=[$SEEDS] DATASETS=[$DATASETS] TECHNIQUES=[$TECHNIQUES] FOLDS=$FOLDS"
for seed in $SEEDS; do
  for ds in $DATASETS; do
    for p in $PARTICLES; do
      for t in $TECHNIQUES; do
        t0=$(date +%s)
        python runner.py --technique "$t" --dataset "$ds" --cv-folds "$FOLDS" \
          --n-sessions 1 --n-rounds 1 --n-particles "$p" --warm-percent "$WARM_PCT" \
          --max-iter "$ITERS" --patience "$PATIENCE" --seed "$seed" --output-dir "$OUT" \
          > /dev/null || echo "FALLO seed=$seed $ds P=$p $t"
        echo "seed=$seed $ds P=$p $t $(( $(date +%s) - t0 ))s"
      done
    done
  done
done
