#!/usr/bin/env bash
# Barridos de sensibilidad del paper 2 (MLP): T1, T2 y T3 con una sola sesión.
#
# Corre el producto de todas las listas. Para variar UN factor a la vez, se da una
# lista en ese factor y un solo valor (el punto base) en los demás.
#
# Igualdad de condiciones entre las tres técnicas:
#   - Iteraciones POR CAPA (enjambre); en total iteraciones × n_capas en las tres.
#   - Partículas POR CAPA (enjambre); en total partículas × n_capas en las tres.
#   - Cluster inicial = WARM_PCT % de las partículas (parte entera).
#   - Paciencia = iteraciones de la corrida (o PATIENCE si se da), para que T1 no
#     corte antes; T2 y T3 nunca cortan.
#   - T2 y T3 re-evalúan el pbest en cada paso (es parte de la técnica): gastan ~2×
#     las evaluaciones de T1. Queda registrado en cada run.json.
#
# Ejemplos (desde ann/swarm_mlp):
#   L2_FORM=sample LAMBDAS="0 0.001 0.01 0.1 1" ./sweep.sh          # regularización
#   G_VALUES="1.0 1.2 1.4 1.6 1.8 2.0" ITERS_LIST="100 200" ./sweep.sh  # g × iteraciones
#   PARTICLES="25 50 75 100" ./sweep.sh                            # partículas
#   SEL_METRIC=bacc SEL_SE=1 DATA_SEED=200 ./sweep.sh             # regla de 1 SE, otra partición
#
# SEL_METRIC / SEL_SE: regla de selección por validación (core/selection.py). Por
# defecto acc y 0 = regla histórica. DATA_SEED: semilla de la partición de los datos
# (folds, val y test); --seed sólo cambia la inicialización. El directorio de salida
# NO incluye DATA_SEED: usar un OUT distinto por partición.
# Corre en serie para que los tiempos de pared sean comparables.
set -u
OUT=${OUT:-output/paper2-1s}
PARTICLES=${PARTICLES:-"50"}
ITERS_LIST=${ITERS_LIST:-"100"}
G_VALUES=${G_VALUES:-"1.13"}
L2_FORM=${L2_FORM:-mean}
LAMBDAS=${LAMBDAS:-"10"}
WARM_PCT=${WARM_PCT:-30}
PATIENCE=${PATIENCE:-}
SEEDS=${SEEDS:-"42 43 44"}
DATASETS=${DATASETS:-"iris breast segment satimage digits"}
TECHNIQUES=${TECHNIQUES:-"t1 t2 t3"}
FOLDS=${FOLDS:-5}
SEL_METRIC=${SEL_METRIC:-acc}
SEL_SE=${SEL_SE:-0}
DATA_SEED=${DATA_SEED:-100}

echo "OUT=$OUT PARTICLES=[$PARTICLES] ITERS_LIST=[$ITERS_LIST] G_VALUES=[$G_VALUES]"
echo "L2_FORM=$L2_FORM LAMBDAS=[$LAMBDAS] WARM_PCT=$WARM_PCT PATIENCE=${PATIENCE:-=iteraciones}"
echo "SEEDS=[$SEEDS] DATASETS=[$DATASETS] TECHNIQUES=[$TECHNIQUES] FOLDS=$FOLDS"
echo "SEL_METRIC=$SEL_METRIC SEL_SE=$SEL_SE DATA_SEED=$DATA_SEED"
for seed in $SEEDS; do
  for ds in $DATASETS; do
    for p in $PARTICLES; do
      for it in $ITERS_LIST; do
        for g in $G_VALUES; do
          for lam in $LAMBDAS; do
            for t in $TECHNIQUES; do
              pat=${PATIENCE:-$it}
              t0=$(date +%s)
              python runner.py --technique "$t" --dataset "$ds" --cv-folds "$FOLDS" \
                --n-sessions 1 --n-rounds 1 --n-particles "$p" --warm-percent "$WARM_PCT" \
                --max-iter "$it" --patience "$pat" --g "$g" \
                --l2-form "$L2_FORM" --lambda-l2 "$lam" \
                --selection-metric "$SEL_METRIC" --selection-se "$SEL_SE" --data-seed "$DATA_SEED" \
                --seed "$seed" --output-dir "$OUT" \
                > /dev/null || echo "FALLO seed=$seed $ds P=$p I=$it g=$g l2=$lam $t"
              echo "seed=$seed $ds P=$p I=$it g=$g l2=$L2_FORM:$lam sel=$SEL_METRIC:$SEL_SE $t $(( $(date +%s) - t0 ))s"
            done
          done
        done
      done
    done
  done
done
