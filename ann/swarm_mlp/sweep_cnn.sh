#!/usr/bin/env bash
# Barridos y corridas de la CNN del paper 2 (MEMORY/.../paper2-cnn-plan.md).
#
# Corre el producto de todas las listas. T4 compara a IGUAL número de evaluaciones:
# recibe --match-evals-of $MATCH_T4 (por defecto t3), es decir, las iteraciones que
# caben en lo que gasta T3 con las mismas partículas e iteraciones.
#
# Paciencia: muy alta por defecto, para que ninguna técnica corte antes (T4 recibe
# varias veces las iteraciones de T3; T2 y T3 nunca cortan).
#
# Ejemplos (desde ann/swarm_mlp):
#   ./sweep_cnn.sh                                              # pilotos por defecto
#   FITNESS_LIST="1000 2000 5000" TECHNIQUES="t3" ./sweep_cnn.sh   # tamaño del fitness
#   G_VALUES="1.0 1.2 1.4" ./sweep_cnn.sh                        # g
#   SIZES="small medium" DATASETS="mnist fmnist" SEEDS="100 101 102 103 104" ./sweep_cnn.sh
# Corre en serie para que los tiempos de pared sean comparables.
#
# Variantes de preprocesamiento y rango de búsqueda: VARIANTS es una lista de
# "input_norm,act_norm,bounds_mode" (ver runner.py --help); BOUNDS_SCALE aplica a fanin.
#   VARIANTS="none,none,fixed none,none,fanin standardize,none,fixed none,layer,fixed" ./sweep_cnn.sh
#
# Recursos: con MONITOR=1 (por defecto) cada corrida se envuelve con monitor.py, que
# mide RAM, CPU, VRAM, uso y potencia de GPU, y disco cada MONITOR_INTERVAL segundos.
# Deja <OUT>/_resources/<corrida>.{json,csv}; tabla: python monitor.py summarize <dir>.
set -u
OUT=${OUT:-output/paper2-cnn}
SIZES=${SIZES:-"small"}
DATASETS=${DATASETS:-"fmnist"}
TECHNIQUES=${TECHNIQUES:-"t3 t2 t4"}
SEEDS=${SEEDS:-"100 101 102"}
PARTICLES=${PARTICLES:-"75"}
ITERS_LIST=${ITERS_LIST:-"200"}
G_VALUES=${G_VALUES:-"1.2"}
FITNESS_LIST=${FITNESS_LIST:-"2000"}
VAL_SIZE=${VAL_SIZE:-5000}
L2_FORM=${L2_FORM:-sample}
LAMBDAS=${LAMBDAS:-"0"}
WARM_PCT=${WARM_PCT:-30}
PATIENCE=${PATIENCE:-1000000}
MATCH_T4=${MATCH_T4:-t3}
SEL_METRIC=${SEL_METRIC:-bacc}
SEL_SE=${SEL_SE:-1}
DATA_SEED=${DATA_SEED:-100}
VARIANTS=${VARIANTS:-"none,none,fixed"}
BOUNDS_SCALE=${BOUNDS_SCALE:-2.449489742783178}
MONITOR=${MONITOR:-1}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-1.0}
RES_DIR=${RES_DIR:-$OUT/_resources}

echo "OUT=$OUT SIZES=[$SIZES] DATASETS=[$DATASETS] TECHNIQUES=[$TECHNIQUES] SEEDS=[$SEEDS]"
echo "PARTICLES=[$PARTICLES] ITERS_LIST=[$ITERS_LIST] G_VALUES=[$G_VALUES] FITNESS_LIST=[$FITNESS_LIST]"
echo "VAL_SIZE=$VAL_SIZE L2=$L2_FORM:[$LAMBDAS] WARM_PCT=$WARM_PCT PATIENCE=$PATIENCE MATCH_T4=$MATCH_T4"
echo "VARIANTS=[$VARIANTS] BOUNDS_SCALE=$BOUNDS_SCALE"
echo "SEL=$SEL_METRIC:$SEL_SE DATA_SEED=$DATA_SEED MONITOR=$MONITOR($MONITOR_INTERVAL s) -> $RES_DIR"
for seed in $SEEDS; do
  for ds in $DATASETS; do
    for size in $SIZES; do
      for fit in $FITNESS_LIST; do
        for p in $PARTICLES; do
          for it in $ITERS_LIST; do
            for g in $G_VALUES; do
              for lam in $LAMBDAS; do
               for var in $VARIANTS; do
                IFS=, read -r in_norm act_norm b_mode <<< "$var"
                for t in $TECHNIQUES; do
                  match=()
                  [ "$t" = "t4" ] && match=(--match-evals-of "$MATCH_T4")
                  mon=()
                  if [ "$MONITOR" = "1" ]; then
                    label="${t}_${ds}_${size}_fit${fit}_P${p}_I${it}_g${g}_l2-${lam}_in-${in_norm}_act-${act_norm}_b-${b_mode}_seed${seed}"
                    mon=(python monitor.py --out "$RES_DIR/$label" --interval "$MONITOR_INTERVAL" \
                         --disk-path "$OUT" --)
                  fi
                  t0=$(date +%s)
                  "${mon[@]}" python runner.py --model cnn --cnn-size "$size" --technique "$t" --dataset "$ds" \
                    --n-sessions 1 --n-rounds 1 --n-particles "$p" --warm-percent "$WARM_PCT" \
                    --max-iter "$it" --patience "$PATIENCE" "${match[@]}" --g "$g" \
                    --fitness-samples "$fit" --val-size "$VAL_SIZE" \
                    --l2-form "$L2_FORM" --lambda-l2 "$lam" \
                    --input-norm "$in_norm" --act-norm "$act_norm" \
                    --bounds-mode "$b_mode" --bounds-scale "$BOUNDS_SCALE" \
                    --selection-metric "$SEL_METRIC" --selection-se "$SEL_SE" --data-seed "$DATA_SEED" \
                    --seed "$seed" --output-dir "$OUT" \
                    > /dev/null || echo "FALLO seed=$seed $ds $size fit=$fit P=$p I=$it g=$g l2=$lam $var $t"
                  echo "seed=$seed $ds $size fit=$fit P=$p I=$it g=$g l2=$L2_FORM:$lam $var $t $(( $(date +%s) - t0 ))s"
                done
               done
              done
            done
          done
        done
      done
    done
  done
done
