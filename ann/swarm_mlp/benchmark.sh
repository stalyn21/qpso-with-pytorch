#!/usr/bin/env bash
# Benchmark del paper 2 (MLP): validación cruzada de 5 folds a presupuesto igual.
# 5 datasets × 5 técnicas × 3 semillas, red por defecto [3n, 2n, n], QDPSOo.
# Corre en serie para que los tiempos de pared sean comparables (~37 min por semilla
# en una GTX 1050 Ti). Figuras: python visualize.py --output-dir output/paper2
set -u
OUT=${OUT:-output/paper2}
for seed in 42 43 44; do
  for ds in iris breast segment satimage digits; do
    for t in t4 t1 t3 t2 t2-jacobi; do
      extra=""
      [ "$t" = "t2-jacobi" ] && extra="--n-rounds 2"
      t0=$(date +%s)
      python runner.py --technique "$t" --dataset "$ds" --cv-folds 5 --n-sessions 2 \
        --n-particles 50 --eval-budget 80240 --seed "$seed" $extra --output-dir "$OUT" \
        > /dev/null || echo "FALLO seed=$seed $ds $t"
      echo "seed=$seed $ds $t $(( $(date +%s) - t0 ))s"
    done
  done
done
