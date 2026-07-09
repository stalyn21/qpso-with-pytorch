#!/usr/bin/env bash
# Barrido de sensibilidad del coeficiente de contracción-expansión g (R2.6/R3.5):
# QDPSOo con g en {0.6, 0.8, 0.96, 1.13, 1.3, 1.5} sobre los 4 benchmark
# (5 seeds) y sobre MCW isomap/7 (5 seeds). El tag g<valor> hace únicos los
# nombres de salida; el valor de g queda guardado en el JSON (config.g).
set -u
cd "$(dirname "$0")"

GVALUES="0.6 0.8 0.96 1.13 1.3 1.5"
SEEDS="0 1 2 3 4"

for g in $GVALUES; do
  for ds in circle iris wine breast_cancer; do
    for s in $SEEDS; do
      echo "=== g-sweep $ds g=$g seed=$s ==="
      python main_qpsoO.py --dataset "$ds" --g "$g" --seed "$s" --tag "g$g"
    done
  done
done

for g in $GVALUES; do
  for s in $SEEDS; do
    echo "=== g-sweep mcw-isomap7 g=$g seed=$s ==="
    python main_qpsoO.py --dataset mcw --reduction isomap --components 7 \
      --g "$g" --seed "$s" --tag "g$g"
  done
done
echo "BARRIDO G COMPLETO"
