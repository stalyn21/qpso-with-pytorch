#!/usr/bin/env bash
# MCW con QDPSOo: ORG (84) + isomap/pca/mds x {42,21,14,7}, 5 semillas,
# 1000 épocas (config del paper). La primera corrida extrae y cachea features.
set -u
cd "$(dirname "$0")"

SEEDS="0 1 2 3 4"

for s in $SEEDS; do
  echo "=== MCW ORG seed=$s ==="
  python main_qpsoO.py --dataset mcw --seed "$s"
  for red in isomap pca mds; do
    for c in 42 21 14 7; do
      echo "=== MCW $red c=$c seed=$s ==="
      python main_qpsoO.py --dataset mcw --reduction "$red" --components "$c" --seed "$s"
    done
  done
done
echo "TANDA MCW COMPLETA"
