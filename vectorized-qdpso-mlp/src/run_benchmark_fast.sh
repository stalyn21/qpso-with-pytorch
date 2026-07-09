#!/usr/bin/env bash
# Tanda rápida de resubmisión: QDPSOo, PSO, PSO_bound y Adam sobre los 4
# benchmark, 5 semillas. Adam corre a 100 épocas (mismo presupuesto de
# iteraciones que el paper) y a 1000 (a convergencia, comparación justa R2.3).
set -u
cd "$(dirname "$0")"

DATASETS="circle iris wine breast_cancer"
SEEDS="0 1 2 3 4"

for ds in $DATASETS; do
  for s in $SEEDS; do
    echo "=== QPSOo $ds seed=$s ==="
    python main_qpsoO.py --dataset "$ds" --seed "$s"
    echo "=== PSO $ds seed=$s ==="
    python main_pso.py --dataset "$ds" --variant PSO --seed "$s"
    echo "=== PSO_bound $ds seed=$s ==="
    python main_pso.py --dataset "$ds" --variant PSO_bound --seed "$s"
    echo "=== Adam(100) $ds seed=$s ==="
    python main_adam.py --dataset "$ds" --epochs 100 --seed "$s" --tag ep100
    echo "=== Adam(1000) $ds seed=$s ==="
    python main_adam.py --dataset "$ds" --epochs 1000 --seed "$s" --tag ep1000
  done
done
echo "TANDA RAPIDA COMPLETA"
