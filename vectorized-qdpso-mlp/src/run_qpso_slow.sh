#!/usr/bin/env bash
# QDPSO lento (implementación original por partícula): 1 semilla, 4 datasets.
# Solo para la comparación de tiempos QDPSOo vs QDPSO (~5 h estimadas).
set -u
cd "$(dirname "$0")"

for ds in circle iris wine breast_cancer; do
  echo "=== QPSO (lento) $ds seed=0 ==="
  python main_qpso.py --dataset "$ds" --seed 0
done
echo "TANDA QPSO LENTO COMPLETA"
