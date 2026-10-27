#!/usr/bin/env bash
# Barrido de partículas del paper 2 (MLP): T1, T2 y T3 con una sola sesión.
# Es sweep.sh con la lista de partículas; las condiciones comunes están en sweep.sh.
#
# Cada variable se puede cambiar sin editar el archivo, p. ej.:
#   ITERS=200 WARM_PCT=40 PARTICLES="25 50" SEEDS="42" ./sweep_particles.sh
#   L2_FORM=sample LAMBDA=0.01 ./sweep_particles.sh
set -u
cd "$(dirname "$0")"
export PARTICLES=${PARTICLES:-"25 50 75 100"}
export ITERS_LIST=${ITERS:-${ITERS_LIST:-100}}
export LAMBDAS=${LAMBDA:-${LAMBDAS:-10}}
exec ./sweep.sh
