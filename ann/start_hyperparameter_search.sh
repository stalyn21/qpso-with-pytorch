#!/bin/bash
# =============================================================================
# start.sh - Script de configuracion para busqueda de hiperparametros QPSO/QDPSO
# =============================================================================
#
# Uso:
#   ./start.sh                    # Ejecutar con valores por defecto
#   ./start.sh --help             # Mostrar ayuda del script Python
#
# Configuracion:
#   Modifica las variables de abajo segun tus necesidades
#
# =============================================================================

# =============================================================================
# CONFIGURACION PRINCIPAL
# =============================================================================

# Optimizadores a explorar
# Opciones: "QPSO" "QDPSO" o ambos "QPSO QDPSO"
OPTIMIZERS="QDPSO"

# Estrategias a explorar
# Opciones: "forward" "weighted" "layerwise" o combinaciones
STRATEGIES="forward weighted layerwise"

# Numero de trials (configuraciones a probar)
N_TRIALS=100

# =============================================================================
# CONFIGURACION DE EXPLORACION
# =============================================================================

# Forzar exploracion inicial de todas las combinaciones?
# - false: Busqueda 100% Bayesiana (default - recomendado para exploracion optima)
# - true:  Encola 1 trial inicial por cada combinacion optimizer x strategy
#          Util para garantizar que todas las combinaciones sean evaluadas
FORCE_INITIAL_TRIALS=false

# =============================================================================
# ESPACIO DE BUSQUEDA - QPSO
# =============================================================================

# Rango para alpha inicial (QPSO)
ALPHA_START_MIN=0.7
ALPHA_START_MAX=1.0

# Rango para alpha final (QPSO)
ALPHA_END_MIN=0.3
ALPHA_END_MAX=0.7

# =============================================================================
# ESPACIO DE BUSQUEDA - QDPSO
# =============================================================================

# Rango para factor g (QDPSO)
G_MIN=0.90
G_MAX=0.99

# =============================================================================
# ESPACIO DE BUSQUEDA - ENJAMBRE
# =============================================================================

# Rango de particulas
N_PARTICLES_MIN=20
N_PARTICLES_MAX=80

# Rango de iteraciones maximas
MAX_ITERS_MIN=50
MAX_ITERS_MAX=300

# =============================================================================
# ESPACIO DE BUSQUEDA - ARQUITECTURA
# =============================================================================

# Rango de capas ocultas
N_HIDDEN_LAYERS_MIN=1
N_HIDDEN_LAYERS_MAX=7

# Rango del multiplicador de neuronas (input_dim * multiplier)
NEURONS_MULTIPLIER_MIN=1.5
NEURONS_MULTIPLIER_MAX=4.0

# Rango de decaimiento de neuronas entre capas
NEURON_DECAY_MIN=0.5
NEURON_DECAY_MAX=0.9

# =============================================================================
# ESPACIO DE BUSQUEDA - ESTRATEGIA WEIGHTED
# =============================================================================

# Rango de decaimiento por capa
LAYER_DECAY_MIN=0.5
LAYER_DECAY_MAX=0.9

# Rango de regularizacion
REGULARIZATION_MIN=0.001
REGULARIZATION_MAX=0.1

# =============================================================================
# ESPACIO DE BUSQUEDA - ESTRATEGIA LAYERWISE
# =============================================================================

# Rango de iteraciones por capa
ITERS_PER_LAYER_MIN=20
ITERS_PER_LAYER_MAX=80

# Rango de iteraciones de fine-tuning
FINE_TUNE_ITERS_MIN=20
FINE_TUNE_ITERS_MAX=80

# =============================================================================
# ESPACIO DE BUSQUEDA - OTROS
# =============================================================================

# Rango de limite de pesos
WEIGHT_BOUND_MIN=0.5
WEIGHT_BOUND_MAX=2.0

# Rango de paciencia (early stopping)
PATIENCE_MIN=20
PATIENCE_MAX=60

# =============================================================================
# CONFIGURACION AVANZADA
# =============================================================================

# Semilla para reproducibilidad
SEED=42

# Timeout en segundos (dejar vacio para sin limite)
TIMEOUT=""

# Ruta al dataset MCW
DATASET_PATH="./data/img/mcw"

# Directorio de salida para resultados
OUTPUT_DIR="./results/hyperparameter_search"

# =============================================================================
# CONFIGURACION DE RENDIMIENTO (FASE 1 MEJORAS)
# =============================================================================

# Paralelismo: -1 = auto (75% de cores), 1 = secuencial
N_JOBS=-1

# Dispositivo: auto, cuda, mps, cpu
DEVICE="auto"

# Persistencia SQLite (permite continuar búsquedas interrumpidas)
# Dejar vacio para en memoria (no persistente)
STORAGE_PATH="./results/hyperparameter_search/optuna_study.db"

# Continuar estudio existente: true o false
RESUME_STUDY=true

# =============================================================================
# FASE 2: ESPACIO DE BUSQUEDA AMPLIADO
# =============================================================================

# Activaciones a explorar (separadas por espacio)
# Opciones: relu tanh sigmoid leaky_relu elu gelu
ACTIVATIONS="relu tanh leaky_relu elu gelu"  # Default: solo tanh (fija)
# ACTIVATIONS="relu tanh gelu"  # Ejemplo: optimizar entre varias

# Dropout (min, max). Si min == max, se usa fijo
DROPOUT_MIN=0.0
DROPOUT_MAX=0.0  # Default: sin dropout
# DROPOUT_MAX=0.5  # Ejemplo: optimizar entre 0 y 0.5

# Batch normalization (separadas por espacio)
# Opciones: true false
BATCH_NORM="false"  # Default: sin batch norm
# BATCH_NORM="true false"  # Ejemplo: optimizar

# Boundary strategies para QPSO (separadas por espacio)
# Opciones: clamp reflect wrap random
BOUNDARY_STRATEGIES="clamp"  # Default: solo clamp (fija)
# BOUNDARY_STRATEGIES="clamp reflect"  # Ejemplo: optimizar

# Tolerancia para convergencia (min, max). Escala logaritmica
TOL_MIN=1e-12
TOL_MAX=1e-12  # Default: fija en 1e-12
# TOL_MAX=1e-8  # Ejemplo: optimizar

# =============================================================================
# FASE 2: PRUNER Y CALLBACKS
# =============================================================================

# Tipo de pruner: median (conservador) o hyperband (agresivo)
PRUNER="median"

# Early stopping GLOBAL (detiene toda la búsqueda si no mejora)
# 0 = desactivado (default), >0 = numero de trials sin mejora para detener
EARLY_STOPPING_PATIENCE=0  # Default: desactivado
# EARLY_STOPPING_PATIENCE=20  # Ejemplo: detener si 20 trials sin mejora

# Frecuencia de checkpoints (cada N trials)
# 0 = desactivado, >0 = guardar checkpoint cada N trials
CHECKPOINT_FREQUENCY=10  # Default: cada 10 trials

# =============================================================================
# NO MODIFICAR DEBAJO DE ESTA LINEA
# =============================================================================

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Obtener directorio del script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/main_hyperparameter_search.py"

# Funcion para imprimir encabezado
print_header() {
    echo ""
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

# Funcion para imprimir sub-seccion
print_section() {
    echo -e "\n${CYAN}--- $1 ---${NC}"
}

# Funcion para imprimir configuracion
print_config() {
    print_header "CONFIGURACION DE BUSQUEDA"

    print_section "Principal"
    echo -e "  ${GREEN}Optimizadores:${NC}    ${OPTIMIZERS}"
    echo -e "  ${GREEN}Estrategias:${NC}      ${STRATEGIES}"
    echo -e "  ${GREEN}Numero de trials:${NC} ${N_TRIALS}"

    if [ "$FORCE_INITIAL_TRIALS" = true ]; then
        n_opt=$(echo $OPTIMIZERS | wc -w)
        n_strat=$(echo $STRATEGIES | wc -w)
        n_combos=$((n_opt * n_strat))
        echo -e "  ${YELLOW}Modo:${NC}             Hibrido (${n_combos} trial(s) inicial(es) + Bayesiano)"
    else
        echo -e "  ${GREEN}Modo:${NC}             100% Bayesiano (TPE Sampler)"
    fi

    print_section "Espacio de Busqueda - QPSO"
    echo -e "  alpha_start:      [${ALPHA_START_MIN}, ${ALPHA_START_MAX}]"
    echo -e "  alpha_end:        [${ALPHA_END_MIN}, ${ALPHA_END_MAX}]"

    print_section "Espacio de Busqueda - QDPSO"
    echo -e "  g:                [${G_MIN}, ${G_MAX}]"

    print_section "Espacio de Busqueda - Enjambre"
    echo -e "  n_particles:      [${N_PARTICLES_MIN}, ${N_PARTICLES_MAX}]"
    echo -e "  max_iters:        [${MAX_ITERS_MIN}, ${MAX_ITERS_MAX}]"

    print_section "Espacio de Busqueda - Arquitectura"
    echo -e "  n_hidden_layers:  [${N_HIDDEN_LAYERS_MIN}, ${N_HIDDEN_LAYERS_MAX}]"
    echo -e "  neurons_mult:     [${NEURONS_MULTIPLIER_MIN}, ${NEURONS_MULTIPLIER_MAX}]"
    echo -e "  neuron_decay:     [${NEURON_DECAY_MIN}, ${NEURON_DECAY_MAX}]"

    print_section "Espacio de Busqueda - Weighted"
    echo -e "  layer_decay:      [${LAYER_DECAY_MIN}, ${LAYER_DECAY_MAX}]"
    echo -e "  regularization:   [${REGULARIZATION_MIN}, ${REGULARIZATION_MAX}]"

    print_section "Espacio de Busqueda - Layerwise"
    echo -e "  iters_per_layer:  [${ITERS_PER_LAYER_MIN}, ${ITERS_PER_LAYER_MAX}]"
    echo -e "  fine_tune_iters:  [${FINE_TUNE_ITERS_MIN}, ${FINE_TUNE_ITERS_MAX}]"

    print_section "Espacio de Busqueda - Otros"
    echo -e "  weight_bound:     [${WEIGHT_BOUND_MIN}, ${WEIGHT_BOUND_MAX}]"
    echo -e "  patience:         [${PATIENCE_MIN}, ${PATIENCE_MAX}]"

    print_section "Configuracion Avanzada"
    echo -e "  ${GREEN}Semilla:${NC}          ${SEED}"
    echo -e "  ${GREEN}Dataset:${NC}          ${DATASET_PATH}"
    echo -e "  ${GREEN}Salida:${NC}           ${OUTPUT_DIR}"
    if [ -n "$TIMEOUT" ]; then
        echo -e "  ${GREEN}Timeout:${NC}          ${TIMEOUT}s"
    else
        echo -e "  ${GREEN}Timeout:${NC}          Sin limite"
    fi

    print_section "Rendimiento (Fase 1 Mejoras)"
    echo -e "  ${GREEN}Paralelismo:${NC}      ${N_JOBS} (auto = 75% cores)"
    echo -e "  ${GREEN}Dispositivo:${NC}      ${DEVICE}"
    if [ -n "$STORAGE_PATH" ]; then
        echo -e "  ${GREEN}Storage SQLite:${NC}   ${STORAGE_PATH}"
    else
        echo -e "  ${GREEN}Storage SQLite:${NC}   En memoria (no persistente)"
    fi
    echo -e "  ${GREEN}Continuar estudio:${NC} ${RESUME_STUDY}"

    print_section "Fase 2: Espacio Ampliado"
    echo -e "  ${GREEN}Activaciones:${NC}     ${ACTIVATIONS}"
    echo -e "  ${GREEN}Dropout:${NC}          [${DROPOUT_MIN}, ${DROPOUT_MAX}]"
    echo -e "  ${GREEN}Batch Norm:${NC}       ${BATCH_NORM}"
    echo -e "  ${GREEN}Boundary Strat:${NC}   ${BOUNDARY_STRATEGIES}"
    echo -e "  ${GREEN}Tolerancia:${NC}       [${TOL_MIN}, ${TOL_MAX}]"

    print_section "Fase 2: Pruner y Callbacks"
    echo -e "  ${GREEN}Pruner:${NC}           ${PRUNER}"
    if [ "$EARLY_STOPPING_PATIENCE" -eq 0 ]; then
        echo -e "  ${GREEN}Early Stop Global:${NC} desactivado"
    else
        echo -e "  ${GREEN}Early Stop Global:${NC} ${EARLY_STOPPING_PATIENCE} trials"
    fi
    if [ "$CHECKPOINT_FREQUENCY" -eq 0 ]; then
        echo -e "  ${GREEN}Checkpoint freq:${NC}  desactivado"
    else
        echo -e "  ${GREEN}Checkpoint freq:${NC}  cada ${CHECKPOINT_FREQUENCY} trials"
    fi
    echo ""
}

# Verificar que el script Python existe
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}ERROR: No se encontro el script Python en: ${PYTHON_SCRIPT}${NC}"
    exit 1
fi

# Si se pasa --help, mostrar ayuda del script Python
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    python "$PYTHON_SCRIPT" --help
    exit 0
fi

# Mostrar configuracion
print_config

# Construir comando
CMD="python $PYTHON_SCRIPT"

# Configuracion principal
CMD="$CMD --optimizer $OPTIMIZERS"
CMD="$CMD --strategy $STRATEGIES"
CMD="$CMD --n-trials $N_TRIALS"
CMD="$CMD --seed $SEED"
CMD="$CMD --dataset-path $DATASET_PATH"
CMD="$CMD --output-dir $OUTPUT_DIR"

# Agregar timeout si esta definido
if [ -n "$TIMEOUT" ]; then
    CMD="$CMD --timeout $TIMEOUT"
fi

# Agregar flag para desactivar trials iniciales forzados (busqueda 100% Bayesiana)
if [ "$FORCE_INITIAL_TRIALS" = false ]; then
    CMD="$CMD --no-ensure-combinations"
fi

# Rendimiento (Fase 1 Mejoras)
CMD="$CMD --n-jobs $N_JOBS"
CMD="$CMD --device $DEVICE"

# Storage SQLite
if [ -n "$STORAGE_PATH" ]; then
    CMD="$CMD --storage $STORAGE_PATH"
fi

# Resume study
if [ "$RESUME_STUDY" = false ]; then
    CMD="$CMD --no-resume"
fi

# Espacio de busqueda - QPSO
CMD="$CMD --alpha-start $ALPHA_START_MIN $ALPHA_START_MAX"
CMD="$CMD --alpha-end $ALPHA_END_MIN $ALPHA_END_MAX"

# Espacio de busqueda - QDPSO
CMD="$CMD --g-range $G_MIN $G_MAX"

# Espacio de busqueda - Enjambre
CMD="$CMD --n-particles $N_PARTICLES_MIN $N_PARTICLES_MAX"
CMD="$CMD --max-iters $MAX_ITERS_MIN $MAX_ITERS_MAX"

# Espacio de busqueda - Arquitectura
CMD="$CMD --n-hidden-layers $N_HIDDEN_LAYERS_MIN $N_HIDDEN_LAYERS_MAX"
CMD="$CMD --neurons-multiplier $NEURONS_MULTIPLIER_MIN $NEURONS_MULTIPLIER_MAX"
CMD="$CMD --neuron-decay $NEURON_DECAY_MIN $NEURON_DECAY_MAX"

# Espacio de busqueda - Weighted
CMD="$CMD --layer-decay $LAYER_DECAY_MIN $LAYER_DECAY_MAX"
CMD="$CMD --regularization $REGULARIZATION_MIN $REGULARIZATION_MAX"

# Espacio de busqueda - Layerwise
CMD="$CMD --iters-per-layer $ITERS_PER_LAYER_MIN $ITERS_PER_LAYER_MAX"
CMD="$CMD --fine-tune-iters $FINE_TUNE_ITERS_MIN $FINE_TUNE_ITERS_MAX"

# Espacio de busqueda - Otros
CMD="$CMD --weight-bound $WEIGHT_BOUND_MIN $WEIGHT_BOUND_MAX"
CMD="$CMD --patience $PATIENCE_MIN $PATIENCE_MAX"

# =========================================================================
# FASE 2: Espacio de busqueda ampliado
# =========================================================================

# Activaciones
CMD="$CMD --activations $ACTIVATIONS"

# Dropout
CMD="$CMD --dropout $DROPOUT_MIN $DROPOUT_MAX"

# Batch normalization
CMD="$CMD --batch-norm $BATCH_NORM"

# Boundary strategies
CMD="$CMD --boundary-strategies $BOUNDARY_STRATEGIES"

# Tolerancia
CMD="$CMD --tol $TOL_MIN $TOL_MAX"

# =========================================================================
# FASE 2: Pruner y Callbacks
# =========================================================================

# Pruner
CMD="$CMD --pruner $PRUNER"

# Early stopping global
CMD="$CMD --early-stopping-patience $EARLY_STOPPING_PATIENCE"

# Checkpoint frequency
CMD="$CMD --checkpoint-frequency $CHECKPOINT_FREQUENCY"

# Mostrar comando
echo -e "${YELLOW}Comando a ejecutar:${NC}"
echo "  $CMD"
echo ""

# Confirmar ejecucion
read -p "Presiona ENTER para continuar o Ctrl+C para cancelar..."
echo ""

# Ejecutar
print_header "INICIANDO BUSQUEDA DE HIPERPARAMETROS"
eval $CMD

# Mostrar resultado
EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    print_header "BUSQUEDA COMPLETADA EXITOSAMENTE"
    echo -e "\n${GREEN}Resultados guardados en: ${OUTPUT_DIR}${NC}\n"
else
    echo -e "\n${RED}ERROR: La busqueda termino con codigo de error ${EXIT_CODE}${NC}\n"
fi

exit $EXIT_CODE
