#!/usr/bin/env python
"""
start_hyperparameter_search.py - Script de configuracion para busqueda de hiperparametros

Uso:
    python start_hyperparameter_search.py              # Ejecutar con valores configurados
    python start_hyperparameter_search.py --help       # Mostrar esta ayuda

Configuracion:
    Modifica las variables en la seccion CONFIGURACION segun tus necesidades
"""

import sys
import os

# Agregar path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_hyperparameter_search import main, SearchConfig, build_search_space

# =============================================================================
# CONFIGURACION PRINCIPAL
# =============================================================================

# Optimizadores a explorar
# Opciones: ['QPSO'], ['QDPSO'], o ['QPSO', 'QDPSO']
OPTIMIZERS = ['QPSO']

# Estrategias a explorar
# Opciones: ['forward'], ['weighted'], ['layerwise'], o combinaciones
STRATEGIES = ['forward', 'weighted', 'layerwise']

# Numero de trials (configuraciones a probar)
N_TRIALS = 30

# =============================================================================
# CONFIGURACION DE EXPLORACION
# =============================================================================

# Forzar exploracion inicial de todas las combinaciones?
# - False: Busqueda 100% Bayesiana (default - recomendado para exploracion optima)
# - True:  Encola 1 trial inicial por cada combinacion optimizer x strategy
#          Util para garantizar que todas las combinaciones sean evaluadas
FORCE_INITIAL_TRIALS = False

# =============================================================================
# ESPACIO DE BUSQUEDA - QPSO
# =============================================================================

# Rango para alpha inicial (QPSO)
ALPHA_START = (0.7, 1.0)

# Rango para alpha final (QPSO)
ALPHA_END = (0.3, 0.7)

# =============================================================================
# ESPACIO DE BUSQUEDA - QDPSO
# =============================================================================

# Rango para factor g (QDPSO)
G_RANGE = (0.90, 0.99)

# =============================================================================
# ESPACIO DE BUSQUEDA - ENJAMBRE
# =============================================================================

# Rango de particulas
N_PARTICLES = (20, 80)

# Rango de iteraciones maximas
MAX_ITERS = (50, 300)

# =============================================================================
# ESPACIO DE BUSQUEDA - ARQUITECTURA
# =============================================================================

# Rango de capas ocultas
N_HIDDEN_LAYERS = (1, 4)

# Rango del multiplicador de neuronas (input_dim * multiplier)
NEURONS_MULTIPLIER = (1.5, 5.0)

# Rango de decaimiento de neuronas entre capas
NEURON_DECAY = (0.6, 0.9)

# =============================================================================
# ESPACIO DE BUSQUEDA - ESTRATEGIA WEIGHTED
# =============================================================================

# Rango de decaimiento por capa
LAYER_DECAY = (0.5, 0.9)

# Rango de regularizacion
REGULARIZATION = (0.001, 0.1)

# =============================================================================
# ESPACIO DE BUSQUEDA - ESTRATEGIA LAYERWISE
# =============================================================================

# Rango de iteraciones por capa
ITERS_PER_LAYER = (20, 80)

# Rango de iteraciones de fine-tuning
FINE_TUNE_ITERS = (20, 80)

# =============================================================================
# ESPACIO DE BUSQUEDA - OTROS
# =============================================================================

# Rango de limite de pesos
WEIGHT_BOUND = (0.5, 2.0)

# Rango de paciencia (early stopping)
PATIENCE = (20, 60)

# =============================================================================
# CONFIGURACION AVANZADA
# =============================================================================

# Semilla para reproducibilidad
SEED = 42

# Timeout en segundos (None para sin limite)
TIMEOUT = None

# Ruta al dataset MCW
DATASET_PATH = './data/img/mcw'

# Directorio de salida para resultados
OUTPUT_DIR = './results/hyperparameter_search'

# =============================================================================
# CONFIGURACION DE RENDIMIENTO (FASE 1 MEJORAS)
# =============================================================================

# Paralelismo: -1 = auto (75% de cores), 1 = secuencial
N_JOBS = -1

# Dispositivo: 'auto', 'cuda', 'mps', 'cpu'
DEVICE = 'auto'

# Persistencia SQLite (permite continuar búsquedas interrumpidas)
# None = en memoria (no persistente), str = ruta al archivo .db
STORAGE_PATH = './results/hyperparameter_search/optuna_study.db'

# Continuar estudio existente si existe
RESUME_STUDY = True

# =============================================================================
# FASE 2: ESPACIO DE BUSQUEDA AMPLIADO
# =============================================================================

# Activaciones a explorar (si solo 1, se usa fija; si varias, se optimiza)
# Opciones: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'gelu'
ACTIVATIONS = ['tanh']  # Default: solo tanh (fija)
# ACTIVATIONS = ['relu', 'tanh', 'gelu']  # Ejemplo: optimizar entre varias

# Dropout (min, max). Si min == max, se usa fijo
DROPOUT_RANGE = (0.0, 0.0)  # Default: sin dropout
# DROPOUT_RANGE = (0.0, 0.5)  # Ejemplo: optimizar entre 0 y 0.5

# Batch normalization (lista de opciones a explorar)
USE_BATCH_NORM_OPTIONS = [False]  # Default: sin batch norm
# USE_BATCH_NORM_OPTIONS = [True, False]  # Ejemplo: optimizar

# Boundary strategies para QPSO
# Opciones: 'clamp', 'reflect', 'wrap', 'random'
BOUNDARY_STRATEGIES = ['clamp']  # Default: solo clamp (fija)
# BOUNDARY_STRATEGIES = ['clamp', 'reflect']  # Ejemplo: optimizar

# Tolerancia para convergencia (min, max). Escala logaritmica
TOL_RANGE = (1e-12, 1e-12)  # Default: fija en 1e-12
# TOL_RANGE = (1e-14, 1e-8)  # Ejemplo: optimizar

# =============================================================================
# FASE 2: PRUNER Y CALLBACKS
# =============================================================================

# Tipo de pruner: 'median' (conservador) o 'hyperband' (agresivo)
PRUNER_TYPE = 'median'

# Early stopping GLOBAL (detiene toda la búsqueda si no mejora)
# 0 = desactivado (default), >0 = numero de trials sin mejora para detener
EARLY_STOPPING_PATIENCE = 0  # Default: desactivado
# EARLY_STOPPING_PATIENCE = 20  # Ejemplo: detener si 20 trials sin mejora

# Frecuencia de checkpoints (cada N trials)
# 0 = desactivado, >0 = guardar checkpoint cada N trials
CHECKPOINT_FREQUENCY = 10  # Default: cada 10 trials

# =============================================================================
# NO MODIFICAR DEBAJO DE ESTA LINEA
# =============================================================================

def print_header(title: str):
    """Imprime un encabezado formateado."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title: str):
    """Imprime una seccion."""
    print(f"\n--- {title} ---")


def print_config():
    """Imprime la configuracion actual."""
    print_header("CONFIGURACION DE BUSQUEDA")

    print_section("Principal")
    print(f"  Optimizadores:    {OPTIMIZERS}")
    print(f"  Estrategias:      {STRATEGIES}")
    print(f"  Numero de trials: {N_TRIALS}")

    n_combos = len(OPTIMIZERS) * len(STRATEGIES)
    if FORCE_INITIAL_TRIALS:
        print(f"  Modo:             Hibrido ({n_combos} trial(s) inicial(es) + Bayesiano)")
    else:
        print(f"  Modo:             100% Bayesiano (TPE Sampler)")

    print_section("Espacio de Busqueda - QPSO")
    print(f"  alpha_start:      {ALPHA_START}")
    print(f"  alpha_end:        {ALPHA_END}")

    print_section("Espacio de Busqueda - QDPSO")
    print(f"  g:                {G_RANGE}")

    print_section("Espacio de Busqueda - Enjambre")
    print(f"  n_particles:      {N_PARTICLES}")
    print(f"  max_iters:        {MAX_ITERS}")

    print_section("Espacio de Busqueda - Arquitectura")
    print(f"  n_hidden_layers:  {N_HIDDEN_LAYERS}")
    print(f"  neurons_mult:     {NEURONS_MULTIPLIER}")
    print(f"  neuron_decay:     {NEURON_DECAY}")

    print_section("Espacio de Busqueda - Weighted")
    print(f"  layer_decay:      {LAYER_DECAY}")
    print(f"  regularization:   {REGULARIZATION}")

    print_section("Espacio de Busqueda - Layerwise")
    print(f"  iters_per_layer:  {ITERS_PER_LAYER}")
    print(f"  fine_tune_iters:  {FINE_TUNE_ITERS}")

    print_section("Espacio de Busqueda - Otros")
    print(f"  weight_bound:     {WEIGHT_BOUND}")
    print(f"  patience:         {PATIENCE}")

    print_section("Configuracion Avanzada")
    print(f"  Semilla:          {SEED}")
    print(f"  Dataset:          {DATASET_PATH}")
    print(f"  Salida:           {OUTPUT_DIR}")
    print(f"  Timeout:          {TIMEOUT if TIMEOUT else 'Sin limite'}")

    print_section("Rendimiento (Fase 1 Mejoras)")
    print(f"  Paralelismo:      {N_JOBS} {'(auto)' if N_JOBS == -1 else ''}")
    print(f"  Dispositivo:      {DEVICE}")
    print(f"  Storage SQLite:   {STORAGE_PATH if STORAGE_PATH else 'En memoria'}")
    print(f"  Continuar estudio:{RESUME_STUDY}")

    print_section("Fase 2: Espacio Ampliado")
    print(f"  Activaciones:     {ACTIVATIONS}")
    print(f"  Dropout:          {DROPOUT_RANGE}")
    print(f"  Batch Norm:       {USE_BATCH_NORM_OPTIONS}")
    print(f"  Boundary Strat:   {BOUNDARY_STRATEGIES}")
    print(f"  Tolerancia:       {TOL_RANGE}")

    print_section("Fase 2: Pruner y Callbacks")
    print(f"  Pruner:           {PRUNER_TYPE}")
    print(f"  Early Stop Global:{EARLY_STOPPING_PATIENCE} {'(desactivado)' if EARLY_STOPPING_PATIENCE == 0 else 'trials'}")
    print(f"  Checkpoint freq:  {CHECKPOINT_FREQUENCY} {'(desactivado)' if CHECKPOINT_FREQUENCY == 0 else 'trials'}")
    print()


def create_config() -> SearchConfig:
    """Crea la configuracion a partir de las variables globales."""
    return SearchConfig(
        # Optimizadores y estrategias
        optimizers=OPTIMIZERS,
        strategies=STRATEGIES,

        # Estudio Optuna
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        seed=SEED,
        ensure_all_combinations=FORCE_INITIAL_TRIALS,

        # Rendimiento (Fase 1 Mejoras)
        n_jobs=N_JOBS,
        device=DEVICE,
        storage_path=STORAGE_PATH,
        resume_study=RESUME_STUDY,

        # Dataset y salida
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,

        # Espacio de busqueda - QPSO
        alpha_start=ALPHA_START,
        alpha_end=ALPHA_END,

        # Espacio de busqueda - QDPSO
        g_range=G_RANGE,

        # Espacio de busqueda - Enjambre
        n_particles=N_PARTICLES,
        max_iters=MAX_ITERS,

        # Espacio de busqueda - Arquitectura
        n_hidden_layers=N_HIDDEN_LAYERS,
        neurons_multiplier=NEURONS_MULTIPLIER,
        neuron_decay=NEURON_DECAY,

        # Espacio de busqueda - Weighted
        layer_decay=LAYER_DECAY,
        regularization=REGULARIZATION,

        # Espacio de busqueda - Layerwise
        iters_per_layer=ITERS_PER_LAYER,
        fine_tune_iters=FINE_TUNE_ITERS,

        # Espacio de busqueda - Otros
        weight_bound=WEIGHT_BOUND,
        patience=PATIENCE,

        # FASE 2: Espacio de busqueda ampliado
        activations=ACTIVATIONS,
        dropout_range=DROPOUT_RANGE,
        use_batch_norm_options=USE_BATCH_NORM_OPTIONS,
        boundary_strategies=BOUNDARY_STRATEGIES,
        tol_range=TOL_RANGE,

        # FASE 2: Pruner y Callbacks
        pruner_type=PRUNER_TYPE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        checkpoint_frequency=CHECKPOINT_FREQUENCY,
    )


def run():
    """Ejecuta la busqueda de hiperparametros con la configuracion actual."""
    import torch
    from datetime import datetime

    # Mostrar configuracion
    print_config()

    # Confirmar ejecucion
    try:
        input("Presiona ENTER para continuar o Ctrl+C para cancelar...")
    except KeyboardInterrupt:
        print("\n\nCancelado por el usuario.")
        return

    print_header("INICIANDO BUSQUEDA DE HIPERPARAMETROS")

    # Importar componentes necesarios
    from main_hyperparameter_search import (
        OPTUNA_AVAILABLE,
        load_mcw,
        ensure_output_dir,
        ObjectiveFunction,
        enqueue_initial_trials,
        analyze_results_by_category,
        fill_missing_combinations,
        print_category_analysis,
        evaluate_all_combinations,
        print_combination_results,
        print_training_comparison,
        evaluate_best_config,
        save_results,
        visualize_study,
        print_header as ph,
    )
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    import time

    if not OPTUNA_AVAILABLE:
        print("ERROR: Optuna no esta instalado.")
        print("Ejecute: pip install optuna optuna-dashboard plotly kaleido")
        return

    # Crear configuracion
    config = create_config()
    config.validate()

    # Construir espacio de busqueda
    search_space = build_search_space(config)

    ph("HYPERPARAMETER SEARCH - QPSO/QDPSO")
    print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dispositivo: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Crear directorio de salida
    ensure_output_dir(config.output_dir)

    # Cargar datos
    ph("CARGANDO DATASET MCW")
    try:
        data = load_mcw(
            root_path=config.dataset_path,
            train_size=0.70,
            val_size=config.val_size,
            test_size=config.test_size,
            reduction_method=config.reduction_method,
            n_components=config.n_components,
            random_state=config.seed,
            verbose=True
        )
        n_samples = len(data.y_train) + len(data.y_val) + len(data.y_test)
        print(f"\nDataset cargado: {n_samples} muestras, {data.n_features} features")
    except Exception as e:
        print(f"Error cargando datos: {e}")
        import traceback
        traceback.print_exc()
        return

    # Configurar estudio
    ph("CONFIGURANDO ESTUDIO OPTUNA")

    n_combos = config.n_combinations
    print(f"\nConfiguracion:")
    print(f"  Nombre del estudio: {config.study_name}")
    print(f"  Numero de trials: {config.n_trials}")
    print(f"  Combinaciones a explorar: {n_combos}")
    if config.ensure_all_combinations:
        print(f"  Garantizar combinaciones: Si")
        print(f"    - {n_combos} trial(s) inicial(es) encolado(s)")
        print(f"    - Primeros {n_combos} trials protegidos del pruning")
    else:
        print(f"  Garantizar combinaciones: No")
    print(f"  Cross-validation: {config.n_folds} folds")
    print(f"  Timeout: {config.timeout or 'Sin limite'}")

    print(f"\nEspacio de busqueda:")
    print(f"  Optimizadores: {config.optimizers}")
    print(f"  Estrategias: {config.strategies}")
    print(f"  Particulas: {search_space['n_particles']}")
    print(f"  Iteraciones: {search_space['max_iters']}")
    print(f"  Capas ocultas: {search_space['n_hidden_layers']}")

    # Crear estudio
    sampler = TPESampler(seed=config.seed)
    n_startup_trials = n_combos if config.ensure_all_combinations else 5
    pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=config.study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )

    # Encolar trials iniciales si es necesario
    if config.ensure_all_combinations:
        enqueue_initial_trials(study, config)

    # Ejecutar busqueda
    ph("EJECUTANDO BUSQUEDA DE HIPERPARAMETROS")
    print(f"\nOptimizando F1-Score (macro)...")
    print(f"Esto puede tomar varios minutos/horas dependiendo de n_trials.\n")

    objective = ObjectiveFunction(data, config, search_space)
    start_time = time.time()

    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout,
        n_jobs=config.n_jobs,
        show_progress_bar=True
    )

    search_time = time.time() - start_time

    # Resultados
    ph("RESULTADOS DE LA BUSQUEDA")

    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    print(f"\nEstadisticas:")
    print(f"  Trials completados: {n_completed}")
    print(f"  Trials pruned: {n_pruned}")
    print(f"  Tiempo total: {search_time/60:.2f} minutos")

    print(f"\nMejor configuracion global encontrada:")
    print(f"  Trial: #{study.best_trial.number}")
    print(f"  F1-Score (CV): {study.best_value:.4f}")

    print(f"\nMejores hiperparametros:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Analisis por categorias
    category_analysis = analyze_results_by_category(study, config)
    category_analysis = fill_missing_combinations(category_analysis, config)
    print_category_analysis(category_analysis, config)

    # Evaluacion de combinaciones
    combination_results = evaluate_all_combinations(category_analysis, data, config)
    winner_info = print_combination_results(combination_results)
    print_training_comparison(combination_results)

    # Resultados finales
    if winner_info:
        winner_key, winner_data = winner_info
        final_results = {
            'hidden_layers': winner_data['hidden_layers'],
            'n_params': winner_data['n_params'],
            'test_accuracy': winner_data['test_accuracy'],
            'test_f1': winner_data['test_f1'],
            'test_precision': winner_data['test_precision'],
            'test_recall': winner_data['test_recall'],
            'test_kappa': winner_data['test_kappa'],
            'test_loss': winner_data['test_loss'],
            'training_time': winner_data['training_time'],
            'iterations': winner_data['iterations'],
            'convergence_reason': winner_data['convergence_reason'],
            'best_params': winner_data['params']
        }
    else:
        ph("EVALUACION DETALLADA - MEJOR CONFIGURACION GLOBAL")
        final_results = evaluate_best_config(study.best_params, data, config)

    # Guardar resultados
    ph("GUARDANDO RESULTADOS")
    save_results(
        study,
        final_results,
        config,
        config.output_dir,
        category_analysis=category_analysis,
        combination_results=combination_results
    )
    visualize_study(study, config.output_dir)

    # Resumen final
    ph("RESUMEN FINAL")
    if winner_info:
        winner_key, winner = winner_info
        print(f"\n{'='*70}")
        print(f" GANADOR ABSOLUTO (Mejor F1-Score en Test)")
        print(f"{'='*70}")
        print(f"\n  Combinacion: {winner['optimizer']} + {winner['strategy'].upper()}")
        print(f"  Trial: #{winner['trial_number']}")
        print(f"\n  --- Validacion Cruzada ---")
        print(f"  F1-Score (CV): {winner['f1_score_cv']:.4f}")
        print(f"\n  --- Conjunto de Test ---")
        print(f"  F1-Score: {winner['test_f1']:.4f}")
        print(f"  Accuracy: {winner['test_accuracy']:.4f}")
        print(f"\n  --- Arquitectura ---")
        print(f"  Capas ocultas: {winner['hidden_layers']}")
        print(f"  Parametros: {winner['n_params']:,}")

    ph("BUSQUEDA COMPLETADA")
    print(f"\nResultados guardados en: {config.output_dir}\n")


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        print("\nPara ver opciones de linea de comandos, ejecute:")
        print("  python main_hyperparameter_search.py --help")
    else:
        run()
