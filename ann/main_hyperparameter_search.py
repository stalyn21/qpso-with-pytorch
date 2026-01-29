"""
Hyperparameter Search: Busqueda de Configuracion Optima para QPSO/QDPSO

Este script utiliza Optuna para encontrar la mejor configuracion de hiperparametros
para entrenar redes neuronales con QPSO/QDPSO en el dataset MCW.

Hiperparametros optimizados:
    - Optimizador: QPSO vs QDPSO
    - Parametros del optimizador: alpha (QPSO), g (QDPSO)
    - Enjambre: n_particles
    - Arquitectura: n_layers, neurons_per_layer
    - Estrategia: forward, weighted, layerwise
    - Parametros de estrategia: layer_decay, regularization, iters_per_layer

Por que Optuna?
    1. Busqueda Bayesiana: Aprende de evaluaciones previas (TPE sampler)
    2. Pruning: Detiene configuraciones malas temprano (MedianPruner)
    3. Eficiencia: Mas inteligente que grid search o random search
    4. Visualizacion: Importancia de parametros, historial, etc.

Requisitos:
    pip install optuna optuna-dashboard plotly kaleido
    conda activate pytorch_qpso_gpu
    python main_hyperparameter_search.py

Autor: QPSO-PyTorch Team
"""

import argparse
import torch
import numpy as np
import time
import sys
import os
import json
import warnings
import logging
import traceback as tb_module
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Agregar path para imports (portable: funciona sin importar el nombre del folder)
sys.path.insert(0, os.path.dirname(__file__))

# Optuna
try:
    import optuna
    from optuna.trial import Trial
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: Optuna no esta instalado. Ejecute: pip install optuna")

# Imports del proyecto
from models import QPSOCompatibleANN
from optimizers.training_strategies import (
    create_training_strategy,
    StrategyConfig,
)
from utils import MulticlassMetrics
from data import load_mcw

# Sklearn
from sklearn.model_selection import StratifiedKFold

# Suprimir warnings específicos (no todos)
warnings.filterwarnings('ignore', category=UserWarning, module='optuna')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
warnings.filterwarnings('ignore', message='.*overflow.*', category=RuntimeWarning)
# Suprimir warnings de sklearn cuando no hay predicciones para alguna clase
# (normal durante exploración inicial de QPSO)
warnings.filterwarnings('ignore', message='.*Precision is ill-defined.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*UndefinedMetricWarning.*', category=UserWarning)

# =============================================================================
# CONFIGURACION DE LOGGING
# =============================================================================

def setup_logging(output_dir: str, verbose: bool = True) -> logging.Logger:
    """
    Configura el sistema de logging para HPO.

    Args:
        output_dir: Directorio donde guardar los logs
        verbose: Si True, también muestra logs en consola

    Returns:
        Logger configurado
    """
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger('hpo')
    logger.setLevel(logging.DEBUG)

    # Limpiar handlers previos
    logger.handlers.clear()

    # Handler para archivo (todos los niveles)
    file_handler = logging.FileHandler(
        os.path.join(output_dir, 'hpo_search.log'),
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)

    # Handler para consola (solo warnings y errores)
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(console_handler)

    return logger

# Logger global (se inicializa en main)
logger: Optional[logging.Logger] = None


# =============================================================================
# DETECCION DE DISPOSITIVO (GPU/MPS/CPU)
# =============================================================================

def get_optimal_device() -> str:
    """
    Detecta el mejor dispositivo disponible para computación.

    Orden de preferencia: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        String con el dispositivo: 'cuda', 'mps', o 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Verificar que MPS realmente funcione
        try:
            torch.zeros(1).to('mps')
            return 'mps'
        except Exception:
            pass
    return 'cpu'


def get_device_info(device: str) -> str:
    """
    Obtiene información detallada del dispositivo.

    Args:
        device: Dispositivo ('cuda', 'mps', 'cpu')

    Returns:
        String con información del dispositivo
    """
    if device == 'cuda':
        return f"CUDA - {torch.cuda.get_device_name(0)}"
    elif device == 'mps':
        return "MPS - Apple Silicon"
    else:
        import platform
        return f"CPU - {platform.processor() or 'Unknown'}"


# =============================================================================
# CONFIGURACION DEL ESTUDIO
# =============================================================================

# =============================================================================
# OPCIONES DISPONIBLES
# =============================================================================

AVAILABLE_OPTIMIZERS = ['QPSO', 'QDPSO']
AVAILABLE_STRATEGIES = ['forward', 'weighted', 'layerwise']
AVAILABLE_ACTIVATIONS = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'gelu']
AVAILABLE_BOUNDARY_STRATEGIES = ['clamp', 'reflect', 'wrap', 'random']
AVAILABLE_PRUNERS = ['median', 'hyperband']

# Parametros por defecto para trials iniciales
DEFAULT_OPTIMIZER_PARAMS = {
    'QPSO': {'alpha_start': 0.9, 'alpha_end': 0.5},
    'QDPSO': {'g': 0.96}
}

DEFAULT_STRATEGY_PARAMS = {
    'forward': {},
    'weighted': {'layer_decay': 0.7, 'regularization': 0.01},
    'layerwise': {'iters_per_layer': 50, 'fine_tune_iters': 50}
}


@dataclass
class SearchConfig:
    """Configuracion para la busqueda de hiperparametros."""

    # Optimizadores y estrategias a explorar
    optimizers: List[str] = field(default_factory=lambda: ['QPSO'])
    strategies: List[str] = field(default_factory=lambda: ['forward'])

    # Dataset
    dataset_path: str = './data/img/mcw'
    reduction_method: str = 'isomap'
    n_components: int = 7

    # Estudio Optuna
    study_name: str = 'qpso_mcw_optimization'
    n_trials: int = 30               # Numero de configuraciones a probar
    timeout: Optional[int] = None    # Tiempo maximo en segundos (None = sin limite)
    n_jobs: int = -1                 # Paralelismo (-1 = auto-detectar cores, 1 = secuencial)
    ensure_all_combinations: bool = True  # Garantizar exploracion de todas las combinaciones

    # Persistencia del estudio (SQLite)
    storage_path: Optional[str] = None  # None = en memoria, str = ruta a SQLite
    resume_study: bool = True        # Si True, continua estudio existente

    # Dispositivo
    device: str = 'auto'             # 'auto', 'cuda', 'mps', 'cpu'

    # Validacion
    n_folds: int = 4                 # Folds para cross-validation
    val_size: float = 0.15           # Tamaño de validacion
    test_size: float = 0.15          # Tamaño de test (holdout final)

    # Reproducibilidad
    seed: int = 21

    # Salida
    output_dir: str = './results/hyperparameter_search'
    save_top_k: int = 10             # Guardar top K configuraciones

    # =========================================================================
    # ESPACIO DE BUSQUEDA
    # =========================================================================

    # Parametros QPSO
    alpha_start: Tuple[float, float] = (0.7, 1.0)
    alpha_end: Tuple[float, float] = (0.3, 0.7)

    # Parametros QDPSO
    g_range: Tuple[float, float] = (0.90, 0.99)

    # Enjambre
    n_particles: Tuple[int, int] = (20, 80)
    max_iters: Tuple[int, int] = (50, 300)

    # Arquitectura
    n_hidden_layers: Tuple[int, int] = (1, 3)
    neurons_multiplier: Tuple[float, float] = (1.5, 4.0)
    neuron_decay: Tuple[float, float] = (0.5, 0.9)

    # Parametros Weighted
    layer_decay: Tuple[float, float] = (0.5, 0.9)
    regularization: Tuple[float, float] = (0.001, 0.1)

    # Parametros Layerwise
    iters_per_layer: Tuple[int, int] = (20, 80)
    fine_tune_iters: Tuple[int, int] = (20, 80)

    # Otros
    weight_bound: Tuple[float, float] = (0.5, 2.0)
    patience: Tuple[int, int] = (20, 60)

    # =========================================================================
    # FASE 2: ESPACIO DE BUSQUEDA AMPLIADO
    # =========================================================================

    # Activaciones a explorar (si solo 1, se usa fija; si varias, se optimiza)
    activations: List[str] = field(default_factory=lambda: ['tanh'])

    # Dropout (0.0 = sin dropout)
    dropout_range: Tuple[float, float] = (0.0, 0.0)  # Default: sin dropout

    # Batch normalization
    use_batch_norm_options: List[bool] = field(default_factory=lambda: [False])

    # Boundary strategies para QPSO
    boundary_strategies: List[str] = field(default_factory=lambda: ['clamp'])

    # Tolerancia para convergencia (escala log)
    tol_range: Tuple[float, float] = (1e-12, 1e-12)  # Default: fijo en 1e-12

    # =========================================================================
    # FASE 2: PRUNER Y CALLBACKS
    # =========================================================================

    # Tipo de pruner: 'median' (conservador) o 'hyperband' (agresivo)
    pruner_type: str = 'median'

    # Early stopping GLOBAL (detiene toda la búsqueda si no mejora)
    # Desactivado por defecto (0 = desactivado)
    early_stopping_patience: int = 0  # 0 = desactivado, >0 = trials sin mejora

    # Frecuencia de checkpoints (cada N trials)
    checkpoint_frequency: int = 10  # 0 = desactivado

    @property
    def n_combinations(self) -> int:
        """Numero total de combinaciones optimizer x strategy."""
        return len(self.optimizers) * len(self.strategies)

    def validate(self):
        """Valida la configuración completa."""
        # Validar optimizadores
        for opt in self.optimizers:
            if opt not in AVAILABLE_OPTIMIZERS:
                raise ValueError(f"Optimizador invalido: {opt}. Opciones: {AVAILABLE_OPTIMIZERS}")

        # Validar estrategias
        for strat in self.strategies:
            if strat not in AVAILABLE_STRATEGIES:
                raise ValueError(f"Estrategia invalida: {strat}. Opciones: {AVAILABLE_STRATEGIES}")

        # Validar activaciones (Fase 2)
        for act in self.activations:
            if act not in AVAILABLE_ACTIVATIONS:
                raise ValueError(f"Activacion invalida: {act}. Opciones: {AVAILABLE_ACTIVATIONS}")

        # Validar boundary strategies (Fase 2)
        for bs in self.boundary_strategies:
            if bs not in AVAILABLE_BOUNDARY_STRATEGIES:
                raise ValueError(f"Boundary strategy invalida: {bs}. Opciones: {AVAILABLE_BOUNDARY_STRATEGIES}")

        # Validar pruner (Fase 2)
        if self.pruner_type not in AVAILABLE_PRUNERS:
            raise ValueError(f"Pruner invalido: {self.pruner_type}. Opciones: {AVAILABLE_PRUNERS}")

        # Validar rangos (min < max)
        range_params = [
            ('alpha_start', self.alpha_start),
            ('alpha_end', self.alpha_end),
            ('g_range', self.g_range),
            ('n_particles', self.n_particles),
            ('max_iters', self.max_iters),
            ('n_hidden_layers', self.n_hidden_layers),
            ('neurons_multiplier', self.neurons_multiplier),
            ('neuron_decay', self.neuron_decay),
            ('layer_decay', self.layer_decay),
            ('regularization', self.regularization),
            ('iters_per_layer', self.iters_per_layer),
            ('fine_tune_iters', self.fine_tune_iters),
            ('weight_bound', self.weight_bound),
            ('patience', self.patience),
            ('dropout_range', self.dropout_range),
            ('tol_range', self.tol_range),
        ]

        for name, (min_val, max_val) in range_params:
            if min_val > max_val:
                raise ValueError(f"Rango invalido para {name}: min ({min_val}) > max ({max_val})")

        # Validar restricción alpha_start > alpha_end (Fase 2)
        # El mínimo de alpha_start debe ser >= mínimo de alpha_end
        if self.alpha_start[0] < self.alpha_end[0]:
            if logger:
                logger.warning(
                    f"Advertencia: alpha_start min ({self.alpha_start[0]}) < alpha_end min ({self.alpha_end[0]}). "
                    "Esto puede generar configuraciones donde alpha aumenta."
                )

        # Validar valores positivos
        if self.n_trials < 1:
            raise ValueError("n_trials debe ser >= 1")
        if self.n_folds < 2:
            raise ValueError("n_folds debe ser >= 2")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience debe ser >= 0")
        if self.checkpoint_frequency < 0:
            raise ValueError("checkpoint_frequency debe ser >= 0")


# =============================================================================
# CALLBACKS DE OPTUNA (FASE 2)
# =============================================================================

class CheckpointCallback:
    """
    Callback para guardar checkpoints periodicos durante la busqueda.

    Guarda el estado actual del estudio cada N trials completados.
    """

    def __init__(self, output_dir: str, frequency: int = 10):
        """
        Args:
            output_dir: Directorio donde guardar checkpoints
            frequency: Frecuencia de checkpoints (cada N trials). 0 = desactivado.
        """
        self.output_dir = output_dir
        self.frequency = frequency
        self._last_checkpoint = 0

    def __call__(self, study: 'optuna.Study', trial: 'optuna.Trial') -> None:
        if self.frequency <= 0:
            return

        n_complete = len([t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE])

        if n_complete > 0 and n_complete % self.frequency == 0 and n_complete != self._last_checkpoint:
            self._last_checkpoint = n_complete
            checkpoint_path = os.path.join(
                self.output_dir,
                f'checkpoint_trial_{n_complete}.json'
            )
            try:
                checkpoint_data = {
                    'n_trials_complete': n_complete,
                    'best_value': study.best_value if study.best_trial else None,
                    'best_params': study.best_params if study.best_trial else None,
                    'best_trial_number': study.best_trial.number if study.best_trial else None,
                    'timestamp': datetime.now().isoformat()
                }
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                if logger:
                    logger.info(f"Checkpoint guardado: {checkpoint_path}")
            except Exception as e:
                if logger:
                    logger.warning(f"Error guardando checkpoint: {e}")


class EarlyStoppingCallback:
    """
    Callback para detener la busqueda si no hay mejora en N trials.

    IMPORTANTE: Desactivado por defecto (patience=0).
    Solo se activa si patience > 0.
    """

    def __init__(self, patience: int = 0, min_delta: float = 0.0001):
        """
        Args:
            patience: Numero de trials sin mejora antes de detener. 0 = desactivado.
            min_delta: Mejora minima para considerar que hubo progreso.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: Optional[float] = None
        self.trials_without_improvement = 0
        self._enabled = patience > 0

    def __call__(self, study: 'optuna.Study', trial: 'optuna.Trial') -> None:
        if not self._enabled:
            return

        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        if trial.value is None:
            return

        current_value = trial.value

        if self.best_value is None:
            self.best_value = current_value
            self.trials_without_improvement = 0
            return

        # Verificar si hubo mejora (asumiendo maximización)
        if current_value > self.best_value + self.min_delta:
            self.best_value = current_value
            self.trials_without_improvement = 0
            if logger:
                logger.info(f"Nueva mejor configuración encontrada: {current_value:.6f}")
        else:
            self.trials_without_improvement += 1
            if logger:
                logger.debug(f"Sin mejora: {self.trials_without_improvement}/{self.patience}")

        # Detener si se alcanzó la paciencia
        if self.trials_without_improvement >= self.patience:
            if logger:
                logger.warning(
                    f"Early stopping GLOBAL activado: {self.patience} trials sin mejora. "
                    f"Mejor valor: {self.best_value:.6f}"
                )
            print(f"\n[Early Stopping] Deteniendo búsqueda: {self.patience} trials sin mejora")
            print(f"  Mejor valor alcanzado: {self.best_value:.6f}")
            study.stop()


def enqueue_initial_trials(study: 'optuna.Study', config: SearchConfig):
    """
    Encola trials iniciales para garantizar que todas las combinaciones sean exploradas.

    Crea un trial inicial por cada combinacion optimizer x strategy
    con parametros razonables dentro del espacio de busqueda.

    Args:
        study: Estudio de Optuna
        config: Configuracion del estudio
    """
    # Parametros base comunes
    base_params = {
        'n_particles': 50,
        'max_iters': 150,
        'n_hidden_layers': 2,
        'neurons_multiplier': 2.5,
        'neuron_decay': 0.7,
        'weight_bound': 1.0,
        'patience': 40,
    }

    n_combos = config.n_combinations
    print(f"\nEncolando {n_combos} trial(s) inicial(es) para garantizar exploracion completa:")

    for optimizer in config.optimizers:
        for strategy in config.strategies:
            # Construir parametros para este trial
            trial_params = base_params.copy()
            trial_params['optimizer'] = optimizer
            trial_params['strategy'] = strategy

            # Agregar parametros del optimizador
            trial_params.update(DEFAULT_OPTIMIZER_PARAMS[optimizer])

            # Agregar parametros de la estrategia
            trial_params.update(DEFAULT_STRATEGY_PARAMS[strategy])

            # Encolar el trial
            study.enqueue_trial(trial_params)
            print(f"  - Encolado: {optimizer} + {strategy.upper()}")

    print(f"\nTotal encolados: {n_combos} trial(s)")
    print("Estos trials se ejecutaran primero antes de la exploracion Bayesiana.\n")


def build_search_space(config: SearchConfig) -> Dict[str, Any]:
    """
    Construye el espacio de busqueda basado en la configuracion.

    Args:
        config: Configuracion del estudio

    Returns:
        Diccionario con el espacio de busqueda
    """
    return {
        # Optimizador (segun configuracion)
        'optimizer': config.optimizers,

        # Parametros QPSO
        'alpha_start': config.alpha_start,
        'alpha_end': config.alpha_end,

        # Parametros QDPSO
        'g': config.g_range,

        # Enjambre
        'n_particles': config.n_particles,
        'max_iters': config.max_iters,

        # Arquitectura
        'n_hidden_layers': config.n_hidden_layers,
        'neurons_multiplier': config.neurons_multiplier,
        'neuron_decay': config.neuron_decay,

        # Estrategia de entrenamiento (segun configuracion)
        'strategy': config.strategies,

        # Parametros Weighted
        'layer_decay': config.layer_decay,
        'regularization': config.regularization,

        # Parametros Layerwise
        'iters_per_layer': config.iters_per_layer,
        'fine_tune_iters': config.fine_tune_iters,

        # Otros
        'weight_bound': config.weight_bound,
        'patience': config.patience,

        # =====================================================================
        # FASE 2: Espacio de busqueda ampliado
        # =====================================================================

        # Activaciones (si solo 1, se usa fija; si varias, se optimiza)
        'activations': config.activations,

        # Dropout
        'dropout': config.dropout_range,

        # Batch normalization
        'use_batch_norm': config.use_batch_norm_options,

        # Boundary strategy para QPSO
        'boundary_strategy': config.boundary_strategies,

        # Tolerancia (si min == max, se usa fija)
        'tol': config.tol_range,
    }


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def print_header(title: str, char: str = "="):
    """Imprime encabezado formateado."""
    width = 70
    print("\n" + char * width)
    print(f" {title}")
    print(char * width)


def ensure_output_dir(path: str):
    """Crea directorio de salida."""
    os.makedirs(path, exist_ok=True)


def generate_architecture(
    input_dim: int,
    n_layers: int,
    multiplier: float,
    decay: float
) -> List[int]:
    """
    Genera arquitectura de red neuronal.

    Args:
        input_dim: Dimension de entrada
        n_layers: Numero de capas ocultas
        multiplier: Multiplicador para primera capa (input_dim * multiplier)
        decay: Factor de decaimiento entre capas

    Returns:
        Lista con neuronas por capa [h1, h2, ...]
    """
    layers = []
    neurons = int(input_dim * multiplier)

    for i in range(n_layers):
        layers.append(max(8, neurons))  # Minimo 8 neuronas
        neurons = int(neurons * decay)

    return layers


# =============================================================================
# FUNCION OBJETIVO PARA OPTUNA
# =============================================================================

class ObjectiveFunction:
    """
    Funcion objetivo para Optuna.

    Encapsula la logica de evaluacion de una configuracion de hiperparametros.
    Soporta ejecución paralela thread-safe.
    """

    def __init__(self, data, config: SearchConfig, search_space: Dict[str, Any]):
        """
        Inicializa la funcion objetivo.

        Args:
            data: MCWDataResult con los datos cargados
            config: Configuracion del estudio
            search_space: Espacio de busqueda de hiperparametros
        """
        self.data = data
        self.config = config
        self.search_space = search_space
        self.input_dim = data.n_features
        self.output_dim = data.n_classes

        # Detectar dispositivo óptimo
        if config.device == 'auto':
            self.device = get_optimal_device()
        else:
            self.device = config.device

        # Lock para acceso thread-safe a GPU (necesario para n_jobs > 1)
        # MPS y CUDA no son thread-safe por defecto
        self._device_lock = threading.Lock() if self.device in ('cuda', 'mps') else None

        # Preparar datos
        self.X = np.vstack([data.X_train, data.X_val])
        self.y = np.concatenate([data.y_train, data.y_val])
        self.X_test = data.X_test
        self.y_test = data.y_test

        # Pre-convertir a tensores para eficiencia
        self.X_tensor = torch.tensor(self.X, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y, dtype=torch.long)

    def __call__(self, trial: Trial) -> float:
        """
        Evalua una configuracion de hiperparametros.

        Args:
            trial: Trial de Optuna con los hiperparametros sugeridos

        Returns:
            Metrica a optimizar (F1-score macro)
        """
        ss = self.search_space  # Alias para brevedad

        # =====================================================================
        # 1. Sugerir hiperparametros
        # =====================================================================

        # Optimizador
        optimizer_name = trial.suggest_categorical('optimizer', ss['optimizer'])

        # Parametros del optimizador
        if optimizer_name == 'QPSO':
            alpha_start = trial.suggest_float('alpha_start', *ss['alpha_start'])
            alpha_end = trial.suggest_float('alpha_end', *ss['alpha_end'])
            alpha = (alpha_start, alpha_end)
            g = 0.96  # No usado
        else:
            alpha = (1.0, 0.5)  # No usado
            g = trial.suggest_float('g', *ss['g'])

        # Enjambre
        n_particles = trial.suggest_int('n_particles', *ss['n_particles'])
        max_iters = trial.suggest_int('max_iters', *ss['max_iters'])

        # Arquitectura
        n_hidden_layers = trial.suggest_int('n_hidden_layers', *ss['n_hidden_layers'])
        neurons_multiplier = trial.suggest_float('neurons_multiplier', *ss['neurons_multiplier'])
        neuron_decay = trial.suggest_float('neuron_decay', *ss['neuron_decay'])

        hidden_layers = generate_architecture(
            self.input_dim, n_hidden_layers, neurons_multiplier, neuron_decay
        )

        # Estrategia
        strategy_name = trial.suggest_categorical('strategy', ss['strategy'])

        # Parametros de estrategia
        strategy_params = {}
        if strategy_name == 'weighted':
            strategy_params['layer_decay'] = trial.suggest_float(
                'layer_decay', *ss['layer_decay']
            )
            strategy_params['regularization'] = trial.suggest_float(
                'regularization', *ss['regularization'], log=True
            )
        elif strategy_name == 'layerwise':
            strategy_params['iters_per_layer'] = trial.suggest_int(
                'iters_per_layer', *ss['iters_per_layer']
            )
            strategy_params['fine_tune_iters'] = trial.suggest_int(
                'fine_tune_iters', *ss['fine_tune_iters']
            )

        # Otros parametros
        weight_bound = trial.suggest_float('weight_bound', *ss['weight_bound'])
        patience = trial.suggest_int('patience', *ss['patience'])

        # =====================================================================
        # FASE 2: Parametros ampliados
        # =====================================================================

        # Activacion (si solo hay una opcion, usar fija)
        if len(ss['activations']) == 1:
            activation = ss['activations'][0]
        else:
            activation = trial.suggest_categorical('activation', ss['activations'])

        # Dropout (si min == max, usar fijo)
        dropout_min, dropout_max = ss['dropout']
        if dropout_min == dropout_max:
            dropout = dropout_min
        else:
            dropout = trial.suggest_float('dropout', dropout_min, dropout_max)

        # Batch normalization (si solo hay una opcion, usar fija)
        if len(ss['use_batch_norm']) == 1:
            use_batch_norm = ss['use_batch_norm'][0]
        else:
            use_batch_norm = trial.suggest_categorical('use_batch_norm', ss['use_batch_norm'])

        # Boundary strategy (si solo hay una opcion, usar fija)
        if len(ss['boundary_strategy']) == 1:
            boundary_strategy = ss['boundary_strategy'][0]
        else:
            boundary_strategy = trial.suggest_categorical('boundary_strategy', ss['boundary_strategy'])

        # Tolerancia (si min == max, usar fija; sino usar escala log)
        tol_min, tol_max = ss['tol']
        if tol_min == tol_max:
            tol = tol_min
        else:
            tol = trial.suggest_float('tol', tol_min, tol_max, log=True)

        # =====================================================================
        # 2. Cross-validation
        # =====================================================================

        kfold = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.seed
        )

        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(self.X, self.y)):
            # Dividir datos usando tensores pre-convertidos (más eficiente)
            X_train_t = self.X_tensor[train_idx]
            y_train_t = self.y_tensor[train_idx]
            X_val_t = self.X_tensor[val_idx]
            y_val_t = self.y_tensor[val_idx]
            y_val_fold = self.y[val_idx]  # NumPy para métricas

            # Crear modelo (Fase 2: con activation, dropout, batch_norm)
            model = QPSOCompatibleANN(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_layers=hidden_layers,
                activation=activation,
                output_activation='softmax',
                dropout=dropout,
                use_batch_norm=use_batch_norm
            )

            # Configurar estrategia (Fase 2: con boundary_strategy y tol)
            strategy_config = StrategyConfig(
                n_particles=n_particles,
                max_iters=max_iters,
                alpha=alpha,
                g=g,
                weight_bound=weight_bound,
                patience=patience,
                seed=self.config.seed + fold_idx,
                boundary_strategy=boundary_strategy,
                tol=tol,
                **strategy_params
            )

            # Crear estrategia
            strategy = create_training_strategy(
                model=model,
                strategy=strategy_name,
                config=strategy_config,
                use_qdpso=(optimizer_name == 'QDPSO'),
                device=self.device
            )

            # Entrenar
            strategy.set_data(X_train_t, y_train_t, X_val_t, y_val_t)

            try:
                # Usar lock si hay GPU para thread-safety
                if self._device_lock:
                    with self._device_lock:
                        result = strategy.train(verbose=False)
                else:
                    result = strategy.train(verbose=False)

            except Exception as e:
                # Logging detallado del error
                error_msg = f"Trial {trial.number}, Fold {fold_idx} falló: {str(e)}"
                error_tb = tb_module.format_exc()

                if logger:
                    logger.error(error_msg)
                    logger.debug(f"Traceback:\n{error_tb}")

                # Guardar error en el trial para análisis posterior
                trial.set_user_attr('error', str(e))
                trial.set_user_attr('error_traceback', error_tb)
                trial.set_user_attr('error_fold', fold_idx)
                trial.set_user_attr('failed_config', {
                    'optimizer': optimizer_name,
                    'strategy': strategy_name,
                    'n_particles': n_particles,
                    'hidden_layers': hidden_layers
                })

                # Retornar score bajo
                return 0.0

            # Evaluar en validacion (con lock si hay GPU)
            try:
                if self._device_lock:
                    with self._device_lock:
                        with torch.no_grad():
                            model.to(self.device)
                            X_val_device = X_val_t.to(self.device)
                            output = model(X_val_device)
                            preds = output.argmax(dim=1).cpu().numpy()
                else:
                    with torch.no_grad():
                        model.to(self.device)
                        X_val_device = X_val_t.to(self.device)
                        output = model(X_val_device)
                        preds = output.argmax(dim=1).cpu().numpy()

            except Exception as e:
                if logger:
                    logger.error(f"Trial {trial.number}, Fold {fold_idx} - Error en evaluación: {e}")
                return 0.0

            # Calcular F1-score
            metrics_calc = MulticlassMetrics()
            metrics = metrics_calc.calculate_all_metrics(y_val_fold, preds)
            fold_scores.append(metrics['f1_score']['macro'])

            # Pruning: reportar score intermedio
            trial.report(np.mean(fold_scores), fold_idx)

            # Verificar si debe detenerse
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Guardar métricas del trial exitoso
        trial.set_user_attr('fold_scores', fold_scores)
        trial.set_user_attr('std_f1', float(np.std(fold_scores)))

        # Retornar promedio de folds
        return np.mean(fold_scores)


# =============================================================================
# EVALUACION FINAL DEL MEJOR MODELO
# =============================================================================

def evaluate_best_config(
    best_params: Dict[str, Any],
    data,
    config: SearchConfig
) -> Dict[str, Any]:
    """
    Evalua la mejor configuracion en el conjunto de test.

    Args:
        best_params: Mejores hiperparametros encontrados
        data: Datos MCW
        config: Configuracion del estudio

    Returns:
        Diccionario con resultados detallados
    """
    print_header("EVALUACION FINAL - MEJOR CONFIGURACION")

    # Usar dispositivo configurado
    if config.device == 'auto':
        device = get_optimal_device()
    else:
        device = config.device

    input_dim = data.n_features
    output_dim = data.n_classes

    # Reconstruir arquitectura
    hidden_layers = generate_architecture(
        input_dim,
        best_params['n_hidden_layers'],
        best_params['neurons_multiplier'],
        best_params['neuron_decay']
    )

    # Preparar datos
    X_train = np.vstack([data.X_train, data.X_val])
    y_train = np.concatenate([data.y_train, data.y_val])

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(data.X_test, dtype=torch.float32)
    y_test_t = torch.tensor(data.y_test, dtype=torch.long)

    # Crear modelo
    model = QPSOCompatibleANN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=hidden_layers,
        activation='tanh',
        output_activation='softmax'
    )

    print(f"\nArquitectura: {input_dim} -> {hidden_layers} -> {output_dim}")
    print(f"Parametros: {model.num_params:,}")

    # Configurar estrategia
    optimizer_name = best_params['optimizer']
    strategy_name = best_params['strategy']

    if optimizer_name == 'QPSO':
        alpha = (best_params['alpha_start'], best_params['alpha_end'])
        g = 0.96
    else:
        alpha = (1.0, 0.5)
        g = best_params['g']

    strategy_params = {}
    if strategy_name == 'weighted':
        strategy_params['layer_decay'] = best_params['layer_decay']
        strategy_params['regularization'] = best_params['regularization']
    elif strategy_name == 'layerwise':
        strategy_params['iters_per_layer'] = best_params['iters_per_layer']
        strategy_params['fine_tune_iters'] = best_params['fine_tune_iters']

    strategy_config = StrategyConfig(
        n_particles=best_params['n_particles'],
        max_iters=best_params['max_iters'],
        alpha=alpha,
        g=g,
        weight_bound=best_params['weight_bound'],
        patience=best_params['patience'],
        seed=config.seed,
        **strategy_params
    )

    strategy = create_training_strategy(
        model=model,
        strategy=strategy_name,
        config=strategy_config,
        use_qdpso=(optimizer_name == 'QDPSO'),
        device=device
    )

    # Entrenar con todos los datos de train
    print(f"\nEntrenando modelo final con {optimizer_name} + {strategy_name.upper()}...")

    # Usar parte para validacion interna
    split_idx = int(len(y_train) * 0.85)
    X_tr = X_train_t[:split_idx]
    y_tr = y_train_t[:split_idx]
    X_vl = X_train_t[split_idx:]
    y_vl = y_train_t[split_idx:]

    strategy.set_data(X_tr, y_tr, X_vl, y_vl)

    start_time = time.time()
    result = strategy.train(verbose=True)
    training_time = time.time() - start_time

    # Evaluar en test
    print("\nEvaluando en conjunto de test...")

    with torch.no_grad():
        model.to(device)
        X_test_t = X_test_t.to(device)
        output = model(X_test_t)
        test_preds = output.argmax(dim=1).cpu().numpy()
        test_loss = torch.nn.CrossEntropyLoss()(output, y_test_t.to(device)).item()

    # Metricas detalladas
    metrics_calc = MulticlassMetrics()
    detailed_metrics = metrics_calc.calculate_all_metrics(data.y_test, test_preds)

    print(f"\n--- Resultados en Test ---")
    print(f"  Accuracy: {detailed_metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {detailed_metrics['precision']['macro']:.4f}")
    print(f"  Recall (macro): {detailed_metrics['recall']['macro']:.4f}")
    print(f"  F1-Score (macro): {detailed_metrics['f1_score']['macro']:.4f}")
    print(f"  Cohen's Kappa: {detailed_metrics['cohen_kappa']:.4f}")
    print(f"  Loss: {test_loss:.6f}")
    print(f"  Tiempo: {training_time:.2f}s")

    return {
        'best_params': best_params,
        'hidden_layers': hidden_layers,
        'n_params': model.num_params,
        'test_accuracy': detailed_metrics['accuracy'],
        'test_f1': detailed_metrics['f1_score']['macro'],
        'test_precision': detailed_metrics['precision']['macro'],
        'test_recall': detailed_metrics['recall']['macro'],
        'test_kappa': detailed_metrics['cohen_kappa'],
        'test_loss': test_loss,
        'training_time': training_time,
        'iterations': result.iterations,
        'convergence_reason': result.convergence_reason,
        'detailed_metrics': detailed_metrics
    }


# =============================================================================
# ANALISIS POR CATEGORIAS
# =============================================================================

def analyze_results_by_category(study: 'optuna.Study', config: SearchConfig) -> Dict[str, Any]:
    """
    Analiza los resultados agrupados por optimizador, estrategia, activacion y combinacion.

    Args:
        study: Estudio completado de Optuna
        config: Configuracion del estudio

    Returns:
        Diccionario con analisis por categoria:
        - by_optimizer: Mejor trial por cada optimizador
        - by_strategy: Mejor trial por cada estrategia
        - by_activation: Mejor trial por cada funcion de activacion
        - by_activation_strategy: Mejor trial por cada combinacion activation x strategy
        - by_combination: Mejor trial por cada combinacion optimizer x strategy
        - ranking: Lista ordenada de combinaciones de mejor a peor
    """
    # Filtrar solo trials completados con valor valido
    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]

    if not completed_trials:
        return {
            'by_optimizer': {},
            'by_strategy': {},
            'by_activation': {},
            'by_activation_strategy': {},
            'by_combination': {},
            'ranking': []
        }

    # Agrupar por optimizador
    by_optimizer = {}
    for optimizer in config.optimizers:
        trials_opt = [t for t in completed_trials if t.params.get('optimizer') == optimizer]
        if trials_opt:
            best = max(trials_opt, key=lambda t: t.value)
            by_optimizer[optimizer] = {
                'trial_number': best.number,
                'f1_score_cv': best.value,
                'params': best.params,
                'strategy': best.params.get('strategy'),
                'n_trials': len(trials_opt)
            }

    # Agrupar por estrategia
    by_strategy = {}
    for strategy in config.strategies:
        trials_strat = [t for t in completed_trials if t.params.get('strategy') == strategy]
        if trials_strat:
            best = max(trials_strat, key=lambda t: t.value)
            by_strategy[strategy] = {
                'trial_number': best.number,
                'f1_score_cv': best.value,
                'params': best.params,
                'optimizer': best.params.get('optimizer'),
                'n_trials': len(trials_strat)
            }

    # Agrupar por funcion de activacion (Fase 2)
    by_activation = {}
    # Obtener todas las activaciones usadas en los trials
    activations_used = set(t.params.get('activation', 'tanh') for t in completed_trials)
    for activation in activations_used:
        trials_act = [t for t in completed_trials if t.params.get('activation', 'tanh') == activation]
        if trials_act:
            best = max(trials_act, key=lambda t: t.value)
            by_activation[activation] = {
                'trial_number': best.number,
                'f1_score_cv': best.value,
                'params': best.params,
                'strategy': best.params.get('strategy'),
                'optimizer': best.params.get('optimizer'),
                'n_trials': len(trials_act)
            }

    # Agrupar por combinacion (activation x strategy) - Para referencia futura
    by_activation_strategy = {}
    for activation in activations_used:
        for strategy in config.strategies:
            combo_key = f"{activation}_{strategy}"
            trials_combo = [
                t for t in completed_trials
                if t.params.get('activation', 'tanh') == activation and t.params.get('strategy') == strategy
            ]
            if trials_combo:
                best = max(trials_combo, key=lambda t: t.value)
                by_activation_strategy[combo_key] = {
                    'activation': activation,
                    'strategy': strategy,
                    'trial_number': best.number,
                    'f1_score_cv': best.value,
                    'params': best.params,
                    'optimizer': best.params.get('optimizer'),
                    'n_trials': len(trials_combo)
                }

    # Agrupar por combinacion (optimizer x strategy)
    by_combination = {}
    for optimizer in config.optimizers:
        for strategy in config.strategies:
            combo_key = f"{optimizer}_{strategy}"
            trials_combo = [
                t for t in completed_trials
                if t.params.get('optimizer') == optimizer and t.params.get('strategy') == strategy
            ]
            if trials_combo:
                best = max(trials_combo, key=lambda t: t.value)
                by_combination[combo_key] = {
                    'optimizer': optimizer,
                    'strategy': strategy,
                    'trial_number': best.number,
                    'f1_score_cv': best.value,
                    'params': best.params,
                    'n_trials': len(trials_combo)
                }

    # Crear ranking de combinaciones
    ranking = sorted(
        by_combination.values(),
        key=lambda x: x['f1_score_cv'],
        reverse=True
    )

    return {
        'by_optimizer': by_optimizer,
        'by_strategy': by_strategy,
        'by_activation': by_activation,
        'by_activation_strategy': by_activation_strategy,
        'by_combination': by_combination,
        'ranking': ranking
    }


def create_synthetic_config(
    optimizer: str,
    strategy: str,
    by_optimizer: Dict[str, Any],
    by_strategy: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Crea una configuracion sintetica para una combinacion no explorada.

    Combina los mejores parametros del optimizador con los mejores de la estrategia.

    Args:
        optimizer: Nombre del optimizador (QPSO o QDPSO)
        strategy: Nombre de la estrategia (forward, weighted, layerwise)
        by_optimizer: Mejores configuraciones por optimizador
        by_strategy: Mejores configuraciones por estrategia

    Returns:
        Diccionario con la configuracion sintetica
    """
    # Obtener parametros base del optimizador
    if optimizer in by_optimizer:
        opt_params = by_optimizer[optimizer]['params'].copy()
    else:
        # Usar valores por defecto si no hay datos del optimizador
        opt_params = {
            'optimizer': optimizer,
            'n_particles': 50,
            'max_iters': 150,
            'n_hidden_layers': 2,
            'neurons_multiplier': 2.5,
            'neuron_decay': 0.7,
            'weight_bound': 1.0,
            'patience': 40
        }
        if optimizer == 'QPSO':
            opt_params['alpha_start'] = 0.9
            opt_params['alpha_end'] = 0.5
        else:
            opt_params['g'] = 0.96

    # Crear copia y establecer optimizador y estrategia
    synthetic_params = opt_params.copy()
    synthetic_params['optimizer'] = optimizer
    synthetic_params['strategy'] = strategy

    # Agregar parametros especificos de la estrategia si existen
    if strategy in by_strategy:
        strat_params = by_strategy[strategy]['params']

        # Para weighted: agregar layer_decay y regularization
        if strategy == 'weighted':
            if 'layer_decay' in strat_params:
                synthetic_params['layer_decay'] = strat_params['layer_decay']
            else:
                synthetic_params['layer_decay'] = 0.7
            if 'regularization' in strat_params:
                synthetic_params['regularization'] = strat_params['regularization']
            else:
                synthetic_params['regularization'] = 0.01

        # Para layerwise: agregar iters_per_layer y fine_tune_iters
        elif strategy == 'layerwise':
            if 'iters_per_layer' in strat_params:
                synthetic_params['iters_per_layer'] = strat_params['iters_per_layer']
            else:
                synthetic_params['iters_per_layer'] = 50
            if 'fine_tune_iters' in strat_params:
                synthetic_params['fine_tune_iters'] = strat_params['fine_tune_iters']
            else:
                synthetic_params['fine_tune_iters'] = 50
    else:
        # Valores por defecto si no hay datos de la estrategia
        if strategy == 'weighted':
            synthetic_params['layer_decay'] = 0.7
            synthetic_params['regularization'] = 0.01
        elif strategy == 'layerwise':
            synthetic_params['iters_per_layer'] = 50
            synthetic_params['fine_tune_iters'] = 50

    return synthetic_params


def fill_missing_combinations(analysis: Dict[str, Any], config: SearchConfig) -> Dict[str, Any]:
    """
    Rellena las combinaciones faltantes con configuraciones sinteticas.

    Args:
        analysis: Resultado de analyze_results_by_category()
        config: Configuracion del estudio

    Returns:
        Analisis actualizado con todas las combinaciones
    """
    by_optimizer = analysis['by_optimizer']
    by_strategy = analysis['by_strategy']
    by_combination = analysis['by_combination'].copy()

    # Identificar combinaciones faltantes
    all_possible_combos = set()
    for opt in config.optimizers:
        for strat in config.strategies:
            all_possible_combos.add(f"{opt}_{strat}")

    found_combos = set(by_combination.keys())
    missing_combos = all_possible_combos - found_combos

    # Crear configuraciones sinteticas para las faltantes
    for combo_key in missing_combos:
        opt, strat = combo_key.split('_')

        synthetic_params = create_synthetic_config(
            optimizer=opt,
            strategy=strat,
            by_optimizer=by_optimizer,
            by_strategy=by_strategy
        )

        by_combination[combo_key] = {
            'optimizer': opt,
            'strategy': strat,
            'trial_number': -1,  # Indicador de config sintetica
            'f1_score_cv': 0.0,  # Sin CV porque no fue evaluada
            'params': synthetic_params,
            'n_trials': 0,
            'is_synthetic': True  # Marcador de config sintetica
        }

    # Actualizar ranking (las sinteticas al final)
    explored_combos = [c for c in by_combination.values() if not c.get('is_synthetic', False)]
    synthetic_combos = [c for c in by_combination.values() if c.get('is_synthetic', False)]

    ranking = sorted(explored_combos, key=lambda x: x['f1_score_cv'], reverse=True)
    ranking.extend(sorted(synthetic_combos, key=lambda x: f"{x['optimizer']}_{x['strategy']}"))

    return {
        'by_optimizer': by_optimizer,
        'by_strategy': by_strategy,
        'by_combination': by_combination,
        'ranking': ranking,
        'missing_combos': list(missing_combos)
    }


def print_category_analysis(analysis: Dict[str, Any], config: SearchConfig):
    """
    Imprime el analisis por categorias en formato de tabla.

    Args:
        analysis: Resultado de analyze_results_by_category()
        config: Configuracion del estudio
    """
    print_header("ANALISIS POR CATEGORIAS")

    # 1. Mejor por Optimizador
    print("\n" + "-" * 70)
    print(" MEJOR CONFIGURACION POR OPTIMIZADOR")
    print("-" * 70)
    print(f"{'Optimizador':<12} {'F1-CV':>8} {'Trial':>7} {'Estrategia':<12} {'Trials':>7}")
    print("-" * 70)

    for opt_name, opt_data in analysis['by_optimizer'].items():
        print(f"{opt_name:<12} {opt_data['f1_score_cv']:>8.4f} "
              f"#{opt_data['trial_number']:>5} {opt_data['strategy']:<12} "
              f"{opt_data['n_trials']:>7}")

    # 2. Mejor por Estrategia
    print("\n" + "-" * 70)
    print(" MEJOR CONFIGURACION POR ESTRATEGIA")
    print("-" * 70)
    print(f"{'Estrategia':<12} {'F1-CV':>8} {'Trial':>7} {'Optimizador':<12} {'Trials':>7}")
    print("-" * 70)

    for strat_name, strat_data in analysis['by_strategy'].items():
        print(f"{strat_name:<12} {strat_data['f1_score_cv']:>8.4f} "
              f"#{strat_data['trial_number']:>5} {strat_data['optimizer']:<12} "
              f"{strat_data['n_trials']:>7}")

    # 3. Mejor por Funcion de Activacion (Fase 2 - para referencia futura)
    if analysis.get('by_activation'):
        print("\n" + "-" * 75)
        print(" MEJOR CONFIGURACION POR FUNCION DE ACTIVACION")
        print("-" * 75)
        print(f"{'Activacion':<12} {'F1-CV':>8} {'Trial':>7} {'Estrategia':<12} {'Trials':>7}")
        print("-" * 75)

        # Ordenar por F1-score descendente
        sorted_activations = sorted(
            analysis['by_activation'].items(),
            key=lambda x: x[1]['f1_score_cv'],
            reverse=True
        )

        for act_name, act_data in sorted_activations:
            print(f"{act_name:<12} {act_data['f1_score_cv']:>8.4f} "
                  f"#{act_data['trial_number']:>5} {act_data['strategy']:<12} "
                  f"{act_data['n_trials']:>7}")

    # 4. Mejor por Combinacion Activacion x Estrategia (para referencia futura)
    if analysis.get('by_activation_strategy'):
        print("\n" + "-" * 80)
        print(" MEJOR CONFIGURACION POR ACTIVACION x ESTRATEGIA (Referencia Futura)")
        print("-" * 80)
        print(f"{'Rank':<5} {'Activacion':<12} {'Estrategia':<12} {'F1-CV':>8} {'Trial':>7} {'Trials':>7}")
        print("-" * 80)

        # Ordenar por F1-score descendente
        sorted_act_strat = sorted(
            analysis['by_activation_strategy'].values(),
            key=lambda x: x['f1_score_cv'],
            reverse=True
        )

        for rank, combo in enumerate(sorted_act_strat, 1):
            print(f"{rank:<5} {combo['activation']:<12} {combo['strategy']:<12} "
                  f"{combo['f1_score_cv']:>8.4f} #{combo['trial_number']:>5} "
                  f"{combo['n_trials']:>7}")

        # Destacar la mejor combinacion
        if sorted_act_strat:
            best = sorted_act_strat[0]
            print("-" * 80)
            print(f" MEJOR: {best['activation'].upper()} + {best['strategy'].upper()} "
                  f"(F1-CV: {best['f1_score_cv']:.4f})")

    # 5. Mejor por Combinacion Optimizer x Strategy
    print("\n" + "-" * 75)
    print(" MEJOR CONFIGURACION POR COMBINACION (Optimizer x Strategy)")
    print("-" * 75)
    print(f"{'Rank':<5} {'Combinacion':<20} {'F1-CV':>8} {'Trial':>7} {'Trials':>7} {'Estado':>12}")
    print("-" * 75)

    # Identificar combinaciones posibles y cuales faltan/son sinteticas
    all_possible_combos = set()
    for opt in config.optimizers:
        for strat in config.strategies:
            all_possible_combos.add(f"{opt}_{strat}")

    # Obtener combinaciones sinteticas si existen
    missing_combos = analysis.get('missing_combos', [])
    synthetic_combos_set = set(missing_combos)

    for rank, combo in enumerate(analysis['ranking'], 1):
        combo_name = f"{combo['optimizer']} + {combo['strategy']}"
        is_synthetic = combo.get('is_synthetic', False)

        if is_synthetic:
            # Configuracion sintetica - aun no evaluada en CV
            print(f"{rank:<5} {combo_name:<20} {'N/A':>8} "
                  f"{'N/A':>7} {combo['n_trials']:>7} {'SINTETICA':>12}")
        else:
            print(f"{rank:<5} {combo_name:<20} {combo['f1_score_cv']:>8.4f} "
                  f"#{combo['trial_number']:>5} {combo['n_trials']:>7} {'EXPLORADA':>12}")

    # Mostrar nota sobre combinaciones sinteticas
    if missing_combos:
        print("-" * 75)
        print(f" NOTA: {len(missing_combos)} combinacion(es) no fueron exploradas por Optuna.")
        print(f"       Se crearon configuraciones SINTETICAS combinando:")
        print(f"       - Mejores parametros del optimizador")
        print(f"       - Mejores parametros de la estrategia")
        print(f"       Estas seran evaluadas en test para comparacion completa.")

    # Destacar la mejor combinacion explorada
    explored_ranking = [c for c in analysis['ranking'] if not c.get('is_synthetic', False)]
    if explored_ranking:
        best = explored_ranking[0]
        print("\n" + "=" * 75)
        print(f" MEJOR COMBINACION EXPLORADA (CV): {best['optimizer']} + {best['strategy'].upper()}")
        print(f" F1-Score (CV): {best['f1_score_cv']:.4f} (Trial #{best['trial_number']})")
        print("=" * 75)


def evaluate_single_config(
    params: Dict[str, Any],
    data,
    config: SearchConfig,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evalua una configuracion especifica en el conjunto de test.

    Args:
        params: Hiperparametros de la configuracion
        data: Datos MCW
        config: Configuracion del estudio
        verbose: Si True, muestra progreso del entrenamiento

    Returns:
        Diccionario con resultados de evaluacion
    """
    # Usar dispositivo configurado
    if config.device == 'auto':
        device = get_optimal_device()
    else:
        device = config.device

    input_dim = data.n_features
    output_dim = data.n_classes

    # Reconstruir arquitectura
    hidden_layers = generate_architecture(
        input_dim,
        params['n_hidden_layers'],
        params['neurons_multiplier'],
        params['neuron_decay']
    )

    # Preparar datos
    X_train = np.vstack([data.X_train, data.X_val])
    y_train = np.concatenate([data.y_train, data.y_val])

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(data.X_test, dtype=torch.float32)
    y_test_t = torch.tensor(data.y_test, dtype=torch.long)

    # Crear modelo
    model = QPSOCompatibleANN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=hidden_layers,
        activation='tanh',
        output_activation='softmax'
    )

    # Configurar estrategia
    optimizer_name = params['optimizer']
    strategy_name = params['strategy']

    if optimizer_name == 'QPSO':
        alpha = (params['alpha_start'], params['alpha_end'])
        g = 0.96
    else:
        alpha = (1.0, 0.5)
        g = params['g']

    strategy_params = {}
    if strategy_name == 'weighted':
        strategy_params['layer_decay'] = params['layer_decay']
        strategy_params['regularization'] = params['regularization']
    elif strategy_name == 'layerwise':
        strategy_params['iters_per_layer'] = params['iters_per_layer']
        strategy_params['fine_tune_iters'] = params['fine_tune_iters']

    strategy_config = StrategyConfig(
        n_particles=params['n_particles'],
        max_iters=params['max_iters'],
        alpha=alpha,
        g=g,
        weight_bound=params['weight_bound'],
        patience=params['patience'],
        seed=config.seed,
        **strategy_params
    )

    strategy = create_training_strategy(
        model=model,
        strategy=strategy_name,
        config=strategy_config,
        use_qdpso=(optimizer_name == 'QDPSO'),
        device=device
    )

    # Entrenar con datos de train (85% train, 15% val interna)
    split_idx = int(len(y_train) * 0.85)
    X_tr = X_train_t[:split_idx]
    y_tr = y_train_t[:split_idx]
    X_vl = X_train_t[split_idx:]
    y_vl = y_train_t[split_idx:]

    strategy.set_data(X_tr, y_tr, X_vl, y_vl)

    start_time = time.time()
    try:
        result = strategy.train(verbose=verbose)
        training_time = time.time() - start_time
    except Exception as e:
        return {
            'error': str(e),
            'test_f1': 0.0,
            'test_accuracy': 0.0
        }

    # Evaluar en test
    with torch.no_grad():
        model.to(device)
        X_test_t = X_test_t.to(device)
        output = model(X_test_t)
        test_preds = output.argmax(dim=1).cpu().numpy()
        test_loss = torch.nn.CrossEntropyLoss()(output, y_test_t.to(device)).item()

    # Metricas detalladas
    metrics_calc = MulticlassMetrics()
    detailed_metrics = metrics_calc.calculate_all_metrics(data.y_test, test_preds)

    return {
        'params': params,
        'hidden_layers': hidden_layers,
        'n_params': model.num_params,
        'test_accuracy': detailed_metrics['accuracy'],
        'test_f1': detailed_metrics['f1_score']['macro'],
        'test_precision': detailed_metrics['precision']['macro'],
        'test_recall': detailed_metrics['recall']['macro'],
        'test_kappa': detailed_metrics['cohen_kappa'],
        'test_loss': test_loss,
        'training_time': training_time,
        'iterations': result.iterations,
        'convergence_reason': result.convergence_reason,
        'history': result.history,
        'best_loss': result.best_loss,
        'best_accuracy': result.best_accuracy
    }


def evaluate_all_combinations(
    analysis: Dict[str, Any],
    data,
    config: SearchConfig
) -> Dict[str, Dict[str, Any]]:
    """
    Evalua la mejor configuracion de cada combinacion (optimizer x strategy) en test.

    Args:
        analysis: Resultado de analyze_results_by_category()
        data: Datos MCW
        config: Configuracion del estudio

    Returns:
        Diccionario con resultados de evaluacion por cada combinacion
    """
    print_header("EVALUACION DE TODAS LAS COMBINACIONES EN TEST")

    results = {}
    total_combos = len(analysis['by_combination'])

    # Separar exploradas y sinteticas para mejor visualizacion
    explored_combos = [(k, v) for k, v in analysis['by_combination'].items()
                       if not v.get('is_synthetic', False)]
    synthetic_combos = [(k, v) for k, v in analysis['by_combination'].items()
                        if v.get('is_synthetic', False)]

    # Ordenar: primero exploradas (por F1-CV), luego sinteticas
    sorted_combos = sorted(explored_combos, key=lambda x: x[1]['f1_score_cv'], reverse=True)
    sorted_combos.extend(sorted(synthetic_combos, key=lambda x: f"{x[1]['optimizer']}_{x[1]['strategy']}"))

    print(f"\n  Total de combinaciones: {total_combos}")
    print(f"  - Exploradas por Optuna: {len(explored_combos)}")
    print(f"  - Sinteticas (generadas): {len(synthetic_combos)}")

    for idx, (combo_key, combo_data) in enumerate(sorted_combos, 1):
        optimizer = combo_data['optimizer']
        strategy = combo_data['strategy']
        is_synthetic = combo_data.get('is_synthetic', False)

        synthetic_tag = " [SINTETICA]" if is_synthetic else ""
        print(f"\n[{idx}/{total_combos}] Evaluando: {optimizer} + {strategy.upper()}{synthetic_tag}")

        if is_synthetic:
            print(f"    Config sintetica (sin CV previo)")
        else:
            print(f"    Trial #{combo_data['trial_number']} | F1-CV: {combo_data['f1_score_cv']:.4f}")

        # Evaluar en test
        eval_result = evaluate_single_config(
            params=combo_data['params'],
            data=data,
            config=config,
            verbose=False
        )

        eval_result['f1_score_cv'] = combo_data['f1_score_cv']
        eval_result['trial_number'] = combo_data['trial_number']
        eval_result['optimizer'] = optimizer
        eval_result['strategy'] = strategy
        eval_result['is_synthetic'] = is_synthetic

        results[combo_key] = eval_result

        # Mostrar resultado
        if 'error' not in eval_result:
            print(f"    Test F1: {eval_result['test_f1']:.4f} | "
                  f"Acc: {eval_result['test_accuracy']:.4f} | "
                  f"Time: {eval_result['training_time']:.2f}s")
        else:
            print(f"    ERROR: {eval_result['error']}")

    return results


def print_single_combination_details(rank: int, result: Dict[str, Any], is_winner: bool = False):
    """
    Imprime los detalles completos de una combinacion.

    Args:
        rank: Posicion en el ranking
        result: Resultados de la combinacion
        is_winner: Si es el ganador absoluto
    """
    if 'error' in result:
        print(f"\n[{rank}] {result.get('optimizer', '?')} + {result.get('strategy', '?').upper()} - ERROR")
        print(f"    Error: {result['error']}")
        return

    combo_name = f"{result['optimizer']} + {result['strategy'].upper()}"
    is_synthetic = result.get('is_synthetic', False)
    synthetic_tag = " [CONFIG SINTETICA]" if is_synthetic else ""

    if is_winner:
        print("\n" + "=" * 70)
        print(f" #{rank} {combo_name} *** GANADOR ABSOLUTO ***{synthetic_tag}")
        print("=" * 70)
    else:
        print("\n" + "-" * 70)
        print(f" #{rank} {combo_name}{synthetic_tag}")
        print("-" * 70)

    if is_synthetic:
        print(f"  Origen: Config sintetica (combinacion no explorada por Optuna)")
    else:
        print(f"  Trial: #{result['trial_number']}")

    # Metricas
    print(f"\n  --- Metricas ---")
    print(f"  {'':15} {'CV':>10} {'Test':>10}")

    # Mostrar CV como N/A si es sintetica
    if is_synthetic:
        print(f"  {'F1-Score':<15} {'N/A':>10} {result['test_f1']:>10.4f}")
    else:
        print(f"  {'F1-Score':<15} {result['f1_score_cv']:>10.4f} {result['test_f1']:>10.4f}")
    print(f"  {'Accuracy':<15} {'-':>10} {result['test_accuracy']:>10.4f}")
    print(f"  {'Precision':<15} {'-':>10} {result['test_precision']:>10.4f}")
    print(f"  {'Recall':<15} {'-':>10} {result['test_recall']:>10.4f}")
    print(f"  {'Kappa':<15} {'-':>10} {result['test_kappa']:>10.4f}")
    print(f"  {'Loss':<15} {'-':>10} {result['test_loss']:>10.6f}")

    # Arquitectura
    print(f"\n  --- Arquitectura ---")
    print(f"  Capas ocultas: {result['hidden_layers']}")
    print(f"  Parametros: {result['n_params']:,}")
    print(f"  Tiempo entrenamiento: {result['training_time']:.2f}s")

    # Hiperparametros
    params = result['params']
    print(f"\n  --- Hiperparametros ---")
    print(f"  Particulas: {params['n_particles']}")
    print(f"  Max Iteraciones: {params['max_iters']}")
    print(f"  Weight Bound: {params['weight_bound']:.3f}")
    print(f"  Patience: {params['patience']}")
    # Fase 2: Mostrar funcion de activacion
    activation = params.get('activation', 'tanh')
    print(f"  Activacion: {activation}")

    if result['optimizer'] == 'QPSO':
        print(f"  Alpha: ({params['alpha_start']:.4f}, {params['alpha_end']:.4f})")
    else:
        print(f"  g: {params['g']:.4f}")

    if result['strategy'] == 'weighted':
        print(f"  Layer Decay: {params['layer_decay']:.4f}")
        print(f"  Regularization: {params['regularization']:.6f}")
    elif result['strategy'] == 'layerwise':
        print(f"  Iters per Layer: {params['iters_per_layer']}")
        print(f"  Fine-tune Iters: {params['fine_tune_iters']}")

    # Arquitectura detallada
    print(f"\n  --- Arquitectura Detallada ---")
    print(f"  n_hidden_layers: {params['n_hidden_layers']}")
    print(f"  neurons_multiplier: {params['neurons_multiplier']:.4f}")
    print(f"  neuron_decay: {params['neuron_decay']:.4f}")


def print_combination_results(combination_results: Dict[str, Dict[str, Any]]):
    """
    Imprime tabla comparativa de resultados de todas las combinaciones evaluadas en test.

    Args:
        combination_results: Resultado de evaluate_all_combinations()
    """
    print_header("COMPARATIVA DE COMBINACIONES EN TEST")

    # Ordenar por F1-Score en test
    sorted_results = sorted(
        combination_results.items(),
        key=lambda x: x[1].get('test_f1', 0),
        reverse=True
    )

    # Contar sinteticas
    n_synthetic = sum(1 for _, r in sorted_results if r.get('is_synthetic', False))

    # Tabla resumen
    print("\n" + "=" * 105)
    print(f"{'Rank':<5} {'Combinacion':<22} {'F1-CV':>8} {'F1-Test':>8} "
          f"{'Acc':>7} {'Kappa':>7} {'Time':>7} {'Arch':>15} {'Tipo':>10}")
    print("=" * 105)

    for rank, (combo_key, result) in enumerate(sorted_results, 1):
        if 'error' in result:
            print(f"{rank:<5} {combo_key:<22} {'ERROR':<60}")
            continue

        combo_name = f"{result['optimizer']} + {result['strategy']}"
        arch_str = str(result['hidden_layers'])
        is_synthetic = result.get('is_synthetic', False)

        # Marcar el ganador y tipo
        marker = " *" if rank == 1 else ""
        tipo = "SINT" if is_synthetic else "EXPL"

        # F1-CV como N/A para sinteticas
        f1_cv_str = "N/A" if is_synthetic else f"{result['f1_score_cv']:.4f}"

        print(f"{rank:<5} {combo_name:<22} {f1_cv_str:>8} "
              f"{result['test_f1']:>8.4f} {result['test_accuracy']:>7.4f} "
              f"{result['test_kappa']:>7.4f} {result['training_time']:>6.2f}s "
              f"{arch_str:>15} {tipo:>10}{marker}")

    print("=" * 105)
    print("  * = Ganador Absoluto | EXPL = Explorada por Optuna | SINT = Config Sintetica")
    if n_synthetic > 0:
        print(f"  {n_synthetic} combinacion(es) fueron generadas sinteticamente para comparacion completa")

    # Detalles de cada combinacion
    print_header("CONFIGURACION DETALLADA DE CADA COMBINACION")

    for rank, (combo_key, result) in enumerate(sorted_results, 1):
        is_winner = (rank == 1)
        print_single_combination_details(rank, result, is_winner)

    return sorted_results[0] if sorted_results else None


def print_training_comparison(combination_results: Dict[str, Dict[str, Any]]):
    """
    Imprime comparativa del entrenamiento de todas las combinaciones.

    Args:
        combination_results: Resultado de evaluate_all_combinations()
    """
    print_header("COMPARATIVA DE ENTRENAMIENTO")

    # Ordenar por F1-Score en test
    sorted_results = sorted(
        combination_results.items(),
        key=lambda x: x[1].get('test_f1', 0),
        reverse=True
    )

    # Mostrar resumen de convergencia
    print("\n" + "=" * 85)
    print(f"{'Rank':<5} {'Combinacion':<22} {'Iters':>7} {'Best Loss':>10} "
          f"{'Best Acc':>9} {'Final Loss':>11} {'Razon':>15}")
    print("=" * 85)

    for rank, (combo_key, result) in enumerate(sorted_results, 1):
        if 'error' in result:
            print(f"{rank:<5} {combo_key:<22} {'ERROR':<50}")
            continue

        combo_name = f"{result['optimizer']} + {result['strategy']}"

        # Obtener loss final del historial
        history = result.get('history', {})
        val_losses = history.get('val_loss', [])
        final_loss = val_losses[-1] if val_losses else result.get('test_loss', 0)

        print(f"{rank:<5} {combo_name:<22} {result['iterations']:>7} "
              f"{result['best_loss']:>10.6f} {result['best_accuracy']:>9.4f} "
              f"{final_loss:>11.6f} {result['convergence_reason']:>15}")

    print("=" * 85)

    # Mostrar curvas de entrenamiento simplificadas
    print_header("CURVAS DE ENTRENAMIENTO (Resumen)")

    for rank, (combo_key, result) in enumerate(sorted_results, 1):
        if 'error' in result:
            continue

        combo_name = f"{result['optimizer']} + {result['strategy'].upper()}"
        history = result.get('history', {})

        if not history:
            print(f"\n[{rank}] {combo_name}: Sin historial disponible")
            continue

        val_losses = history.get('val_loss', [])
        val_accs = history.get('val_accuracy', [])
        train_losses = history.get('loss', [])
        train_accs = history.get('accuracy', [])

        n_iters = len(val_losses) if val_losses else len(train_losses)

        if n_iters == 0:
            print(f"\n[{rank}] {combo_name}: Sin datos de entrenamiento")
            continue

        # Mostrar resumen cada cierto intervalo
        interval = max(1, n_iters // 10)  # Mostrar ~10 puntos

        print(f"\n" + "-" * 70)
        print(f" [{rank}] {combo_name}")
        print("-" * 70)
        print(f"  {'Iter':>6} {'Train Loss':>12} {'Train Acc':>11} "
              f"{'Val Loss':>12} {'Val Acc':>11}")
        print("-" * 70)

        for i in range(0, n_iters, interval):
            t_loss = train_losses[i] if i < len(train_losses) else '-'
            t_acc = train_accs[i] if i < len(train_accs) else '-'
            v_loss = val_losses[i] if i < len(val_losses) else '-'
            v_acc = val_accs[i] if i < len(val_accs) else '-'

            t_loss_str = f"{t_loss:.6f}" if isinstance(t_loss, float) else str(t_loss)
            t_acc_str = f"{t_acc:.4f}" if isinstance(t_acc, float) else str(t_acc)
            v_loss_str = f"{v_loss:.6f}" if isinstance(v_loss, float) else str(v_loss)
            v_acc_str = f"{v_acc:.4f}" if isinstance(v_acc, float) else str(v_acc)

            print(f"  {i:>6} {t_loss_str:>12} {t_acc_str:>11} "
                  f"{v_loss_str:>12} {v_acc_str:>11}")

        # Mostrar ultimo punto si no se mostro
        if (n_iters - 1) % interval != 0:
            i = n_iters - 1
            t_loss = train_losses[i] if i < len(train_losses) else '-'
            t_acc = train_accs[i] if i < len(train_accs) else '-'
            v_loss = val_losses[i] if i < len(val_losses) else '-'
            v_acc = val_accs[i] if i < len(val_accs) else '-'

            t_loss_str = f"{t_loss:.6f}" if isinstance(t_loss, float) else str(t_loss)
            t_acc_str = f"{t_acc:.4f}" if isinstance(t_acc, float) else str(t_acc)
            v_loss_str = f"{v_loss:.6f}" if isinstance(v_loss, float) else str(v_loss)
            v_acc_str = f"{v_acc:.4f}" if isinstance(v_acc, float) else str(v_acc)

            print(f"  {i:>6} {t_loss_str:>12} {t_acc_str:>11} "
                  f"{v_loss_str:>12} {v_acc_str:>11}")

        print(f"  Convergencia: {result['convergence_reason']}")

    # Comparativa final de convergencia
    print("\n" + "=" * 70)
    print(" ANALISIS DE CONVERGENCIA")
    print("=" * 70)

    # Encontrar quien convergio mas rapido
    valid_results = [(k, r) for k, r in sorted_results if 'error' not in r]

    if valid_results:
        fastest = min(valid_results, key=lambda x: x[1]['iterations'])
        slowest = max(valid_results, key=lambda x: x[1]['iterations'])
        best_loss = min(valid_results, key=lambda x: x[1]['best_loss'])

        print(f"\n  Convergencia mas rapida: {fastest[1]['optimizer']} + {fastest[1]['strategy'].upper()}")
        print(f"    - Iteraciones: {fastest[1]['iterations']}")
        print(f"    - Tiempo: {fastest[1]['training_time']:.2f}s")

        print(f"\n  Convergencia mas lenta: {slowest[1]['optimizer']} + {slowest[1]['strategy'].upper()}")
        print(f"    - Iteraciones: {slowest[1]['iterations']}")
        print(f"    - Tiempo: {slowest[1]['training_time']:.2f}s")

        print(f"\n  Mejor loss de entrenamiento: {best_loss[1]['optimizer']} + {best_loss[1]['strategy'].upper()}")
        print(f"    - Best Loss: {best_loss[1]['best_loss']:.6f}")
        print(f"    - Best Accuracy: {best_loss[1]['best_accuracy']:.4f}")


# =============================================================================
# VISUALIZACION DE RESULTADOS
# =============================================================================

def visualize_study(study: 'optuna.Study', output_dir: str):
    """
    Genera visualizaciones del estudio de Optuna.

    Args:
        study: Estudio completado
        output_dir: Directorio de salida
    """
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
            plot_contour
        )
        import plotly.io as pio

        print("\nGenerando visualizaciones...")

        # 1. Historia de optimizacion
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(output_dir, 'optimization_history.png'))
        print("  - optimization_history.png")

        # 2. Importancia de parametros
        try:
            fig = plot_param_importances(study)
            fig.write_image(os.path.join(output_dir, 'param_importances.png'))
            print("  - param_importances.png")
        except Exception:
            print("  - param_importances.png (no disponible)")

        # 3. Coordenadas paralelas
        fig = plot_parallel_coordinate(study)
        fig.write_image(os.path.join(output_dir, 'parallel_coordinate.png'))
        print("  - parallel_coordinate.png")

        # 4. Slice plot para parametros principales
        try:
            fig = plot_slice(study, params=['n_particles', 'max_iters', 'neurons_multiplier'])
            fig.write_image(os.path.join(output_dir, 'slice_plot.png'))
            print("  - slice_plot.png")
        except Exception:
            pass

    except ImportError:
        print("  Visualizaciones no disponibles. Instale: pip install plotly kaleido")
    except Exception as e:
        print(f"  Error generando visualizaciones: {e}")


def save_results(
    study: 'optuna.Study',
    final_results: Dict[str, Any],
    config: SearchConfig,
    output_dir: str,
    category_analysis: Optional[Dict[str, Any]] = None,
    combination_results: Optional[Dict[str, Dict[str, Any]]] = None
):
    """
    Guarda los resultados del estudio.

    Args:
        study: Estudio completado
        final_results: Resultados de evaluacion final
        config: Configuracion usada
        output_dir: Directorio de salida
        category_analysis: Analisis por categorias (opcional)
        combination_results: Resultados de evaluacion de todas las combinaciones (opcional)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Mejores parametros (JSON)
    best_params_path = os.path.join(output_dir, f'best_params_{timestamp}.json')
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
            'final_test_results': {
                'accuracy': final_results['test_accuracy'],
                'f1': final_results['test_f1'],
                'precision': final_results['test_precision'],
                'recall': final_results['test_recall'],
                'kappa': final_results['test_kappa'],
                'loss': final_results['test_loss']
            },
            'architecture': {
                'hidden_layers': final_results['hidden_layers'],
                'n_params': final_results['n_params']
            },
            'config': asdict(config)
        }, f, indent=2)
    print(f"\nMejores parametros guardados en: {best_params_path}")

    # 2. Top K configuraciones (JSON)
    top_k_path = os.path.join(output_dir, f'top_{config.save_top_k}_configs_{timestamp}.json')
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:config.save_top_k]

    top_configs = []
    for i, trial in enumerate(top_trials):
        if trial.value is not None:
            top_configs.append({
                'rank': i + 1,
                'trial': trial.number,
                'value': trial.value,
                'params': trial.params
            })

    with open(top_k_path, 'w') as f:
        json.dump(top_configs, f, indent=2)
    print(f"Top {config.save_top_k} configuraciones guardadas en: {top_k_path}")

    # 3. Historial completo (CSV)
    try:
        df = study.trials_dataframe()
        csv_path = os.path.join(output_dir, f'trials_history_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Historial de trials guardado en: {csv_path}")
    except Exception:
        pass

    # 4. Analisis por categorias (JSON)
    if category_analysis:
        category_path = os.path.join(output_dir, f'category_analysis_{timestamp}.json')

        # Preparar datos serializables
        category_data = {
            'by_optimizer': {},
            'by_strategy': {},
            'by_combination': {},
            'ranking': []
        }

        # Por optimizador
        for opt_name, opt_data in category_analysis['by_optimizer'].items():
            category_data['by_optimizer'][opt_name] = {
                'trial_number': opt_data['trial_number'],
                'f1_score_cv': opt_data['f1_score_cv'],
                'strategy': opt_data['strategy'],
                'n_trials': opt_data['n_trials'],
                'params': opt_data['params']
            }

        # Por estrategia
        for strat_name, strat_data in category_analysis['by_strategy'].items():
            category_data['by_strategy'][strat_name] = {
                'trial_number': strat_data['trial_number'],
                'f1_score_cv': strat_data['f1_score_cv'],
                'optimizer': strat_data['optimizer'],
                'n_trials': strat_data['n_trials'],
                'params': strat_data['params']
            }

        # Por combinacion
        for combo_key, combo_data in category_analysis['by_combination'].items():
            category_data['by_combination'][combo_key] = {
                'optimizer': combo_data['optimizer'],
                'strategy': combo_data['strategy'],
                'trial_number': combo_data['trial_number'],
                'f1_score_cv': combo_data['f1_score_cv'],
                'n_trials': combo_data['n_trials'],
                'params': combo_data['params']
            }

        # Ranking
        for combo in category_analysis['ranking']:
            category_data['ranking'].append({
                'optimizer': combo['optimizer'],
                'strategy': combo['strategy'],
                'f1_score_cv': combo['f1_score_cv'],
                'trial_number': combo['trial_number']
            })

        with open(category_path, 'w') as f:
            json.dump(category_data, f, indent=2)
        print(f"Analisis por categorias guardado en: {category_path}")

    # 5. Resultados de evaluacion de todas las combinaciones (JSON)
    if combination_results:
        combo_results_path = os.path.join(output_dir, f'combination_test_results_{timestamp}.json')

        # Preparar datos serializables y ordenar por F1-test
        sorted_combos = sorted(
            combination_results.items(),
            key=lambda x: x[1].get('test_f1', 0),
            reverse=True
        )

        combo_data = {
            'timestamp': timestamp,
            'n_combinations_evaluated': len(combination_results),
            'ranking': []
        }

        for rank, (combo_key, result) in enumerate(sorted_combos, 1):
            if 'error' in result:
                combo_data['ranking'].append({
                    'rank': rank,
                    'combination': combo_key,
                    'error': result['error']
                })
            else:
                combo_data['ranking'].append({
                    'rank': rank,
                    'combination': combo_key,
                    'optimizer': result['optimizer'],
                    'strategy': result['strategy'],
                    'trial_number': result['trial_number'],
                    'f1_score_cv': result['f1_score_cv'],
                    'test_results': {
                        'f1': result['test_f1'],
                        'accuracy': result['test_accuracy'],
                        'precision': result['test_precision'],
                        'recall': result['test_recall'],
                        'kappa': result['test_kappa'],
                        'loss': result['test_loss']
                    },
                    'architecture': {
                        'hidden_layers': result['hidden_layers'],
                        'n_params': result['n_params']
                    },
                    'training_time': result['training_time'],
                    'params': result['params']
                })

        # Identificar ganador
        if sorted_combos and 'error' not in sorted_combos[0][1]:
            combo_data['winner'] = {
                'combination': sorted_combos[0][0],
                'test_f1': sorted_combos[0][1]['test_f1']
            }

        with open(combo_results_path, 'w') as f:
            json.dump(combo_data, f, indent=2)
        print(f"Resultados de combinaciones en test guardados en: {combo_results_path}")


# =============================================================================
# ARGUMENTOS DE LINEA DE COMANDOS
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parsea los argumentos de linea de comandos.

    Returns:
        Namespace con los argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description='Busqueda de hiperparametros para QPSO/QDPSO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Busqueda con valores por defecto (QPSO + forward)
  python main_hyperparameter_search.py

  # Solo QPSO con todas las estrategias
  python main_hyperparameter_search.py --optimizer QPSO --strategy forward weighted layerwise

  # Solo QDPSO con forward
  python main_hyperparameter_search.py --optimizer QDPSO --strategy forward

  # Ambos optimizadores con todas las estrategias (6 combinaciones)
  python main_hyperparameter_search.py --optimizer QPSO QDPSO --strategy forward weighted layerwise

  # Mas trials para busqueda mas exhaustiva
  python main_hyperparameter_search.py --n-trials 100 --optimizer QPSO QDPSO
        """
    )

    # Optimizador y estrategia
    parser.add_argument(
        '--optimizer', '-o',
        nargs='+',
        choices=AVAILABLE_OPTIMIZERS,
        default=['QPSO'],
        help=f'Optimizador(es) a explorar. Opciones: {AVAILABLE_OPTIMIZERS}. Default: QPSO'
    )

    parser.add_argument(
        '--strategy', '-s',
        nargs='+',
        choices=AVAILABLE_STRATEGIES,
        default=['forward'],
        help=f'Estrategia(s) a explorar. Opciones: {AVAILABLE_STRATEGIES}. Default: forward'
    )

    # Configuracion del estudio
    parser.add_argument(
        '--n-trials', '-n',
        type=int,
        default=30,
        help='Numero de trials a ejecutar. Default: 30'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Tiempo maximo en segundos. Default: sin limite'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla para reproducibilidad. Default: 42'
    )

    parser.add_argument(
        '--no-ensure-combinations',
        action='store_true',
        help='Desactiva la garantia de explorar todas las combinaciones iniciales'
    )

    parser.add_argument(
        '--n-jobs', '-j',
        type=int,
        default=-1,
        help='Numero de workers paralelos. -1=auto (75%% de cores), 1=secuencial. Default: -1'
    )

    # Persistencia
    parser.add_argument(
        '--storage',
        type=str,
        default=None,
        help='Ruta a base de datos SQLite para persistir estudio. Ej: ./study.db. Default: en memoria'
    )

    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='No continuar estudio existente, crear uno nuevo'
    )

    # Dispositivo
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'mps', 'cpu'],
        default='auto',
        help='Dispositivo para computacion. Default: auto (detecta CUDA/MPS/CPU)'
    )

    # Dataset
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='./data/img/mcw',
        help='Ruta al dataset MCW. Default: ./data/img/mcw'
    )

    # Salida
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/hyperparameter_search',
        help='Directorio de salida. Default: ./results/hyperparameter_search'
    )

    # =========================================================================
    # ESPACIO DE BUSQUEDA
    # =========================================================================

    search_group = parser.add_argument_group('Espacio de Busqueda')

    # Parametros QPSO
    search_group.add_argument(
        '--alpha-start',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[0.7, 1.0],
        help='Rango para alpha inicial (QPSO). Default: 0.7 1.0'
    )

    search_group.add_argument(
        '--alpha-end',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[0.3, 0.7],
        help='Rango para alpha final (QPSO). Default: 0.3 0.7'
    )

    # Parametros QDPSO
    search_group.add_argument(
        '--g-range',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[0.90, 0.99],
        help='Rango para factor g (QDPSO). Default: 0.90 0.99'
    )

    # Enjambre
    search_group.add_argument(
        '--n-particles',
        type=int,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[20, 80],
        help='Rango de particulas. Default: 20 80'
    )

    search_group.add_argument(
        '--max-iters',
        type=int,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[50, 300],
        help='Rango de iteraciones maximas. Default: 50 300'
    )

    # Arquitectura
    search_group.add_argument(
        '--n-hidden-layers',
        type=int,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[1, 3],
        help='Rango de capas ocultas. Default: 1 3'
    )

    search_group.add_argument(
        '--neurons-multiplier',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[1.5, 4.0],
        help='Rango del multiplicador de neuronas. Default: 1.5 4.0'
    )

    search_group.add_argument(
        '--neuron-decay',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[0.5, 0.9],
        help='Rango de decaimiento de neuronas entre capas. Default: 0.5 0.9'
    )

    # Parametros Weighted
    search_group.add_argument(
        '--layer-decay',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[0.5, 0.9],
        help='Rango de decaimiento por capa (weighted). Default: 0.5 0.9'
    )

    search_group.add_argument(
        '--regularization',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[0.001, 0.1],
        help='Rango de regularizacion (weighted). Default: 0.001 0.1'
    )

    # Parametros Layerwise
    search_group.add_argument(
        '--iters-per-layer',
        type=int,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[20, 80],
        help='Rango de iteraciones por capa (layerwise). Default: 20 80'
    )

    search_group.add_argument(
        '--fine-tune-iters',
        type=int,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[20, 80],
        help='Rango de iteraciones de fine-tuning (layerwise). Default: 20 80'
    )

    # Otros
    search_group.add_argument(
        '--weight-bound',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[0.5, 2.0],
        help='Rango de limite de pesos. Default: 0.5 2.0'
    )

    search_group.add_argument(
        '--patience',
        type=int,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[20, 60],
        help='Rango de paciencia (early stopping). Default: 20 60'
    )

    # =========================================================================
    # FASE 2: ESPACIO DE BUSQUEDA AMPLIADO
    # =========================================================================

    phase2_group = parser.add_argument_group('Fase 2: Espacio Ampliado')

    phase2_group.add_argument(
        '--activations',
        type=str,
        nargs='+',
        default=['tanh'],
        choices=['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'gelu'],
        help='Activaciones a explorar. Default: tanh (fija). Ej: --activations relu tanh gelu'
    )

    phase2_group.add_argument(
        '--dropout',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[0.0, 0.0],
        help='Rango de dropout. Default: 0.0 0.0 (sin dropout). Ej: --dropout 0.0 0.5'
    )

    phase2_group.add_argument(
        '--batch-norm',
        type=str,
        nargs='+',
        default=['false'],
        choices=['true', 'false'],
        help='Opciones de batch normalization. Default: false. Ej: --batch-norm true false'
    )

    phase2_group.add_argument(
        '--boundary-strategies',
        type=str,
        nargs='+',
        default=['clamp'],
        choices=['clamp', 'reflect', 'wrap', 'random'],
        help='Estrategias de limites para QPSO. Default: clamp. Ej: --boundary-strategies clamp reflect'
    )

    phase2_group.add_argument(
        '--tol',
        type=float,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[1e-12, 1e-12],
        help='Rango de tolerancia (escala log). Default: 1e-12 1e-12 (fija). Ej: --tol 1e-14 1e-8'
    )

    # =========================================================================
    # FASE 2: PRUNER Y CALLBACKS
    # =========================================================================

    callbacks_group = parser.add_argument_group('Fase 2: Pruner y Callbacks')

    callbacks_group.add_argument(
        '--pruner',
        type=str,
        choices=['median', 'hyperband'],
        default='median',
        help='Tipo de pruner. median=conservador, hyperband=agresivo. Default: median'
    )

    callbacks_group.add_argument(
        '--early-stopping-patience',
        type=int,
        default=0,
        help='Trials sin mejora para detener busqueda (0=desactivado). Default: 0'
    )

    callbacks_group.add_argument(
        '--checkpoint-frequency',
        type=int,
        default=10,
        help='Guardar checkpoint cada N trials (0=desactivado). Default: 10'
    )

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> SearchConfig:
    """
    Crea un SearchConfig a partir de los argumentos parseados.

    Args:
        args: Argumentos parseados

    Returns:
        SearchConfig configurado
    """
    # Determinar ruta de storage (SQLite)
    storage_path = args.storage
    if storage_path is None:
        # Por defecto, crear archivo SQLite en el directorio de salida
        storage_path = os.path.join(args.output_dir, 'optuna_study.db')

    config = SearchConfig(
        # Optimizadores y estrategias
        optimizers=args.optimizer,
        strategies=args.strategy,

        # Estudio Optuna
        n_trials=args.n_trials,
        timeout=args.timeout,
        seed=args.seed,
        ensure_all_combinations=not args.no_ensure_combinations,
        n_jobs=args.n_jobs,

        # Persistencia
        storage_path=storage_path,
        resume_study=not args.no_resume,

        # Dispositivo
        device=args.device,

        # Dataset y salida
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,

        # Espacio de busqueda - QPSO
        alpha_start=tuple(args.alpha_start),
        alpha_end=tuple(args.alpha_end),

        # Espacio de busqueda - QDPSO
        g_range=tuple(args.g_range),

        # Espacio de busqueda - Enjambre
        n_particles=tuple(args.n_particles),
        max_iters=tuple(args.max_iters),

        # Espacio de busqueda - Arquitectura
        n_hidden_layers=tuple(args.n_hidden_layers),
        neurons_multiplier=tuple(args.neurons_multiplier),
        neuron_decay=tuple(args.neuron_decay),

        # Espacio de busqueda - Weighted
        layer_decay=tuple(args.layer_decay),
        regularization=tuple(args.regularization),

        # Espacio de busqueda - Layerwise
        iters_per_layer=tuple(args.iters_per_layer),
        fine_tune_iters=tuple(args.fine_tune_iters),

        # Espacio de busqueda - Otros
        weight_bound=tuple(args.weight_bound),
        patience=tuple(args.patience),

        # =====================================================================
        # FASE 2: Espacio de busqueda ampliado
        # =====================================================================
        activations=args.activations,
        dropout_range=tuple(args.dropout),
        use_batch_norm_options=[s.lower() == 'true' for s in args.batch_norm],
        boundary_strategies=args.boundary_strategies,
        tol_range=tuple(args.tol),

        # FASE 2: Pruner y Callbacks
        pruner_type=args.pruner,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_frequency=args.checkpoint_frequency,
    )
    config.validate()
    return config


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Ejecuta la busqueda de hiperparametros."""
    global logger

    if not OPTUNA_AVAILABLE:
        print("ERROR: Optuna no esta instalado.")
        print("Ejecute: pip install optuna optuna-dashboard plotly kaleido")
        return None, None

    # Parsear argumentos y crear configuracion
    args = parse_args()
    config = create_config_from_args(args)

    # Crear directorio de salida primero (necesario para logging)
    ensure_output_dir(config.output_dir)

    # Inicializar sistema de logging
    logger = setup_logging(config.output_dir, verbose=True)
    logger.info(f"Iniciando búsqueda de hiperparámetros")
    logger.info(f"Configuración: {config}")

    # Construir espacio de busqueda
    search_space = build_search_space(config)

    print_header("HYPERPARAMETER SEARCH - QPSO/QDPSO")
    print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Detectar dispositivo
    if config.device == 'auto':
        device = get_optimal_device()
    else:
        device = config.device
    print(f"Dispositivo: {get_device_info(device)}")
    logger.info(f"Dispositivo seleccionado: {device}")

    # Calcular número de workers para paralelismo
    import multiprocessing
    if config.n_jobs == -1:
        # Usar 75% de los cores disponibles
        actual_n_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))
    elif config.n_jobs <= 0:
        actual_n_jobs = 1
    else:
        actual_n_jobs = config.n_jobs

    print(f"Workers paralelos: {actual_n_jobs} (de {multiprocessing.cpu_count()} cores)")
    logger.info(f"Paralelismo: {actual_n_jobs} workers")

    # Crear directorio de salida
    ensure_output_dir(config.output_dir)

    # =========================================================================
    # CARGAR DATOS
    # =========================================================================

    print_header("CARGANDO DATASET MCW")

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
        tb_module.print_exc()
        return None, None

    # =========================================================================
    # CONFIGURAR ESTUDIO
    # =========================================================================

    print_header("CONFIGURANDO ESTUDIO OPTUNA")

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

    # Configurar almacenamiento persistente (SQLite)
    storage = None
    if config.storage_path:
        storage = f"sqlite:///{config.storage_path}"
        print(f"\n  Almacenamiento: SQLite ({config.storage_path})")
        logger.info(f"Storage SQLite: {config.storage_path}")
    else:
        print(f"\n  Almacenamiento: En memoria (no persistente)")
        logger.warning("Estudio en memoria - se perderá si se interrumpe")

    # Crear sampler
    sampler = TPESampler(seed=config.seed)

    # Proteger los primeros N trials del pruning (N = numero de combinaciones)
    n_startup_trials = n_combos if config.ensure_all_combinations else 5

    # Crear pruner segun configuracion (Fase 2)
    if config.pruner_type == 'hyperband':
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=config.n_folds,
            reduction_factor=3
        )
        print(f"  Pruner: HyperbandPruner (agresivo)")
        logger.info("Usando HyperbandPruner")
    else:
        pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=1)
        print(f"  Pruner: MedianPruner (conservador, {n_startup_trials} startup trials)")
        logger.info(f"Usando MedianPruner con {n_startup_trials} startup trials")

    # Intentar cargar estudio existente o crear uno nuevo
    study = None
    n_previous_trials = 0

    if storage and config.resume_study:
        try:
            study = optuna.load_study(
                study_name=config.study_name,
                storage=storage
            )
            n_previous_trials = len(study.trials)
            print(f"  Continuando estudio existente con {n_previous_trials} trials previos")
            logger.info(f"Estudio cargado: {n_previous_trials} trials previos")

            # Verificar si ya se completaron suficientes trials
            if n_previous_trials >= config.n_trials:
                print(f"\n  NOTA: Ya hay {n_previous_trials} trials (>= {config.n_trials} solicitados)")
                print(f"  Use --no-resume para crear un nuevo estudio o aumente --n-trials")

        except KeyError:
            # Estudio no existe, crear uno nuevo
            print(f"  Creando nuevo estudio persistente")
            logger.info("Creando nuevo estudio (no existía previamente)")
            study = None

    if study is None:
        study = optuna.create_study(
            study_name=config.study_name,
            direction='maximize',  # Maximizar F1-score
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=False
        )
        logger.info(f"Nuevo estudio creado: {config.study_name}")

    # Encolar trials iniciales para garantizar exploracion de todas las combinaciones
    # Solo si no hay trials previos
    if config.ensure_all_combinations and n_previous_trials == 0:
        enqueue_initial_trials(study, config)
    elif config.ensure_all_combinations and n_previous_trials > 0:
        print(f"  Saltando trials iniciales (ya hay {n_previous_trials} previos)")

    # =========================================================================
    # EJECUTAR BUSQUEDA
    # =========================================================================

    print_header("EJECUTANDO BUSQUEDA DE HIPERPARAMETROS")
    print(f"\nOptimizando F1-Score (macro)...")
    print(f"Trials a ejecutar: {config.n_trials - n_previous_trials} nuevos (total: {config.n_trials})")
    print(f"Esto puede tomar varios minutos/horas dependiendo de n_trials.\n")

    # Crear funcion objetivo
    objective = ObjectiveFunction(data, config, search_space)

    # =========================================================================
    # FASE 2: Crear callbacks
    # =========================================================================
    callbacks = []

    # Callback de checkpoint (si frequency > 0)
    if config.checkpoint_frequency > 0:
        checkpoint_callback = CheckpointCallback(
            output_dir=config.output_dir,
            frequency=config.checkpoint_frequency
        )
        callbacks.append(checkpoint_callback)
        print(f"  Checkpoint: cada {config.checkpoint_frequency} trials")
        logger.info(f"CheckpointCallback activado: cada {config.checkpoint_frequency} trials")

    # Callback de early stopping global (si patience > 0)
    if config.early_stopping_patience > 0:
        early_stopping_callback = EarlyStoppingCallback(
            patience=config.early_stopping_patience,
            min_delta=0.0001
        )
        callbacks.append(early_stopping_callback)
        print(f"  Early stopping global: {config.early_stopping_patience} trials sin mejora")
        logger.info(f"EarlyStoppingCallback activado: patience={config.early_stopping_patience}")
    else:
        print(f"  Early stopping global: Desactivado")

    logger.info(f"Iniciando optimización: {config.n_trials} trials, {actual_n_jobs} workers")

    # Calcular trials restantes
    n_trials_remaining = max(0, config.n_trials - n_previous_trials)
    search_time = 0  # Inicializar para el caso sin trials nuevos

    if n_trials_remaining == 0:
        print("No hay trials nuevos que ejecutar.")
        logger.info("No hay trials nuevos - usando resultados existentes")
    else:
        # Ejecutar optimizacion
        start_time = time.time()

        try:
            study.optimize(
                objective,
                n_trials=n_trials_remaining,
                timeout=config.timeout,
                n_jobs=actual_n_jobs,
                show_progress_bar=True,
                gc_after_trial=True,  # Liberar memoria después de cada trial
                callbacks=callbacks if callbacks else None  # Fase 2: callbacks
            )
        except KeyboardInterrupt:
            print("\n\nBúsqueda interrumpida por el usuario.")
            print("El progreso se ha guardado en la base de datos SQLite.")
            print("Puede continuar ejecutando el mismo comando.")
            logger.warning("Búsqueda interrumpida por KeyboardInterrupt")
        except Exception as e:
            logger.error(f"Error durante optimización: {e}")
            logger.error(tb_module.format_exc())
            raise

        search_time = time.time() - start_time
        logger.info(f"Optimización completada en {search_time/60:.2f} minutos")

    # =========================================================================
    # RESULTADOS DE LA BUSQUEDA
    # =========================================================================

    print_header("RESULTADOS DE LA BUSQUEDA")

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

    # =========================================================================
    # ANALISIS POR CATEGORIAS
    # =========================================================================

    category_analysis = analyze_results_by_category(study, config)

    # Rellenar combinaciones faltantes con configuraciones sinteticas
    category_analysis = fill_missing_combinations(category_analysis, config)

    print_category_analysis(category_analysis, config)

    # =========================================================================
    # EVALUACION DE TODAS LAS COMBINACIONES EN TEST
    # =========================================================================

    combination_results = evaluate_all_combinations(category_analysis, data, config)

    # =========================================================================
    # COMPARATIVA DE RESULTADOS EN TEST
    # =========================================================================

    winner_info = print_combination_results(combination_results)

    # =========================================================================
    # COMPARATIVA DE ENTRENAMIENTO
    # =========================================================================

    print_training_comparison(combination_results)

    # =========================================================================
    # EVALUACION DETALLADA DEL MEJOR GLOBAL (para compatibilidad)
    # =========================================================================

    # Obtener final_results del ganador o evaluar el mejor de CV
    if winner_info:
        winner_key, winner_data = winner_info
        # Usar datos del ganador para compatibilidad
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
        # Fallback: evaluar el mejor de CV
        print_header("EVALUACION DETALLADA - MEJOR CONFIGURACION GLOBAL")
        final_results = evaluate_best_config(study.best_params, data, config)

    # =========================================================================
    # GUARDAR Y VISUALIZAR
    # =========================================================================

    print_header("GUARDANDO RESULTADOS")

    save_results(
        study,
        final_results,
        config,
        config.output_dir,
        category_analysis=category_analysis,
        combination_results=combination_results
    )
    visualize_study(study, config.output_dir)

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================

    print_header("RESUMEN FINAL", "=")

    # Mostrar ganador absoluto basado en test
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
        print(f"  Precision: {winner['test_precision']:.4f}")
        print(f"  Recall: {winner['test_recall']:.4f}")
        print(f"  Kappa: {winner['test_kappa']:.4f}")

        print(f"\n  --- Arquitectura ---")
        print(f"  Capas ocultas: {winner['hidden_layers']}")
        print(f"  Parametros: {winner['n_params']:,}")

        # Comparar con mejor global de CV
        bp = study.best_params
        if (winner['optimizer'] != bp['optimizer'] or
            winner['strategy'] != bp['strategy']):
            print(f"\n  --- Nota ---")
            print(f"  El ganador en TEST difiere del mejor en CV.")
            print(f"  Mejor CV: {bp['optimizer']} + {bp['strategy'].upper()} "
                  f"(F1-CV: {study.best_value:.4f})")

    # Resumen de todas las combinaciones
    print(f"\n{'='*70}")
    print(f" RESUMEN DE COMBINACIONES EVALUADAS")
    print(f"{'='*70}")

    sorted_combos = sorted(
        combination_results.items(),
        key=lambda x: x[1].get('test_f1', 0),
        reverse=True
    )

    print(f"\n  {'Rank':<5} {'Combinacion':<22} {'F1-Test':>8} {'F1-CV':>8}")
    print(f"  {'-'*50}")
    for rank, (combo_key, result) in enumerate(sorted_combos, 1):
        if 'error' not in result:
            combo_name = f"{result['optimizer']} + {result['strategy']}"
            print(f"  {rank:<5} {combo_name:<22} {result['test_f1']:>8.4f} "
                  f"{result['f1_score_cv']:>8.4f}")

    print(f"\n{'='*70}")
    print(f" BUSQUEDA COMPLETADA")
    print(f"{'='*70}")
    print(f"\nResultados guardados en: {config.output_dir}")
    print(f"\nArchivos generados:")
    print(f"  - best_params_*.json (mejor configuracion global)")
    print(f"  - top_{config.save_top_k}_configs_*.json (top {config.save_top_k} configuraciones)")
    print(f"  - trials_history_*.csv (historial completo)")
    print(f"  - category_analysis_*.json (analisis por categorias)")
    print(f"  - combination_test_results_*.json (resultados de las 6 combinaciones en test)")
    print(f"  - hpo_search.log (log detallado de la búsqueda)")

    if config.storage_path:
        print(f"\nBase de datos SQLite: {config.storage_path}")
        print(f"  - Puede continuar la búsqueda ejecutando el mismo comando")
        print(f"  - Use --no-resume para crear un nuevo estudio")
        print(f"  - Use optuna-dashboard {config.storage_path} para visualizar")

    logger.info("Búsqueda completada exitosamente")

    return study, final_results, category_analysis, combination_results


if __name__ == "__main__":
    study, results, category_analysis, combination_results = main()
