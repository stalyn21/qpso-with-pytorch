"""
Estrategias de Entrenamiento para QPSO/QDPSO

Este modulo implementa diferentes estrategias de entrenamiento para
redes neuronales usando optimizadores QPSO y QDPSO.

Estrategias disponibles:
    - forward: Entrenamiento estandar (todos los pesos a la vez)
    - weighted: Forward con pesos decrecientes por capa
    - layerwise: Entrenamiento capa por capa (output -> input)

Concepto:
    - Forward: Optimiza todos los parametros simultaneamente
    - Weighted: Prioriza capas cercanas a la salida con pesos
    - Layerwise: Entrena capas secuencialmente desde output hacia input

Autor: Implementacion optimizada basada en conceptos de backward training
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# Importar optimizadores QPSO
from tensor_qpso import (
    QPSOTensorOptimized,
    QDPSOTensorOptimized,
    OptimizationResult,
    get_device
)


# =============================================================================
# ENUMS Y CONFIGURACION
# =============================================================================

class TrainingStrategy(Enum):
    """Estrategias de entrenamiento disponibles."""
    FORWARD = "forward"      # Entrenamiento estandar
    WEIGHTED = "weighted"    # Forward con pesos por capa
    LAYERWISE = "layerwise"  # Capa por capa (backward order)


@dataclass
class StrategyConfig:
    """Configuracion para estrategias de entrenamiento."""
    # Parametros comunes
    n_particles: int = 50
    max_iters: int = 100
    alpha: Union[float, Tuple[float, float]] = (1.0, 0.5)  # QPSO
    g: float = 0.96  # QDPSO
    weight_bound: float = 1.0
    patience: int = 50
    tol: float = 1e-12
    seed: Optional[int] = None
    boundary_strategy: str = 'clamp'  # Estrategia de límites: clamp, reflect, wrap, random

    # Parametros para weighted
    layer_decay: float = 0.8      # Decaimiento entre capas
    output_weight: float = 1.0    # Peso de la capa de salida
    regularization: float = 0.01  # Factor de regularizacion

    # Parametros para layerwise
    iters_per_layer: Optional[int] = None  # None = max_iters / n_layers
    fine_tune_iters: int = 50     # Iteraciones de ajuste fino final
    freeze_trained: bool = True   # Congelar capas ya entrenadas


@dataclass
class LayerInfo:
    """Informacion de una capa del modelo."""
    name: str
    module: nn.Module
    param_indices: List[Tuple[int, int]]  # (start, end) para cada param
    total_params: int
    layer_idx: int  # Indice en orden original (input -> output)


@dataclass
class TrainingResult:
    """Resultado del entrenamiento con estrategia."""
    best_loss: float
    best_accuracy: float
    iterations: int
    elapsed_time: float
    strategy: str
    history: Dict[str, List[float]]
    layer_history: Optional[Dict[str, Dict]] = None  # Para layerwise
    convergence_reason: str = ""


# =============================================================================
# CLASE BASE: ESTRATEGIA DE ENTRENAMIENTO
# =============================================================================

class BaseTrainingStrategy:
    """
    Clase base para estrategias de entrenamiento.

    Proporciona funcionalidad comun para todas las estrategias.
    """

    def __init__(
        self,
        model: nn.Module,
        config: StrategyConfig,
        use_qdpso: bool = False,
        device: str = "auto"
    ):
        self.model = model
        self.config = config
        self.use_qdpso = use_qdpso
        self._device = get_device(device)

        # Mover modelo al dispositivo
        self.model.to(self._device)

        # Loss function
        self._loss_fn = nn.CrossEntropyLoss()

        # Datos de entrenamiento
        self._X_train: Optional[torch.Tensor] = None
        self._y_train: Optional[torch.Tensor] = None
        self._X_val: Optional[torch.Tensor] = None
        self._y_val: Optional[torch.Tensor] = None

        # Obtener informacion de capas
        self._layer_info = self._extract_layer_info()
        self._num_params = sum(p.numel() for p in model.parameters())

        # Historial
        self._history: Dict[str, List] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def _extract_layer_info(self) -> List[LayerInfo]:
        """
        Extrae informacion detallada de cada capa con parametros.

        Returns:
            Lista de LayerInfo en orden (input -> output)
        """
        layer_info = []
        param_offset = 0
        layer_idx = 0

        for name, module in self.model.named_modules():
            # Solo capas con parametros propios (hojas)
            module_params = list(module.parameters(recurse=False))
            if module_params:
                param_indices = []
                total = 0

                for param in module_params:
                    numel = param.numel()
                    param_indices.append((param_offset, param_offset + numel))
                    param_offset += numel
                    total += numel

                layer_info.append(LayerInfo(
                    name=name,
                    module=module,
                    param_indices=param_indices,
                    total_params=total,
                    layer_idx=layer_idx
                ))
                layer_idx += 1

        return layer_info

    def _get_layer_params(self, layer: LayerInfo) -> torch.Tensor:
        """Obtiene los parametros de una capa como tensor plano."""
        params = []
        for param in layer.module.parameters(recurse=False):
            params.append(param.data.view(-1))
        return torch.cat(params)

    def _set_layer_params(self, layer: LayerInfo, flat_params: torch.Tensor) -> None:
        """Establece los parametros de una capa desde tensor plano."""
        offset = 0
        for param in layer.module.parameters(recurse=False):
            numel = param.numel()
            param.data.copy_(flat_params[offset:offset + numel].view(param.shape))
            offset += numel

    def _get_all_params(self) -> torch.Tensor:
        """Obtiene todos los parametros como tensor plano."""
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])

    def _set_all_params(self, flat_params: torch.Tensor) -> None:
        """Establece todos los parametros desde tensor plano."""
        offset = 0
        for param in self.model.parameters():
            numel = param.numel()
            param.data.copy_(flat_params[offset:offset + numel].view(param.shape))
            offset += numel

    def _compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Calcula loss sin gradientes."""
        with torch.no_grad():
            output = self.model(X)
            return self._loss_fn(output, y).item()

    def _compute_accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Calcula accuracy sin gradientes."""
        with torch.no_grad():
            output = self.model(X)
            preds = output.argmax(dim=1)
            return (preds == y).float().mean().item()

    def _evaluate(self) -> Dict[str, float]:
        """Evalua el modelo en train y val."""
        metrics = {
            'train_loss': self._compute_loss(self._X_train, self._y_train),
            'train_acc': self._compute_accuracy(self._X_train, self._y_train)
        }

        if self._X_val is not None:
            metrics['val_loss'] = self._compute_loss(self._X_val, self._y_val)
            metrics['val_acc'] = self._compute_accuracy(self._X_val, self._y_val)

        return metrics

    def _create_qpso_optimizer(
        self,
        fitness_fn: Callable,
        dim: int,
        bounds: List[Tuple[float, float]],
        max_iters: int
    ) -> Union[QPSOTensorOptimized, QDPSOTensorOptimized]:
        """Crea un optimizador QPSO o QDPSO."""
        common_params = {
            'cf': fitness_fn,
            'size': self.config.n_particles,
            'dim': dim,
            'bounds': bounds,
            'maxIters': max_iters,
            'device': str(self._device),
            'dtype': torch.float32,
            'seed': self.config.seed,
            'boundary_strategy': self.config.boundary_strategy,
            'tol': self.config.tol,
            'patience': self.config.patience,
            'track_history': True,
            'minimize': True
        }

        if self.use_qdpso:
            return QDPSOTensorOptimized(g=self.config.g, **common_params)
        else:
            return QPSOTensorOptimized(alpha=self.config.alpha, **common_params)

    def set_data(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None
    ) -> None:
        """Establece los datos de entrenamiento."""
        self._X_train = X_train.to(self._device)
        self._y_train = y_train.to(self._device)

        if X_val is not None and y_val is not None:
            self._X_val = X_val.to(self._device)
            self._y_val = y_val.to(self._device)

    def train(self, verbose: bool = True) -> TrainingResult:
        """Metodo abstracto para entrenar."""
        raise NotImplementedError


# =============================================================================
# ESTRATEGIA: FORWARD (ESTANDAR)
# =============================================================================

class ForwardStrategy(BaseTrainingStrategy):
    """
    Estrategia de entrenamiento Forward (estandar).

    Optimiza todos los pesos de la red simultaneamente.
    Es el metodo tradicional usado en QPSO.
    """

    def _create_fitness_fn(self) -> Callable:
        """Crea funcion de fitness para todos los parametros."""
        def fitness(particles: torch.Tensor) -> torch.Tensor:
            if particles.dim() == 1:
                particles = particles.unsqueeze(0)

            n_particles = particles.shape[0]
            losses = torch.zeros(n_particles, device=self._device)

            for i in range(n_particles):
                self._set_all_params(particles[i])
                losses[i] = self._compute_loss(self._X_train, self._y_train)

            return losses if n_particles > 1 else losses[0]

        return fitness

    def train(self, verbose: bool = True) -> TrainingResult:
        """Ejecuta entrenamiento forward."""
        import time
        start_time = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f" ESTRATEGIA: FORWARD")
            print(f" Optimizador: {'QDPSO' if self.use_qdpso else 'QPSO'}")
            print(f" Parametros: {self._num_params:,}")
            print(f"{'='*60}")

        # Crear bounds
        bounds = [(-self.config.weight_bound, self.config.weight_bound)] * self._num_params

        # Crear optimizador
        optimizer = self._create_qpso_optimizer(
            fitness_fn=self._create_fitness_fn(),
            dim=self._num_params,
            bounds=bounds,
            max_iters=self.config.max_iters
        )

        # Callback para historial
        def callback(opt):
            self._set_all_params(opt.gbest)
            metrics = self._evaluate()

            self._history['train_loss'].append(metrics['train_loss'])
            self._history['train_acc'].append(metrics['train_acc'])

            if 'val_loss' in metrics:
                self._history['val_loss'].append(metrics['val_loss'])
                self._history['val_acc'].append(metrics['val_acc'])

            if verbose and opt.iters % 10 == 0:
                msg = f"Iter {opt.iters:4d}: loss={metrics['train_loss']:.6f}, acc={metrics['train_acc']:.4f}"
                if 'val_loss' in metrics:
                    msg += f" | val_loss={metrics['val_loss']:.6f}, val_acc={metrics['val_acc']:.4f}"
                print(msg)

        # Optimizar
        result = optimizer.optimize(callback=callback, interval=1)

        # Establecer mejores pesos
        self._set_all_params(result.best_position)

        elapsed = time.time() - start_time
        final_metrics = self._evaluate()

        if verbose:
            print(f"\nCompletado en {elapsed:.2f}s")
            print(f"Loss: {final_metrics['train_loss']:.6f}, Acc: {final_metrics['train_acc']:.4f}")

        return TrainingResult(
            best_loss=final_metrics['train_loss'],
            best_accuracy=final_metrics['train_acc'],
            iterations=result.iterations,
            elapsed_time=elapsed,
            strategy='forward',
            history=self._history,
            convergence_reason=result.convergence_reason
        )


# =============================================================================
# ESTRATEGIA: WEIGHTED
# =============================================================================

class WeightedStrategy(BaseTrainingStrategy):
    """
    Estrategia de entrenamiento Weighted.

    Forward pass con pesos decrecientes por capa. Las capas
    cercanas a la salida tienen mayor influencia en la funcion
    de fitness, permitiendo que se ajusten primero.

    Concepto:
        - Capa de salida: peso = 1.0
        - Capas intermedias: peso = layer_decay^(distancia_a_salida)
        - Se usa regularizacion para estabilizar capas tempranas
    """

    def _compute_layer_weights(self) -> List[float]:
        """
        Calcula los pesos para cada capa.

        Capas cercanas a la salida tienen mayor peso.
        """
        n_layers = len(self._layer_info)
        weights = []

        for layer in self._layer_info:
            # Distancia desde la salida (ultima capa = 0)
            dist_from_output = n_layers - 1 - layer.layer_idx
            weight = self.config.output_weight * (self.config.layer_decay ** dist_from_output)
            weights.append(weight)

        return weights

    def _forward_with_layer_outputs(self, X: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass capturando salidas de cada capa.

        Returns:
            Lista de tensores con salida de cada capa
        """
        layer_outputs = []

        hooks = []
        def make_hook(idx):
            def hook(module, input, output):
                layer_outputs.append(output.detach())
            return hook

        # Registrar hooks
        for i, layer in enumerate(self._layer_info):
            hooks.append(layer.module.register_forward_hook(make_hook(i)))

        # Forward
        with torch.no_grad():
            self.model(X)

        # Limpiar hooks
        for h in hooks:
            h.remove()

        return layer_outputs

    def _compute_weighted_loss(self) -> float:
        """
        Calcula loss ponderado por capas.

        Returns:
            Loss total con ponderacion
        """
        # Loss principal
        main_loss = self._compute_loss(self._X_train, self._y_train)

        # Obtener salidas por capa
        layer_outputs = self._forward_with_layer_outputs(self._X_train)
        layer_weights = self._compute_layer_weights()

        # Regularizacion por varianza de activaciones
        # Penaliza alta varianza en capas tempranas (estabilidad)
        regularization = 0.0
        for i, (output, weight) in enumerate(zip(layer_outputs, layer_weights)):
            if output.dim() >= 2:  # Asegurar que es un tensor valido
                # Invertir peso: mayor regularizacion para capas tempranas
                inv_weight = 1.0 - weight + 0.1
                var = torch.var(output).item()
                regularization += inv_weight * var

        # Loss total
        total_loss = main_loss + self.config.regularization * regularization

        return total_loss

    def _create_fitness_fn(self) -> Callable:
        """Crea funcion de fitness ponderada."""
        def fitness(particles: torch.Tensor) -> torch.Tensor:
            if particles.dim() == 1:
                particles = particles.unsqueeze(0)

            n_particles = particles.shape[0]
            losses = torch.zeros(n_particles, device=self._device)

            for i in range(n_particles):
                self._set_all_params(particles[i])
                losses[i] = self._compute_weighted_loss()

            return losses if n_particles > 1 else losses[0]

        return fitness

    def train(self, verbose: bool = True) -> TrainingResult:
        """Ejecuta entrenamiento weighted."""
        import time
        start_time = time.time()

        layer_weights = self._compute_layer_weights()

        if verbose:
            print(f"\n{'='*60}")
            print(f" ESTRATEGIA: WEIGHTED")
            print(f" Optimizador: {'QDPSO' if self.use_qdpso else 'QPSO'}")
            print(f" Parametros: {self._num_params:,}")
            print(f" Layer decay: {self.config.layer_decay}")
            print(f" Regularization: {self.config.regularization}")
            print(f"{'='*60}")
            print(f" Pesos por capa (output -> input):")
            for layer, weight in zip(reversed(self._layer_info), reversed(layer_weights)):
                print(f"   {layer.name}: {weight:.4f} ({layer.total_params} params)")

        # Crear bounds
        bounds = [(-self.config.weight_bound, self.config.weight_bound)] * self._num_params

        # Crear optimizador
        optimizer = self._create_qpso_optimizer(
            fitness_fn=self._create_fitness_fn(),
            dim=self._num_params,
            bounds=bounds,
            max_iters=self.config.max_iters
        )

        # Callback
        def callback(opt):
            self._set_all_params(opt.gbest)
            metrics = self._evaluate()

            self._history['train_loss'].append(metrics['train_loss'])
            self._history['train_acc'].append(metrics['train_acc'])

            if 'val_loss' in metrics:
                self._history['val_loss'].append(metrics['val_loss'])
                self._history['val_acc'].append(metrics['val_acc'])

            if verbose and opt.iters % 10 == 0:
                msg = f"Iter {opt.iters:4d}: loss={metrics['train_loss']:.6f}, acc={metrics['train_acc']:.4f}"
                if 'val_loss' in metrics:
                    msg += f" | val_loss={metrics['val_loss']:.6f}, val_acc={metrics['val_acc']:.4f}"
                print(msg)

        # Optimizar
        result = optimizer.optimize(callback=callback, interval=1)

        # Establecer mejores pesos
        self._set_all_params(result.best_position)

        elapsed = time.time() - start_time
        final_metrics = self._evaluate()

        if verbose:
            print(f"\nCompletado en {elapsed:.2f}s")
            print(f"Loss: {final_metrics['train_loss']:.6f}, Acc: {final_metrics['train_acc']:.4f}")

        return TrainingResult(
            best_loss=final_metrics['train_loss'],
            best_accuracy=final_metrics['train_acc'],
            iterations=result.iterations,
            elapsed_time=elapsed,
            strategy='weighted',
            history=self._history,
            convergence_reason=result.convergence_reason
        )


# =============================================================================
# ESTRATEGIA: LAYERWISE
# =============================================================================

class LayerwiseStrategy(BaseTrainingStrategy):
    """
    Estrategia de entrenamiento Layerwise (capa por capa).

    Entrena las capas secuencialmente desde la salida hacia la entrada.
    Cada capa se optimiza individualmente mientras las demas se congelan.

    Concepto:
        1. Entrenar capa de salida (la mas cercana a las labels)
        2. Congelar capa de salida
        3. Entrenar siguiente capa hacia atras
        4. Repetir hasta la capa de entrada
        5. (Opcional) Fine-tuning de toda la red
    """

    def _create_layer_fitness_fn(self, layer: LayerInfo) -> Callable:
        """
        Crea funcion de fitness para una capa especifica.

        Solo modifica los parametros de la capa dada.
        """
        def fitness(particles: torch.Tensor) -> torch.Tensor:
            if particles.dim() == 1:
                particles = particles.unsqueeze(0)

            n_particles = particles.shape[0]
            losses = torch.zeros(n_particles, device=self._device)

            # Guardar parametros originales de la capa
            original_params = self._get_layer_params(layer).clone()

            for i in range(n_particles):
                # Solo modificar esta capa
                self._set_layer_params(layer, particles[i])
                losses[i] = self._compute_loss(self._X_train, self._y_train)

            # Restaurar para la siguiente evaluacion batch
            self._set_layer_params(layer, original_params)

            return losses if n_particles > 1 else losses[0]

        return fitness

    def _train_single_layer(
        self,
        layer: LayerInfo,
        max_iters: int,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        Entrena una sola capa.

        Returns:
            Diccionario con resultados del entrenamiento de la capa
        """
        # Crear bounds para esta capa
        bounds = [(-self.config.weight_bound, self.config.weight_bound)] * layer.total_params

        # Crear fitness para esta capa
        fitness_fn = self._create_layer_fitness_fn(layer)

        # Crear optimizador
        optimizer = self._create_qpso_optimizer(
            fitness_fn=fitness_fn,
            dim=layer.total_params,
            bounds=bounds,
            max_iters=max_iters
        )

        # Historial de la capa
        layer_history = {'loss': [], 'acc': []}

        def callback(opt):
            self._set_layer_params(layer, opt.gbest)
            loss = self._compute_loss(self._X_train, self._y_train)
            acc = self._compute_accuracy(self._X_train, self._y_train)
            layer_history['loss'].append(loss)
            layer_history['acc'].append(acc)

            if verbose and opt.iters % 10 == 0:
                print(f"    Iter {opt.iters:4d}: loss={loss:.6f}, acc={acc:.4f}")

        # Optimizar
        result = optimizer.optimize(callback=callback, interval=1)

        # Establecer mejores parametros
        self._set_layer_params(layer, result.best_position)

        return {
            'best_loss': result.best_value,
            'iterations': result.iterations,
            'history': layer_history
        }

    def train(self, verbose: bool = True) -> TrainingResult:
        """Ejecuta entrenamiento layerwise."""
        import time
        start_time = time.time()

        n_layers = len(self._layer_info)

        # Calcular iteraciones por capa
        if self.config.iters_per_layer is None:
            iters_per_layer = max(10, self.config.max_iters // (n_layers + 1))
        else:
            iters_per_layer = self.config.iters_per_layer

        if verbose:
            print(f"\n{'='*60}")
            print(f" ESTRATEGIA: LAYERWISE (Backward)")
            print(f" Optimizador: {'QDPSO' if self.use_qdpso else 'QPSO'}")
            print(f" Capas: {n_layers}")
            print(f" Iteraciones por capa: {iters_per_layer}")
            print(f" Fine-tune iterations: {self.config.fine_tune_iters}")
            print(f"{'='*60}")

        layer_results = {}
        total_iters = 0

        # Entrenar capas en orden inverso (output -> input)
        for layer in reversed(self._layer_info):
            if verbose:
                print(f"\n--- Capa: {layer.name} ({layer.total_params} params) ---")

            result = self._train_single_layer(layer, iters_per_layer, verbose)
            layer_results[layer.name] = result
            total_iters += result['iterations']

            # Registrar en historial global
            metrics = self._evaluate()
            self._history['train_loss'].append(metrics['train_loss'])
            self._history['train_acc'].append(metrics['train_acc'])
            if 'val_loss' in metrics:
                self._history['val_loss'].append(metrics['val_loss'])
                self._history['val_acc'].append(metrics['val_acc'])

        # Fine-tuning opcional (todos los parametros)
        if self.config.fine_tune_iters > 0:
            if verbose:
                print(f"\n--- Fine-tuning (todas las capas) ---")

            bounds = [(-self.config.weight_bound, self.config.weight_bound)] * self._num_params

            # Usar los pesos actuales como punto de partida
            # (el optimizador inicializara aleatoriamente pero con suerte convergera rapido)

            def fitness(particles: torch.Tensor) -> torch.Tensor:
                if particles.dim() == 1:
                    particles = particles.unsqueeze(0)

                n_particles = particles.shape[0]
                losses = torch.zeros(n_particles, device=self._device)

                for i in range(n_particles):
                    self._set_all_params(particles[i])
                    losses[i] = self._compute_loss(self._X_train, self._y_train)

                return losses if n_particles > 1 else losses[0]

            optimizer = self._create_qpso_optimizer(
                fitness_fn=fitness,
                dim=self._num_params,
                bounds=bounds,
                max_iters=self.config.fine_tune_iters
            )

            def callback(opt):
                self._set_all_params(opt.gbest)
                metrics = self._evaluate()
                self._history['train_loss'].append(metrics['train_loss'])
                self._history['train_acc'].append(metrics['train_acc'])
                if 'val_loss' in metrics:
                    self._history['val_loss'].append(metrics['val_loss'])
                    self._history['val_acc'].append(metrics['val_acc'])

                if verbose and opt.iters % 10 == 0:
                    msg = f"    Iter {opt.iters:4d}: loss={metrics['train_loss']:.6f}, acc={metrics['train_acc']:.4f}"
                    print(msg)

            ft_result = optimizer.optimize(callback=callback, interval=1)
            self._set_all_params(ft_result.best_position)
            total_iters += ft_result.iterations

        elapsed = time.time() - start_time
        final_metrics = self._evaluate()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Completado en {elapsed:.2f}s")
            print(f"Iteraciones totales: {total_iters}")
            print(f"Loss: {final_metrics['train_loss']:.6f}, Acc: {final_metrics['train_acc']:.4f}")
            if 'val_loss' in final_metrics:
                print(f"Val Loss: {final_metrics['val_loss']:.6f}, Val Acc: {final_metrics['val_acc']:.4f}")

        return TrainingResult(
            best_loss=final_metrics['train_loss'],
            best_accuracy=final_metrics['train_acc'],
            iterations=total_iters,
            elapsed_time=elapsed,
            strategy='layerwise',
            history=self._history,
            layer_history=layer_results,
            convergence_reason="Completed all layers"
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_training_strategy(
    model: nn.Module,
    strategy: Union[str, TrainingStrategy],
    config: Optional[StrategyConfig] = None,
    use_qdpso: bool = False,
    device: str = "auto"
) -> BaseTrainingStrategy:
    """
    Factory function para crear estrategias de entrenamiento.

    Args:
        model: Modelo a entrenar
        strategy: Tipo de estrategia ('forward', 'weighted', 'layerwise')
        config: Configuracion de la estrategia
        use_qdpso: Usar QDPSO en lugar de QPSO
        device: Dispositivo de computo

    Returns:
        Instancia de la estrategia solicitada

    Example:
        >>> strategy = create_training_strategy(model, 'weighted', use_qdpso=True)
        >>> strategy.set_data(X_train, y_train, X_val, y_val)
        >>> result = strategy.train()
    """
    if isinstance(strategy, str):
        strategy = TrainingStrategy(strategy.lower())

    config = config or StrategyConfig()

    strategies = {
        TrainingStrategy.FORWARD: ForwardStrategy,
        TrainingStrategy.WEIGHTED: WeightedStrategy,
        TrainingStrategy.LAYERWISE: LayerwiseStrategy,
    }

    if strategy not in strategies:
        raise ValueError(f"Estrategia no soportada: {strategy}")

    return strategies[strategy](model, config, use_qdpso, device)
