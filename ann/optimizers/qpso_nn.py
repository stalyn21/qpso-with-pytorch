"""
Optimizador QPSO para Redes Neuronales

Este modulo implementa un optimizador que usa QPSO para entrenar
redes neuronales. En lugar de usar backpropagation, los pesos
de la red se optimizan tratandolos como particulas en un espacio
de busqueda multidimensional.

Concepto:
    - Cada particula representa un conjunto completo de pesos de la red
    - La funcion de fitness es el loss de la red (o accuracy negativo)
    - QPSO busca la mejor configuracion de pesos

Caracteristicas:
    - Soporte para QPSO y QDPSO
    - Funciones de loss configurables
    - Callbacks para monitoreo
    - Compatible con GPU/CPU
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, List, Tuple, Union, Dict, Any
from dataclasses import dataclass, field

# Importar optimizadores QPSO
from tensor_qpso import (
    QPSOTensorOptimized,
    QDPSOTensorOptimized,
    OptimizationResult,
    CallbackEvent,
    BoundaryStrategy,
    get_device
)


@dataclass
class NNOptimizationConfig:
    """Configuracion para optimizacion de redes neuronales con QPSO."""
    n_particles: int = 50
    max_iters: int = 100
    alpha: Union[float, Tuple[float, float]] = (1.0, 0.5)  # Para QPSO
    g: float = 0.96  # Para QDPSO
    weight_bound: float = 1.0
    boundary_strategy: str = "clamp"
    tol: float = 1e-12
    patience: int = 50
    seed: Optional[int] = None
    track_history: bool = True
    device: str = "auto"
    dtype: torch.dtype = torch.float32


class QPSONNOptimizer:
    """
    Optimizador basado en QPSO para redes neuronales.

    Este optimizador trata los pesos de una red neuronal como particulas
    y utiliza QPSO para encontrar la configuracion optima de pesos.

    Args:
        model: Red neuronal compatible (QPSOCompatibleANN)
        loss_fn: Funcion de perdida (default: CrossEntropyLoss)
        config: Configuracion de optimizacion
        use_qdpso: Si True, usa QDPSO en lugar de QPSO

    Example:
        >>> model = QPSOCompatibleANN(10, 3, [20, 10])
        >>> optimizer = QPSONNOptimizer(model)
        >>> result = optimizer.fit(X_train, y_train)
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[Callable] = None,
        config: Optional[NNOptimizationConfig] = None,
        use_qdpso: bool = False
    ):
        self.model = model
        self.config = config or NNOptimizationConfig()
        self.use_qdpso = use_qdpso

        # Configurar funcion de loss
        if loss_fn is None:
            self._loss_fn = nn.CrossEntropyLoss()
        else:
            self._loss_fn = loss_fn

        # Obtener informacion del modelo
        self._num_params = model.num_params
        self._bounds = model.get_param_bounds(self.config.weight_bound)

        # Configurar dispositivo
        self._device = get_device(self.config.device)

        # Estado de entrenamiento
        self._X_train: Optional[torch.Tensor] = None
        self._y_train: Optional[torch.Tensor] = None
        self._X_val: Optional[torch.Tensor] = None
        self._y_val: Optional[torch.Tensor] = None

        # Optimizador QPSO interno
        self._qpso: Optional[Union[QPSOTensorOptimized, QDPSOTensorOptimized]] = None

        # Resultados
        self._result: Optional[OptimizationResult] = None
        self._history: Dict[str, List] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def _create_fitness_function(self) -> Callable:
        """
        Crea la funcion de fitness para QPSO.

        La funcion de fitness evalua una configuracion de pesos
        calculando el loss en los datos de entrenamiento.

        Returns:
            Funcion de fitness vectorizada
        """
        def fitness(particles: torch.Tensor) -> torch.Tensor:
            """
            Evalua el fitness de multiples particulas.

            Args:
                particles: Tensor [n_particles, n_params] o [n_params]

            Returns:
                Tensor con los valores de loss
            """
            if particles.dim() == 1:
                particles = particles.unsqueeze(0)

            n_particles = particles.shape[0]
            losses = torch.zeros(n_particles, device=self._device)

            for i in range(n_particles):
                # Establecer pesos de la particula
                self.model.set_flat_params(particles[i])

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(self._X_train)
                    loss = self._loss_fn(outputs, self._y_train)
                    losses[i] = loss.item()

            if n_particles == 1:
                return losses[0]
            return losses

        return fitness

    def _create_accuracy_fitness(self) -> Callable:
        """
        Crea funcion de fitness basada en accuracy (para maximizar).

        Returns:
            Funcion de fitness que retorna accuracy negativo
        """
        def fitness(particles: torch.Tensor) -> torch.Tensor:
            if particles.dim() == 1:
                particles = particles.unsqueeze(0)

            n_particles = particles.shape[0]
            neg_accuracies = torch.zeros(n_particles, device=self._device)

            for i in range(n_particles):
                self.model.set_flat_params(particles[i])

                with torch.no_grad():
                    outputs = self.model(self._X_train)
                    preds = outputs.argmax(dim=1)
                    acc = (preds == self._y_train).float().mean()
                    neg_accuracies[i] = -acc.item()  # Negativo para minimizar

            if n_particles == 1:
                return neg_accuracies[0]
            return neg_accuracies

        return fitness

    def set_training_data(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None
    ) -> None:
        """
        Establece los datos de entrenamiento y validacion.

        Args:
            X_train: Features de entrenamiento [n_samples, n_features]
            y_train: Labels de entrenamiento [n_samples]
            X_val: Features de validacion (opcional)
            y_val: Labels de validacion (opcional)
        """
        self._X_train = X_train.to(self._device)
        self._y_train = y_train.to(self._device)

        if X_val is not None and y_val is not None:
            self._X_val = X_val.to(self._device)
            self._y_val = y_val.to(self._device)

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        fitness_type: str = "loss",
        callback: Optional[Callable] = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Entrena la red neuronal usando QPSO.

        Args:
            X_train: Features de entrenamiento
            y_train: Labels de entrenamiento
            X_val: Features de validacion (opcional)
            y_val: Labels de validacion (opcional)
            fitness_type: Tipo de fitness ("loss" o "accuracy")
            callback: Funcion callback por iteracion
            verbose: Imprimir progreso

        Returns:
            OptimizationResult con los resultados
        """
        # Establecer datos
        self.set_training_data(X_train, y_train, X_val, y_val)

        # Crear funcion de fitness
        if fitness_type == "loss":
            fitness_fn = self._create_fitness_function()
        elif fitness_type == "accuracy":
            fitness_fn = self._create_accuracy_fitness()
        else:
            raise ValueError(f"fitness_type debe ser 'loss' o 'accuracy'")

        # Crear optimizador QPSO/QDPSO
        common_params = {
            'cf': fitness_fn,
            'size': self.config.n_particles,
            'dim': self._num_params,
            'bounds': self._bounds,
            'maxIters': self.config.max_iters,
            'device': str(self._device),
            'dtype': self.config.dtype,
            'seed': self.config.seed,
            'boundary_strategy': self.config.boundary_strategy,
            'tol': self.config.tol,
            'patience': self.config.patience,
            'track_history': self.config.track_history,
            'minimize': True
        }

        if self.use_qdpso:
            self._qpso = QDPSOTensorOptimized(
                g=self.config.g,
                **common_params
            )
        else:
            self._qpso = QPSOTensorOptimized(
                alpha=self.config.alpha,
                **common_params
            )

        # Callback wrapper para registrar metricas
        def training_callback(opt):
            # Calcular metricas actuales
            current_params = opt.gbest.clone()
            self.model.set_flat_params(current_params)

            with torch.no_grad():
                # Train metrics
                train_out = self.model(self._X_train)
                train_loss = self._loss_fn(train_out, self._y_train).item()
                train_acc = (train_out.argmax(dim=1) == self._y_train).float().mean().item()

                self._history['train_loss'].append(train_loss)
                self._history['train_acc'].append(train_acc)

                # Val metrics
                if self._X_val is not None:
                    val_out = self.model(self._X_val)
                    val_loss = self._loss_fn(val_out, self._y_val).item()
                    val_acc = (val_out.argmax(dim=1) == self._y_val).float().mean().item()
                    self._history['val_loss'].append(val_loss)
                    self._history['val_acc'].append(val_acc)

            if verbose and opt.iters % 10 == 0:
                msg = f"Iter {opt.iters:4d}: loss={train_loss:.6f}, acc={train_acc:.4f}"
                if self._X_val is not None:
                    msg += f" | val_loss={val_loss:.6f}, val_acc={val_acc:.4f}"
                print(msg)

            # Callback del usuario
            if callback is not None:
                callback(opt)

        # Ejecutar optimizacion
        if verbose:
            print(f"Iniciando entrenamiento con {'QDPSO' if self.use_qdpso else 'QPSO'}")
            print(f"  Particulas: {self.config.n_particles}")
            print(f"  Parametros: {self._num_params:,}")
            print(f"  Iteraciones: {self.config.max_iters}")
            print(f"  Dispositivo: {self._device}")
            print("-" * 50)

        self._result = self._qpso.optimize(
            callback=training_callback,
            interval=1
        )

        # Establecer mejores pesos en el modelo
        self.model.set_flat_params(self._result.best_position)

        if verbose:
            print("-" * 50)
            print(f"Entrenamiento completado en {self._result.elapsed_time:.2f}s")
            print(f"  Mejor loss: {self._result.best_value:.6f}")
            print(f"  Iteraciones: {self._result.iterations}")
            print(f"  Convergencia: {self._result.convergence_reason}")

        return self._result

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Realiza predicciones con el modelo entrenado.

        Args:
            X: Features de entrada

        Returns:
            Predicciones (clases)
        """
        X = X.to(self._device)
        with torch.no_grad():
            outputs = self.model(X)
            return outputs.argmax(dim=1)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Obtiene probabilidades de clase.

        Args:
            X: Features de entrada

        Returns:
            Probabilidades por clase
        """
        X = X.to(self._device)
        with torch.no_grad():
            return self.model(X)

    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evalua el modelo en un conjunto de datos.

        Args:
            X: Features
            y: Labels verdaderos

        Returns:
            Diccionario con metricas
        """
        X = X.to(self._device)
        y = y.to(self._device)

        with torch.no_grad():
            outputs = self.model(X)
            loss = self._loss_fn(outputs, y).item()
            preds = outputs.argmax(dim=1)
            accuracy = (preds == y).float().mean().item()

        return {
            'loss': loss,
            'accuracy': accuracy
        }

    def get_history(self) -> Dict[str, List]:
        """Retorna el historial de entrenamiento."""
        return self._history

    @property
    def best_params(self) -> Optional[torch.Tensor]:
        """Retorna los mejores parametros encontrados."""
        if self._result is not None:
            return self._result.best_position
        return None


class QDPSONNOptimizer(QPSONNOptimizer):
    """
    Optimizador basado en QDPSO para redes neuronales.

    Wrapper conveniente que usa QDPSO por defecto.

    Example:
        >>> optimizer = QDPSONNOptimizer(model, config=config)
        >>> result = optimizer.fit(X_train, y_train)
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[Callable] = None,
        config: Optional[NNOptimizationConfig] = None
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            config=config,
            use_qdpso=True
        )
