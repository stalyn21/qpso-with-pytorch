"""
Trainer para Redes Neuronales con QPSO

Proporciona una interfaz de alto nivel para entrenar, validar y evaluar
redes neuronales usando optimizadores QPSO con soporte para:
- Cross-validation
- Early stopping
- Guardado de checkpoints
- Logging de metricas
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
import time
import json
import os
from datetime import datetime

# Importar componentes internos
from models import QPSOCompatibleANN
from optimizers import QPSONNOptimizer, QDPSONNOptimizer
from optimizers.qpso_nn import NNOptimizationConfig


@dataclass
class TrainingConfig:
    """
    Configuracion completa de entrenamiento.

    Agrupa configuracion de optimizador, entrenamiento y evaluacion.
    """
    # Configuracion del modelo
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    activation: str = "relu"
    dropout: float = 0.0
    weight_bound: float = 1.0

    # Configuracion QPSO
    n_particles: int = 50
    max_iters: int = 100
    alpha: Union[float, Tuple[float, float]] = (1.0, 0.5)
    g: float = 0.96
    use_qdpso: bool = False
    boundary_strategy: str = "clamp"
    tol: float = 1e-12
    patience: int = 50

    # Configuracion de entrenamiento
    n_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    device: str = "auto"

    # Output
    output_dir: str = "./output"
    save_best_model: bool = True
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuracion a diccionario."""
        return {
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'dropout': self.dropout,
            'weight_bound': self.weight_bound,
            'n_particles': self.n_particles,
            'max_iters': self.max_iters,
            'alpha': self.alpha if isinstance(self.alpha, float) else list(self.alpha),
            'g': self.g,
            'use_qdpso': self.use_qdpso,
            'boundary_strategy': self.boundary_strategy,
            'tol': self.tol,
            'patience': self.patience,
            'n_folds': self.n_folds,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'device': self.device,
        }


@dataclass
class TrainingResult:
    """Resultado del entrenamiento."""
    best_model_params: torch.Tensor
    train_accuracy: float
    val_accuracy: float
    test_accuracy: Optional[float]
    train_loss: float
    val_loss: float
    test_loss: Optional[float]
    training_time: float
    n_iterations: int
    convergence_reason: str
    history: Dict[str, List]
    config: TrainingConfig
    fold_results: Optional[List[Dict]] = None


class Trainer:
    """
    Trainer de alto nivel para redes neuronales con QPSO.

    Proporciona una interfaz simplificada para entrenar modelos
    con soporte para cross-validation, evaluacion y guardado.

    Args:
        input_dim: Dimension de entrada
        output_dim: Numero de clases
        config: Configuracion de entrenamiento

    Example:
        >>> config = TrainingConfig(
        ...     hidden_layers=[32, 16],
        ...     n_particles=30,
        ...     max_iters=100
        ... )
        >>> trainer = Trainer(input_dim=10, output_dim=3, config=config)
        >>> result = trainer.fit(X_train, y_train, X_test, y_test)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: Optional[TrainingConfig] = None
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config or TrainingConfig()

        # Modelo y optimizador
        self._model: Optional[QPSOCompatibleANN] = None
        self._optimizer: Optional[QPSONNOptimizer] = None

        # Resultados
        self._best_result: Optional[TrainingResult] = None
        self._all_results: List[TrainingResult] = []

        # Configurar dispositivo
        if self.config.device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.config.device)

    def _create_model(self) -> QPSOCompatibleANN:
        """Crea una nueva instancia del modelo."""
        return QPSOCompatibleANN(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.config.hidden_layers.copy(),
            activation=self.config.activation,
            dropout=self.config.dropout,
            device=str(self._device)
        )

    def _create_optimizer(self, model: QPSOCompatibleANN) -> QPSONNOptimizer:
        """Crea el optimizador para el modelo."""
        opt_config = NNOptimizationConfig(
            n_particles=self.config.n_particles,
            max_iters=self.config.max_iters,
            alpha=self.config.alpha,
            g=self.config.g,
            weight_bound=self.config.weight_bound,
            boundary_strategy=self.config.boundary_strategy,
            tol=self.config.tol,
            patience=self.config.patience,
            seed=self.config.random_state,
            track_history=True,
            device=str(self._device)
        )

        if self.config.use_qdpso:
            return QDPSONNOptimizer(model, config=opt_config)
        else:
            return QPSONNOptimizer(model, config=opt_config)

    def _prepare_data(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convierte datos a tensores."""
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.long)

        return X.to(self._device), y.to(self._device)

    def fit(
        self,
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_test: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> TrainingResult:
        """
        Entrena el modelo con los datos proporcionados.

        Args:
            X_train: Features de entrenamiento
            y_train: Labels de entrenamiento
            X_val: Features de validacion (opcional, se crea split si no se proporciona)
            y_val: Labels de validacion
            X_test: Features de test (opcional)
            y_test: Labels de test

        Returns:
            TrainingResult con los resultados del entrenamiento
        """
        start_time = time.time()

        # Preparar datos
        X_train, y_train = self._prepare_data(X_train, y_train)

        # Crear split de validacion si no se proporciona
        if X_val is None:
            n_samples = X_train.shape[0]
            n_val = int(n_samples * self.config.test_size)
            indices = torch.randperm(n_samples)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            X_val = X_train[val_indices]
            y_val = y_train[val_indices]
            X_train = X_train[train_indices]
            y_train = y_train[train_indices]
        else:
            X_val, y_val = self._prepare_data(X_val, y_val)

        # Preparar datos de test si existen
        if X_test is not None:
            X_test, y_test = self._prepare_data(X_test, y_test)

        # Crear modelo y optimizador
        self._model = self._create_model()
        self._optimizer = self._create_optimizer(self._model)

        if self.config.verbose:
            print("=" * 60)
            print("ENTRENAMIENTO DE RED NEURONAL CON QPSO")
            print("=" * 60)
            print(f"Modelo: {self._model}")
            print(f"Datos: train={len(y_train)}, val={len(y_val)}")
            if X_test is not None:
                print(f"       test={len(y_test)}")
            print("=" * 60)

        # Entrenar
        opt_result = self._optimizer.fit(
            X_train, y_train,
            X_val, y_val,
            verbose=self.config.verbose
        )

        training_time = time.time() - start_time

        # Evaluar en train y val
        train_metrics = self._optimizer.evaluate(X_train, y_train)
        val_metrics = self._optimizer.evaluate(X_val, y_val)

        # Evaluar en test si existe
        test_metrics = None
        if X_test is not None:
            test_metrics = self._optimizer.evaluate(X_test, y_test)

        # Crear resultado
        result = TrainingResult(
            best_model_params=self._optimizer.best_params.clone(),
            train_accuracy=train_metrics['accuracy'],
            val_accuracy=val_metrics['accuracy'],
            test_accuracy=test_metrics['accuracy'] if test_metrics else None,
            train_loss=train_metrics['loss'],
            val_loss=val_metrics['loss'],
            test_loss=test_metrics['loss'] if test_metrics else None,
            training_time=training_time,
            n_iterations=opt_result.iterations,
            convergence_reason=opt_result.convergence_reason,
            history=self._optimizer.get_history(),
            config=self.config
        )

        self._best_result = result
        self._all_results.append(result)

        if self.config.verbose:
            print("=" * 60)
            print("RESULTADOS FINALES")
            print("=" * 60)
            print(f"Train: loss={result.train_loss:.6f}, acc={result.train_accuracy:.4f}")
            print(f"Val:   loss={result.val_loss:.6f}, acc={result.val_accuracy:.4f}")
            if result.test_accuracy is not None:
                print(f"Test:  loss={result.test_loss:.6f}, acc={result.test_accuracy:.4f}")
            print(f"Tiempo: {training_time:.2f}s")
            print("=" * 60)

        # Guardar modelo si esta configurado
        if self.config.save_best_model:
            self.save_model()

        return result

    def fit_cv(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_test: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> TrainingResult:
        """
        Entrena con cross-validation.

        Args:
            X: Features
            y: Labels
            X_test: Features de test (opcional)
            y_test: Labels de test (opcional)

        Returns:
            TrainingResult con resultados promediados
        """
        start_time = time.time()

        # Convertir a numpy para sklearn
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
            y_np = y.cpu().numpy()
        else:
            X_np = X
            y_np = y

        # KFold
        from sklearn.model_selection import KFold
        kfold = KFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )

        fold_results = []
        best_val_acc = 0
        best_params = None

        if self.config.verbose:
            print("=" * 60)
            print(f"CROSS-VALIDATION CON {self.config.n_folds} FOLDS")
            print("=" * 60)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_np)):
            if self.config.verbose:
                print(f"\n--- Fold {fold + 1}/{self.config.n_folds} ---")

            # Preparar datos del fold
            X_train = torch.tensor(X_np[train_idx], dtype=torch.float32).to(self._device)
            y_train = torch.tensor(y_np[train_idx], dtype=torch.long).to(self._device)
            X_val = torch.tensor(X_np[val_idx], dtype=torch.float32).to(self._device)
            y_val = torch.tensor(y_np[val_idx], dtype=torch.long).to(self._device)

            # Crear nuevo modelo y optimizador
            model = self._create_model()
            optimizer = self._create_optimizer(model)

            # Entrenar
            opt_result = optimizer.fit(
                X_train, y_train,
                X_val, y_val,
                verbose=self.config.verbose
            )

            # Evaluar
            train_metrics = optimizer.evaluate(X_train, y_train)
            val_metrics = optimizer.evaluate(X_val, y_val)

            fold_results.append({
                'fold': fold + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'iterations': opt_result.iterations
            })

            # Guardar mejor modelo
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_params = optimizer.best_params.clone()
                self._model = model
                self._optimizer = optimizer

        training_time = time.time() - start_time

        # Calcular promedios
        avg_train_loss = np.mean([r['train_loss'] for r in fold_results])
        avg_train_acc = np.mean([r['train_acc'] for r in fold_results])
        avg_val_loss = np.mean([r['val_loss'] for r in fold_results])
        avg_val_acc = np.mean([r['val_acc'] for r in fold_results])

        # Evaluar en test si existe
        test_metrics = None
        if X_test is not None:
            X_test, y_test = self._prepare_data(X_test, y_test)
            self._model.set_flat_params(best_params)
            test_metrics = self._optimizer.evaluate(X_test, y_test)

        result = TrainingResult(
            best_model_params=best_params,
            train_accuracy=avg_train_acc,
            val_accuracy=avg_val_acc,
            test_accuracy=test_metrics['accuracy'] if test_metrics else None,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            test_loss=test_metrics['loss'] if test_metrics else None,
            training_time=training_time,
            n_iterations=sum(r['iterations'] for r in fold_results),
            convergence_reason="cross-validation completed",
            history={},
            config=self.config,
            fold_results=fold_results
        )

        self._best_result = result

        if self.config.verbose:
            print("\n" + "=" * 60)
            print("RESULTADOS CROSS-VALIDATION")
            print("=" * 60)
            print(f"Train: loss={avg_train_loss:.6f} +/- {np.std([r['train_loss'] for r in fold_results]):.6f}")
            print(f"       acc={avg_train_acc:.4f} +/- {np.std([r['train_acc'] for r in fold_results]):.4f}")
            print(f"Val:   loss={avg_val_loss:.6f} +/- {np.std([r['val_loss'] for r in fold_results]):.6f}")
            print(f"       acc={avg_val_acc:.4f} +/- {np.std([r['val_acc'] for r in fold_results]):.4f}")
            if test_metrics:
                print(f"Test:  loss={test_metrics['loss']:.6f}, acc={test_metrics['accuracy']:.4f}")
            print(f"Tiempo total: {training_time:.2f}s")
            print("=" * 60)

        if self.config.save_best_model:
            self.save_model()

        return result

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.

        Args:
            X: Features de entrada

        Returns:
            Array numpy con predicciones
        """
        if self._model is None:
            raise ValueError("Modelo no entrenado. Ejecute fit() primero.")

        X, _ = self._prepare_data(X, torch.zeros(X.shape[0]))
        preds = self._optimizer.predict(X)
        return preds.cpu().numpy()

    def predict_proba(
        self,
        X: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Obtiene probabilidades de clase.

        Args:
            X: Features de entrada

        Returns:
            Array numpy con probabilidades
        """
        if self._model is None:
            raise ValueError("Modelo no entrenado. Ejecute fit() primero.")

        X, _ = self._prepare_data(X, torch.zeros(X.shape[0]))
        probs = self._optimizer.predict_proba(X)
        return probs.cpu().numpy()

    def save_model(
        self,
        path: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Guarda el modelo entrenado.

        Args:
            path: Directorio de salida
            filename: Nombre del archivo

        Returns:
            Ruta del archivo guardado
        """
        if self._model is None or self._best_result is None:
            raise ValueError("No hay modelo para guardar.")

        # Configurar ruta
        if path is None:
            path = self.config.output_dir
        os.makedirs(path, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            acc = self._best_result.val_accuracy
            filename = f"qpso_nn_acc{acc:.4f}_{timestamp}.pth"

        filepath = os.path.join(path, filename)

        # Guardar
        torch.save({
            'model_params': self._best_result.best_model_params,
            'model_config': self._model.get_architecture_info(),
            'training_config': self.config.to_dict(),
            'results': {
                'train_accuracy': self._best_result.train_accuracy,
                'val_accuracy': self._best_result.val_accuracy,
                'test_accuracy': self._best_result.test_accuracy,
                'training_time': self._best_result.training_time,
                'n_iterations': self._best_result.n_iterations,
            }
        }, filepath)

        if self.config.verbose:
            print(f"Modelo guardado en: {filepath}")

        return filepath

    def load_model(self, filepath: str) -> None:
        """
        Carga un modelo guardado.

        Args:
            filepath: Ruta al archivo del modelo
        """
        checkpoint = torch.load(filepath, map_location=self._device)

        # Reconstruir modelo
        model_config = checkpoint['model_config']
        self._model = QPSOCompatibleANN(
            input_dim=model_config['input_dim'],
            output_dim=model_config['output_dim'],
            hidden_layers=model_config['hidden_layers'],
            activation=model_config['activation'],
            output_activation=model_config['output_activation'],
            dropout=model_config['dropout'],
            use_batch_norm=model_config['use_batch_norm'],
            device=str(self._device)
        )

        # Cargar pesos
        self._model.set_flat_params(checkpoint['model_params'])

        # Recrear optimizador
        self._optimizer = self._create_optimizer(self._model)

        if self.config.verbose:
            print(f"Modelo cargado desde: {filepath}")
            print(f"  Accuracy val: {checkpoint['results']['val_accuracy']:.4f}")

    @property
    def model(self) -> Optional[QPSOCompatibleANN]:
        """Retorna el modelo actual."""
        return self._model

    @property
    def best_result(self) -> Optional[TrainingResult]:
        """Retorna el mejor resultado de entrenamiento."""
        return self._best_result
