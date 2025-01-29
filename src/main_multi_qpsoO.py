import torch
import time
import numpy as np
from sklearn.model_selection import KFold
import logging

# Custom optimizer and model
from data.benchmarck import load_and_preprocess_data
from metrics.plotting import plot_cross_validation_losses
from metrics.metrics import MulticlassMetrics
from multi_swarm.custome_optimizer.qpso_optimizerO import LayerQDPSOoOptimizer
from multi_swarm.ann.model_qpsoO import ExtendedModel

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_best_model(model, config, best_acc):
    """
    Save the model state and configuration to a file.

    Args:
        model (OrderedDict): The state dictionary of the model.
        config (dict): The configuration dictionary of the model.
        best_acc (str): The best model accuracy.
    """
    model_path = f"./models/{config['optimizer']}/{config['dataset']}_{config['optimizer']}_best_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_acc': best_acc
    }, model_path)
    logging.info(f"Best model saved at: {model_path}")


def main():
    """
    Main function to train and evaluate the model using QPSO optimization.

    This function performs the following steps:
    1. Load and preprocess the dataset
    2. Set up the model and optimizer
    3. Train the model using k-fold cross-validation
    4. Evaluate the model on the test set
    5. Save the best model
    6. Log the results
    """

    best_acc = 0
    best_model = None
    metrics_calculator = MulticlassMetrics()

    # Load and preprocess the data (dataset options: iris, breast_cancer, wine, circle)
    dataset_name = 'circle'

    X_train_val, X_test, y_train_val, y_test = load_and_preprocess_data(dataset_name)

    input_shape = X_train_val.shape[1]
    output_shape = len(np.unique(y_train_val))
    n_samples = X_train_val.shape[0]

    config = {
        'optimizer': "multi_QPSOo",
        'dataset': dataset_name,
        'input_dim': input_shape,
        'output_dim': output_shape,
        'n_samples': n_samples,
        'hidden_layers': [input_shape * 3, input_shape * 2, input_shape],
        'n_particles': 20,
        'g': 1.13,
        'interval_parms_updated': 10,
        'n_folds': 4,
        'n_epochs': 100  # Max iterations for training
    }

    # Logging Setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"./output/{config['optimizer']}/{dataset_name}_{config['optimizer']}.output"),
            logging.StreamHandler()
        ]
    )

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    model = ExtendedModel(
        num_samples=config['n_samples'],
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        hidden_layers=config['hidden_layers']
    ).to(device)
    total_params = model.get_flat_params()

    kf = KFold(n_splits=config['n_folds'], shuffle=True, random_state=100)

    swarms = []
    logging.info(f"=============================================")
    logging.info(f"Parameter numbers to train: {total_params.numel()}")
    for i in range(len(model.layers)):
        swarm = LayerQDPSOoOptimizer(
            model, i, total_params,
            n_particles=config['n_particles'],
            max_iters=config['n_epochs'],
            g=config['g'],
            interval_parms_updated=config['interval_parms_updated']
        )
        swarms.append(swarm)
        logging.info(f"Layer {i} swarm initialized with {swarm.dim} dimensions for each particle.")
    logging.info(f"=============================================")

    train_results = []
    val_results = []
    test_results = []

    swarm_times = np.zeros((config['n_folds'], len(model.layers)))  # Matriz para tiempos por capa
    time_results = []

    all_losses = []  # Lista para almacenar las pérdidas de todos los folds

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_val), 1):
        logging.info(f"======= Fold {fold} =======")
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.long).to(device)

        y_train_one_hot = torch.nn.functional.one_hot(y_train, num_classes=config['output_dim']).float().to(device)
        y_val_one_hot = torch.nn.functional.one_hot(y_val, num_classes=config['output_dim']).float().to(device)

        fold_losses = {'train': [], 'val': []}  # Pérdidas para este fold

        with torch.no_grad():  # Disable gradient tracking
            training_start_time = time.perf_counter()
            for layer_idx, optimizer in enumerate(reversed(swarms)):
                optimizer.set_training_data(X_train, y_train_one_hot, X_val, y_val_one_hot)
                optimizer.print_layer_info()
                layer_start_time = time.perf_counter()
                optimizer.step()
                layer_end_time = time.perf_counter()
                swarm_times[fold - 1, layer_idx] = layer_end_time - layer_start_time

                # Guardar las pérdidas de la capa
                train_losses, val_losses = optimizer.get_loss_history()
                fold_losses['train'].append(train_losses)
                fold_losses['val'].append(val_losses)

            training_end_time = time.perf_counter()
            training_elapsed_time = training_end_time - training_start_time
            time_results.append(training_elapsed_time)

            train_pred = model(X_train).argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean().item()
            val_pred = model(X_val).argmax(dim=1)
            val_acc = (val_pred == y_val).float().mean().item()
            y_pred = model(X_test_tensor).argmax(dim=1)
            accuracy = (y_pred == y_test_tensor).float().mean().item()
            logging.info(f'Fold {fold}, Accuracy on training dataset: {train_acc:.4f}')
            logging.info(f'Fold {fold}, Accuracy on validation dataset: {val_acc:.4f}')
            logging.info(f'Fold {fold}, Accuracy on test dataset: {accuracy:.4f}')
            logging.info(f"=========================")

            if accuracy > best_acc:
                best_acc = accuracy
                best_model = model.state_dict()

            train_results.append(train_acc)
            val_results.append(val_acc)
            test_results.append(accuracy)

        all_losses.append(fold_losses)  # Guardar las pérdidas del fold

    # Graficar las pérdidas
    plot_cross_validation_losses(all_losses, dataset_name, config['optimizer'], config['interval_parms_updated'])

    if best_model is not None:
        model.load_state_dict(best_model)
        save_best_model(model, config, best_acc)

        # Calculamos las métricas para el mejor modelo
        logging.info("\n============= BEST MODEL METRICS =============")
        with torch.no_grad():
            # Crear un wrapper para el modelo que implemente predict_proba
            class ModelWrapper:
                def __init__(self, model):
                    self.model = model

                def predict_proba(self, X):
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        outputs = self.model(X_tensor)
                        probas = torch.softmax(outputs, dim=1)
                    return probas.cpu().numpy()

            model_wrapper = ModelWrapper(model)

            # Predicciones con el mejor modelo
            best_train_outputs = model(torch.tensor(X_train_val, dtype=torch.float32).to(device))
            best_train_pred = best_train_outputs.argmax(dim=1)

            best_test_outputs = model(X_test_tensor) # Pasarle tensores
            best_test_pred = best_test_outputs.argmax(dim=1)

            # Calcular métricas detalladas
            best_train_metrics = metrics_calculator.calculate_all_metrics(
                y_train_val,
                best_train_pred.cpu().numpy()
            )

            best_test_metrics = metrics_calculator.calculate_all_metrics(
                y_test_tensor.cpu().numpy(),
                best_test_pred.cpu().numpy()
            )

            # Registrar métricas del mejor modelo
            logging.info("\nBest Model Metrics on Training Set:")
            metrics_calculator.log_metrics(best_train_metrics, f"{dataset_name}_best_train")

            logging.info("\nBest Model Metrics on Test Set:")
            metrics_calculator.log_metrics(best_test_metrics, f"{dataset_name}_best_test")

            # Generar curva ROC para el mejor modelo usando el wrapper
            best_roc_auc = metrics_calculator.plot_multiclass_roc(
                model_wrapper,  # Usar el wrapper en lugar del modelo directamente
                X_test_tensor.cpu().numpy(),
                y_test_tensor.cpu().numpy(),
                n_classes=config['output_dim'],
                dataset_name=f"{dataset_name}_best_model",
                optimizer=config['optimizer']
            )

            logging.info("\nROC-AUC Scores for Best Model:")
            if config['output_dim'] == 2:  # Para clasificación binaria
                logging.info(f"ROC-AUC Score: {best_roc_auc[0]:.4f}")
                logging.info(f"Micro-average: {best_roc_auc['micro']:.4f}")
                logging.info(f"Macro-average: {best_roc_auc['macro']:.4f}")
            else:  # Para clasificación multiclase
                for i in range(config['output_dim']):
                    logging.info(f"Class {i}: {best_roc_auc[i]:.4f}")
                logging.info(f"Micro-average: {best_roc_auc['micro']:.4f}")
                logging.info(f"Macro-average: {best_roc_auc['macro']:.4f}")
            logging.info("=============================================")        

    logging.info("=============================================")
    logging.info("Model Setup:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    logging.info(f"Total Parameters: {total_params.numel()}")
    logging.info("=============================================")

    mean_layer_times = np.mean(swarm_times, axis=0)  # Compute the mean by columns (axis=0)
    std_layer_times = np.std(swarm_times, axis=0)
    for layer_idx in range(len(model.layers)):
        correct_layer_idx = len(model.layers) - layer_idx - 1
        logging.info(f"Mean time per layer {layer_idx}: {mean_layer_times[correct_layer_idx]:.4f} - std: {std_layer_times[correct_layer_idx]:.4f} seconds")

    mean_time = np.mean(time_results)
    std_time = np.std(time_results)
    mean_train = np.mean(train_results)
    std_train = np.std(train_results)
    mean_val = np.mean(val_results)
    std_val = np.std(val_results)
    mean_accuracy = np.mean(test_results)
    std_accuracy = np.std(test_results)
    logging.info(f'Mean time per epoch and folder: {mean_time:.4f} - std: {std_time:.4f} seconds')
    logging.info(f'Mean accuracy on training dataset: {mean_train:.4f} - std: {std_train:.4f}')
    logging.info(f'Mean accuracy on validation dataset: {mean_val:.4f} - std: {std_val:.4f}')
    logging.info(f'Mean accuracy on test dataset: {mean_accuracy:.4f} - std: {std_accuracy:.4f}')
    logging.info("=============================================")


if __name__ == "__main__":
    main()