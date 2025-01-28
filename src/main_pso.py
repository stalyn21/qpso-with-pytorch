import torch
import logging
import numpy as np
import time

from sklearn.model_selection import KFold

from data.benchmarck import load_and_preprocess_data
from metrics.plotting import plot_cross_validation_losses
from metrics.metrics import MulticlassMetrics
from one_swarm.custome_optimizer.pso_optimizer import PSOOptimizer
from one_swarm.ann.ann_pso import ExtendedModelPSO

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
def save_best_model(model, config, best_acc):
    model_path = f"./models/{config['optimizer']}/{config['dataset']}_{config['optimizer']}_best_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_acc': best_acc
    }, model_path)
    logging.info(f"Best model saved at: {model_path}")

def main():
    best_acc = 0
    best_model = None
    metrics_calculator = MulticlassMetrics()
    
    dataset_name = 'breast_cancer'

    X_train_val, X_test, y_train_val, y_test = load_and_preprocess_data(dataset_name)

    input_shape = X_train_val.shape[1]
    output_shape = len(np.unique(y_train_val))
    n_samples = X_train_val.shape[0]

    config = {
        'optimizer': 'PSO_bound', # options: PSO_bound, PSO
        'dataset': dataset_name,
        'input_dim': input_shape,
        'output_dim': output_shape,
        'n_samples': n_samples,
        'hidden_layers': [input_shape*3, input_shape*2, input_shape], # [input_shape * 2, (input_shape * 3) // 2, input_shape],
        'n_particles': 20,
        'n_folds': 4,
        'n_epochs': 100
    }

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

    kf = KFold(n_splits=config['n_folds'], shuffle=True, random_state=100)

    train_results = []
    val_results = []
    test_results = []
    time_results = []

    all_losses = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_val), 1):
        logging.info(f"======= Fold {fold} =======")

        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.long).to(device)

        model = ExtendedModelPSO(
            num_samples=config['n_samples'],
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            hidden_layers=config['hidden_layers']
        ).to(device)
        
        # Comment without Parameters bounds 
        min_param_value=-1 
        max_param_value=1 

        optimizer = PSOOptimizer(
            model=model,
            n_particles=config['n_particles'],
            max_iters=config['n_epochs'],
            # Comment witout bound
            min_param_value=min_param_value,
            max_param_value=max_param_value
        )

        start_time = time.perf_counter()
        optimizer.step(X_train, y_train, X_val, y_val)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        time_results.append(elapsed_time)

        # Guardar las pérdidas del fold actual
        all_losses.append({
            'train': optimizer.train_losses,
            'val': optimizer.val_losses
        })

        with torch.no_grad():
            train_pred = model(X_train).argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean().item()

            val_pred = model(X_val).argmax(dim=1)
            val_acc = (val_pred == y_val).float().mean().item()

            test_pred = model(X_test_tensor).argmax(dim=1)
            test_acc = (test_pred == y_test_tensor).float().mean().item()

            logging.info(f'Fold {fold}, Training accuracy: {train_acc:.4f}')
            logging.info(f'Fold {fold}, Validation accuracy: {val_acc:.4f}')
            logging.info(f'Fold {fold}, Test accuracy: {test_acc:.4f}')
            logging.info("=========================")

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = model.state_dict()

            train_results.append(train_acc)
            val_results.append(val_acc)
            test_results.append(test_acc)
    
    # Graficar las pérdidas
    plot_cross_validation_losses(all_losses, dataset_name, config['optimizer'])

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

            best_test_outputs = model(X_test_tensor)
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

    # Imprimir resultados finales
    logging.info("=============================================")
    logging.info("Model Setup:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    logging.info(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info("=============================================")

    mean_time = np.mean(time_results)
    std_time = np.std(time_results)
    mean_train = np.mean(train_results)
    std_train = np.std(train_results)
    mean_val = np.mean(val_results)
    std_val = np.std(val_results)
    mean_accuracy = np.mean(test_results)
    std_accuracy = np.std(test_results)

    logging.info(f'Mean time per epoch and folder: {mean_time:.4f} - std:{std_time:.4f} seconds')
    logging.info(f'Mean accuracy on training dataset: {mean_train:.4f} - std: {std_train:.4f}')
    logging.info(f'Mean accuracy on validation dataset: {mean_val:.4f} - std: {std_val:.4f}')
    logging.info(f'Mean accuracy on test dataset: {mean_accuracy:.4f} - std: {std_accuracy:.4f}')
    logging.info("=============================================")

    # Liberar memoria CUDA si se está usando
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()