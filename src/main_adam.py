import torch
import logging
import numpy as np
import time

from data.benchmarck import load_and_preprocess_data
from metrics.plotting import plot_cross_validation_losses
from metrics.metrics import MulticlassMetrics
from sklearn.model_selection import KFold
from adam.ann.model import ExtendedModel

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

    dataset_name = "breast_cancer"

    X_trainval, X_test, y_trainval, y_test = load_and_preprocess_data(dataset_name)

    input_shape = X_trainval.shape[1]
    output_shape = len(np.unique(y_trainval))
    n_samples = X_trainval.shape[0]

    config = {
        "optimizer": "Adam",
        "dataset": dataset_name,
        "inputdim": input_shape,
        "outputdim": output_shape,
        "nsamples": n_samples,
        "hiddenlayers": [input_shape * 2, (input_shape * 3) // 2, input_shape],
        "lr": 0.001,
        "n_epochs": 100,
        "n_folds": 4,
    }

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"./output/{config['optimizer']}/{dataset_name}_{config['optimizer']}.output"), 
            logging.StreamHandler()
        ]
    )

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    kf = KFold(n_splits=config["n_folds"], shuffle=True, random_state=100)
    train_results = []
    val_results = []
    test_results = []
    time_results = []

    all_losses = []  # Para almacenar las pérdidas de cada fold

    for fold, (train_index, val_index) in enumerate(kf.split(X_trainval)):
        logging.info(f"======= Fold {fold+1} =======")

        X_train, X_val = X_trainval[train_index], X_trainval[val_index]
        y_train, y_val = y_trainval[train_index], y_trainval[val_index]

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.long).to(device)

        model = ExtendedModel(X_train, config["inputdim"], config["outputdim"], config["hiddenlayers"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        y_train_onehot = torch.nn.functional.one_hot(y_train, num_classes=config["outputdim"]).float().to(device)
        y_val_onehot = torch.nn.functional.one_hot(y_val, num_classes=config["outputdim"]).float().to(device)

        best_acc_fold = float('inf')
        start_time = time.perf_counter()

        # Para almacenar las pérdidas del fold actual
        fold_train_losses = []
        fold_val_losses = []

        for epoch in range(config["n_epochs"]):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = model.lf.crossentropy(y_train_onehot, output)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                val_output = model(X_val)
                val_loss = model.lf.crossentropy(y_val_onehot, val_output)

            # Almacenar las pérdidas
            fold_train_losses.append(loss.item())
            fold_val_losses.append(val_loss.item())

            if val_loss.item() < best_acc_fold:
                best_acc_fold = val_loss.item()

            logging.info(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        time_results.append(elapsed_time)

        # Al final de cada fold, almacenar las pérdidas
        all_losses.append({
            'train': fold_train_losses,
            'val': fold_val_losses
        })

        model.eval()
        with torch.no_grad():
            train_pred = model(X_train).argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean().item()
            val_pred = model(X_val).argmax(dim=1)
            val_acc = (val_pred == y_val).float().mean().item()
            test_pred = model(X_test).argmax(dim=1)
            test_acc = (test_pred == y_test).float().mean().item()

        logging.info(f"Fold {fold+1}, Training accuracy: {train_acc:.4f}")
        logging.info(f"Fold {fold+1}, Validation accuracy: {val_acc:.4f}")
        logging.info(f"Fold {fold+1}, Test accuracy: {test_acc:.4f}")
        logging.info(f"=========================")

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
            best_train_outputs = model(torch.tensor(X_trainval, dtype=torch.float32).to(device))
            best_train_pred = best_train_outputs.argmax(dim=1)

            best_test_outputs = model(X_test)
            best_test_pred = best_test_outputs.argmax(dim=1)

            # Calcular métricas detalladas
            best_train_metrics = metrics_calculator.calculate_all_metrics(
                y_trainval,
                best_train_pred.cpu().numpy()
            )

            best_test_metrics = metrics_calculator.calculate_all_metrics(
                y_test.cpu().numpy(),
                best_test_pred.cpu().numpy()
            )

            # Registrar métricas del mejor modelo
            logging.info("\nBest Model Metrics on Training Set:")
            metrics_calculator.log_metrics(best_train_metrics, f"{dataset_name}_best_train")

            logging.info("\nBest Model Metrics on Test Set:")
            metrics_calculator.log_metrics(best_test_metrics, f"{dataset_name}_best_test")

            # Generar curva ROC para el mejor modelo
            best_roc_auc = metrics_calculator.plot_multiclass_roc(
                model_wrapper,
                X_test.cpu().numpy(),
                y_test.cpu().numpy(),
                n_classes=config['outputdim'],
                dataset_name=f"{dataset_name}_best_model",
                optimizer=config['optimizer']
            )

            logging.info("\nROC-AUC Scores for Best Model:")
            if config['outputdim'] == 2:
                logging.info(f"ROC-AUC Score: {best_roc_auc[0]:.4f}")
                logging.info(f"Micro-average: {best_roc_auc['micro']:.4f}")
                logging.info(f"Macro-average: {best_roc_auc['macro']:.4f}")
            else:
                for i in range(config['outputdim']):
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()