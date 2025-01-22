import torch
import logging
import numpy as np
import time


from sklearn.model_selection import KFold

# custome data, optimizer and model
from data.benchmarck import load_and_preprocess_data
from metrics.plotting import plot_cross_validation_losses
from custome_optimizer.qpso_optimizerO import QDPSOoOptimizer
from ann.ann_qpsoO import ExtendedModel

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_best_model(model, config, best_acc):
    model_path = f"./models/{config['dataset']}O_best_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_acc': best_acc
    }, model_path)
    logging.info(f"Best model saved at: {model_path}")

def main():
    best_acc = 0
    best_model = None

    # load and preprocess the data accepcting the dataset name: iris, breast_cancer, wine, and circle 
    dataset_name = 'wine'

    # Logging Setup
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Logging Setup
    logging.basicConfig(
        level=logging.INFO,  # Establecer el nivel de registro en INFO
        format='%(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"./output/{dataset_name}O.output"),  # Enviar logs al archivo iris.output
            logging.StreamHandler()  # También enviar logs a la consola
        ]
    )

    X_train_val, X_test, y_train_val, y_test = load_and_preprocess_data(dataset_name)

    input_shape = X_train_val.shape[1]
    output_shape = len(np.unique(y_train_val))
    n_samples = X_train_val.shape[0]

    # Config for the model
    config = {
        'dataset': dataset_name,
        'input_dim': input_shape,
        'output_dim': output_shape,
        'n_samples': n_samples,
        'hidden_layers': [input_shape * 2, (input_shape * 3) // 2, input_shape], 
        'n_particles': 20,
        'g': 1.13,
        'interval_parms_updated': 1,
        'n_folds': 4,
        'n_epochs': 100
    }

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    model = ExtendedModel(num_samples=config['n_samples'], input_dim=config['input_dim'], output_dim=config['output_dim'], hidden_layers=config['hidden_layers']).to(device)
    bounds = [(-1, 1) for _ in range(sum(p.numel() for p in model.parameters()))]
    optimizer = QDPSOoOptimizer(model, bounds, n_particles=config['n_particles'], max_iters=config['n_epochs'], g=config['g'], interval_parms_updated=config['interval_parms_updated'])
    
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

        y_train_one_hot = torch.nn.functional.one_hot(y_train, num_classes=model.layers[-1].out_features).float().to(device)
        y_val_one_hot = torch.nn.functional.one_hot(y_val, num_classes=model.layers[-1].out_features).float().to(device)
        optimizer.set_training_data(X_train, y_train_one_hot, X_val, y_val_one_hot)
        
        with torch.no_grad():
            start_time = time.perf_counter()
            optimizer.step()
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            time_results.append(elapsed_time)

            # Guardar las pérdidas del fold actual
            all_losses.append({
                'train': optimizer.train_losses,
                'val': optimizer.val_losses
            })

            model._set_params(optimizer.best_params)
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

    # Graficar las pérdidas
    plot_cross_validation_losses(all_losses, dataset_name, "qpsoO")

    if best_model is not None:
        model.load_state_dict(best_model)
        save_best_model(model, config, best_acc)

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