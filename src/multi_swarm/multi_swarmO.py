import torch
import time
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, make_circles
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import logging

# custome optimizer and model
from custome_optimizer.qpso_optimizerO import LayerQDPSOoOptimizer
from ann.modelO import ExtendedModel

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data(dataset='iris'):
    """
    Load and preprocess the specified dataset.

    Args:
        dataset (str): Name of the dataset to load. Options: 'iris', 'breast_cancer', 'wine', 'circle'.

    Returns:
        tuple: Preprocessed train-validation and test sets (X_train_val, X_test, y_train_val, y_test).

    Raises:
        ValueError: If an unknown dataset is specified.
    """
    dataset_dict = {
        'iris': load_iris,
        'breast_cancer': load_breast_cancer,
        'wine': load_wine,
    }
    if dataset in dataset_dict:
        data = dataset_dict[dataset]()
    elif dataset == 'circle':
        n = 500
        X, y = make_circles(n_samples=n, factor=0.5, noise=0.05)
        Dataset = namedtuple('Dataset', ['data', 'target', 'feature_names', 'target_names'])
        data = Dataset(
            data=X,
            target=y,
            feature_names=['X coordinate', 'Y coordinate'],
            target_names=['outer circle', 'inner circle']
        )
    else:
        raise ValueError("Unknown dataset")

    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=100)

def save_best_model(model, config, best_acc):
    """
    Save the model state and configuration to a file.

    Args:
        model (OrderedDict): The state dictionary of the model.
        config (dict): The configuration dictionary of the model.
        best_acc (str): The best model accuracy.
    """
    model_path = f"./models/{config['dataset']}_best_model_multi_swarm.pth"
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

    # load and preprocess the data accepcting the dataset name: iris, breast_cancer, wine, and circle 
    dataset_name = 'iris'

    # Logging Setup
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Logging Setup
    logging.basicConfig(
        level=logging.INFO,  # Establecer el nivel de registro en INFO
        format='%(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"./output/{dataset_name}_multi_swarm.output"),  # Enviar logs al archivo iris.output
            logging.StreamHandler()  # También enviar logs a la consola
        ]
    )

    X_train_val, X_test, y_train_val, y_test = load_and_preprocess_data(dataset_name)

    input_shape = X_train_val.shape[1]
    output_shape = len(np.unique(y_train_val))
    n_samples = X_train_val.shape[0]

    config = {
        'dataset': dataset_name,
        'input_dim': input_shape,
        'output_dim': output_shape,
        'n_samples': n_samples,
        'hidden_layers': [input_shape * 2, (input_shape * 3) // 2, input_shape], # (input_shape * 3) // 2, int division
        'n_particles': 20,
        'g': 1.13,
        'interval_parms_updated': 1,
        'n_folds': 4,
        'n_epochs': 100 # max iterations on logs callback function on the training
    }

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    model = ExtendedModel(num_samples=config['n_samples'], input_dim=config['input_dim'], output_dim=config['output_dim'], hidden_layers=config['hidden_layers']).to(device)
    total_params = model.get_flat_params()

    kf = KFold(n_splits=config['n_folds'], shuffle=True, random_state=100)

    swarms = []
    logging.info(f"=============================================")
    logging.info(f"Parameter numbers to training {total_params.numel()}")
    for i in range(len(model.layers)):
        swarm = LayerQDPSOoOptimizer(model, i, total_params, n_particles=config['n_particles'], max_iters=config['n_epochs'], g=config['g'], interval_parms_updated=config['interval_parms_updated'])
        swarms.append(swarm)
        logging.info(f"Layer {i} swarm initialized with {swarm.dim} dimensions for each particle.")
    logging.info(f"=============================================")
    
    train_results = []
    val_results = []
    test_results = []

    swarm_times = np.zeros((config['n_folds'], len(model.layers)))  # Matriz
    time_results = []

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

        with torch.no_grad():  # Disable gradient tracking
            #logging.info(f"Starting epoch {epoch + 1} for fold {fold}")
            training_start_time = time.perf_counter()
            for layer_idx, optimizer in enumerate(reversed(swarms)):
                optimizer.set_training_data(X_train, y_train_one_hot, X_val, y_val_one_hot)
                optimizer.print_layer_info() 
                layer_start_time = time.perf_counter()   
                optimizer.step()
                layer_end_time = time.perf_counter()
                swarm_times[fold - 1, layer_idx] = layer_end_time - layer_start_time # start matrix idx in 0 and save the time in order, so then u need to ajusted in the print info 
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

    if best_model is not None:
            model.load_state_dict(best_model)
            save_best_model(model, config, best_acc)

    logging.info("=============================================")
    logging.info("Model Setup:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    logging.info(f"Total Parameters: {total_params.numel()}")
    logging.info("=============================================")
    
    mean_layer_times = np.mean(swarm_times, axis=0) # Compute the meand by columns (axis=0)
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