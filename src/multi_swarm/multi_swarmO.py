import torch
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, make_circles
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import logging

# custome optimizer and model
from custome_optimizer.qpso_optimizerO import LayerQDPSOoOptimizer
from ann.modelO import ExtendedModel

# Configuración de logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

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

def save_model(model_state, config, file_path):
    """
    Save the model state and configuration to a file.

    Args:
        model_state (OrderedDict): The state dictionary of the model.
        config (dict): The configuration dictionary of the model.
        file_path (str): The path where the model will be saved.
    """
    save_dict = {
        'model_state_dict': model_state,
        'config': config
    }
    torch.save(save_dict, file_path)
    logging.info(f"Model saved to {file_path}")

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
    dataset_name = 'iris'
    X_train_val, X_test, y_train_val, y_test = load_and_preprocess_data(dataset_name)
    input_shape = X_train_val.shape[1]
    output_shape = len(np.unique(y_train_val))
    n_samples = X_train_val.shape[0]

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    config = {
        'dataset': dataset_name,
        'input_dim': input_shape,
        'output_dim': output_shape,
        'n_samples': n_samples,
        'hidden_layers': [5, 6, 5, 4],
        'n_particles': 20,
        'max_iters': 60,
        'g': 1.15,
        'interval_parms_updated': 10,
        'n_folds': 4,
        'n_epochs': 4,
    }

    kf = KFold(n_splits=config['n_folds'], shuffle=True, random_state=100)

    model = ExtendedModel(num_samples=config['n_samples'], input_dim=config['input_dim'], output_dim=config['output_dim'], hidden_layers=config['hidden_layers']).to(device)
    total_params = model.get_flat_params()

    swarms = []
    logging.info(f"=============================================")
    logging.info(f"Parameter numbers to training {total_params.numel()}")
    for i in range(len(model.layers)):
        swarm = LayerQDPSOoOptimizer(model, i, total_params, n_particles=config['n_particles'], max_iters=config['max_iters'], g=config['g'], interval_parms_updated=config['interval_parms_updated'])
        swarms.append(swarm)
        logging.info(f"Layer {i} swarm initialized with {swarm.dim} dimensions for each particle.")
    logging.info(f"=============================================")

    test_results = []

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
            
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(config['n_epochs']):
            logging.info(f"Starting epoch {epoch + 1} for fold {fold}")
            for optimizer in reversed(swarms):
                optimizer.set_training_data(X_train, y_train_one_hot)
                optimizer.print_layer_info()
                optimizer.step()

            output = model(X_train)
            loss = model.lf.cross_entropy(y_train_one_hot, output)
            train_losses.append(loss.item())

            with torch.no_grad():
                val_output = model(X_val)
                val_loss = model.lf.cross_entropy(y_val_one_hot, val_output)
                val_losses.append(val_loss.item())

            logging.info(f'Fold {fold}, Epoch {epoch + 1}, Loss: {loss.item():.4f} - Validation Loss: {val_loss.item():.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

        model.load_state_dict(best_model_state)
        save_model(best_model_state, config, 'best_model.pth')
        logging.info(f'Final Validation Loss for fold {fold}: {best_val_loss:.4f}')

        y_pred = model(X_test_tensor).argmax(dim=1)
        accuracy = (y_pred == y_test_tensor).float().mean().item()
        logging.info(f'Fold {fold}, Accuracy on test dataset: {accuracy:.4f}')
        logging.info(f"=========================")

        test_results.append(accuracy)

    logging.info("=============================================")
    logging.info("Model Setup:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    logging.info(f"Total Parameters: {total_params.numel()}")
    logging.info("=============================================")

    mean_accuracy = np.mean(test_results)
    std_accuracy = np.std(test_results)
    logging.info(f'Mean accuracy on test dataset: {mean_accuracy:.4f}')
    logging.info(f'Standard deviation of accuracy on test dataset: {std_accuracy:.4f}')
    logging.info("=============================================")

if __name__ == "__main__":
    main()