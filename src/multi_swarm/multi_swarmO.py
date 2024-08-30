import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import logging

# custome optimizer and model
from custome_optimizer.qpso_optimizerO import LayerQDPSOoOptimizer
from ann.annO import ExtendedModel

# Configuración de logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Aseguramos que PyTorch use GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fuvtion to load and preprocess the datasets
def load_and_preprocess_data(dataset='iris'):
    match dataset:
        case "iris":
            from sklearn.datasets import load_iris
            data = load_iris()
        case "breast_cancer":
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
        case "wine":
            from sklearn.datasets import load_wine
            data = load_wine()
        case "circle":
            from collections import namedtuple
            from sklearn.datasets import make_circles
            n = 500 # register number
            # generating the input X and output y (binary vector)
            X, y = make_circles(n_samples=n, factor=0.5, noise=0.05)
            # Create a namedtuple to mimic the structure of sklearn datasets
            Dataset = namedtuple('Dataset', ['data', 'target', 'feature_names', 'target_names'])            
            # Create the data object
            data = Dataset(
                data=X,
                target=y, 
                feature_names=['X coordinate', 'Y coordinate'],
                target_names=['outer circle', 'inner circle']
            )
        case _:
            raise ValueError("Unknown dataset")

    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=100)

def main():
    # load and preprocess the data accepcting the dataset name: iris, breast_cancer, wine, and circle 
    dataset_name = 'iris'
    X_train_val, X_test, y_train_val, y_test = load_and_preprocess_data(dataset_name)

    input_shape = X_train_val.shape[1]
    output_shape = len(set(y_train_val))
    n_samples = X_train_val.shape[0]

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # Model setup
    config = {
        'dataset': dataset_name,
        'input_dim': input_shape,
        'output_dim': output_shape,
        'n_samples': n_samples,
        'hidden_layers': [5, 6, 5, 4], 
        'n_particles': 20,
        'max_iters': 60,
        'g': 1.13,
        'interval_parms_updated': 10,
        'n_folds': 4,
        'n_epochs': 4,
    }

    kf = KFold(n_splits=config['n_folds'], shuffle=True, random_state=100)

    # Initialize the model
    model = ExtendedModel(num_samples=config['n_samples'], input_dim=config['input_dim'], output_dim=config['output_dim'], hidden_layers=config['hidden_layers']).to(device)
    total_params = model.get_flat_params()

    test_results = []

    # K-Fold Cross-Validation
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

        # Initialize the swarms for each layer
        swarms = []
        for i in range(len(model.layers)):
            swarm = LayerQDPSOoOptimizer(model, i, total_params, n_particles=config['n_particles'], max_iters=config['max_iters'], g=config['g'], interval_parms_updated=config['interval_parms_updated'])
            swarm.set_training_data(X_train, y_train_one_hot)
            swarms.append(swarm)
            logging.info(f"Layer {i} swarm initialized with {swarm.dim} dimensions for each particle.")

        # Training loop for each epoch and swarm
        for epoch in range(config['n_epochs']):
            logging.info(f"Starting epoch {epoch + 1} for fold {fold}")
            for optimizer in reversed(swarms):
                #logging.info(f"Optimizer initialized for training layer {optimizer.layer_idx}.")
                optimizer.print_layer_info()
                optimizer.step()

            output = model(X_train)
            loss = model.lf.cross_entropy(y_train_one_hot, output)

            # Validation loss calculation for each epoch
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = model.lf.cross_entropy(y_val_one_hot, val_output)

            logging.info(f'Fold {fold}, Epoch {epoch + 1}, Loss: {loss.item()} - Validation Loss: {val_loss.item()}')

        # Final validation loss calculation for each fold and epoch
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = model.lf.cross_entropy(y_val_one_hot, val_output)
            logging.info(f'Final Validation Loss for fold {fold}: {val_loss.item()}')

        # Final evaluation on the test dataset for each fold and epoch
        y_pred = model(X_test_tensor).argmax(dim=1)
        accuracy = (y_pred == y_test_tensor).float().mean().item()
        logging.info(f'Fold {fold}, Accuracy on iris test dataset: {accuracy:.4f}')
        logging.info(f"=========================")

        test_results.append(accuracy)

    # Print the model setup configuration and final results on the test dataset for all folds and epochs
    logging.info("=============================================")
    logging.info("Model Setup:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    logging.info("=============================================")
    mean_accuracy = torch.tensor(test_results).mean().item()
    std_accuracy = torch.tensor(test_results).std().item()
    logging.info(f'Mean accuracy on test dataset: {mean_accuracy:.4f}')
    logging.info(f'Standard deviation of accuracy on test dataset: {std_accuracy:.4f}')
    logging.info("=============================================")

if __name__ == "__main__":
    main()