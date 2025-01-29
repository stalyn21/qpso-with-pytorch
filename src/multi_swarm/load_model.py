import torch
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, make_circles
from collections import namedtuple
from multi_swarm.ann.model_qpsoO import ExtendedModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

def load_model(file_path):
    """
    Load a saved model and its configuration from a file.

    Args:
        file_path (str): Path to the saved model file.

    Returns:
        tuple: Loaded model and its configuration.
    """
    save_dict = torch.load(file_path, weights_only=True)
    config = save_dict['config']
    model = ExtendedModel(num_samples=config['n_samples'], 
                          input_dim=config['input_dim'], 
                          output_dim=config['output_dim'], 
                          hidden_layers=config['hidden_layers']).to(device)
    model.load_state_dict(save_dict['model_state_dict'])
    return model, config

def replicate_accuracy():
    """
    Load a saved model, make predictions on the test set, and calculate the accuracy.

    Returns:
        float: The accuracy of the model on the test set.
    """
    # Cargar el modelo y la configuración
    model, config = load_model('best_model.pth')
    model.eval()  # Poner el modelo en modo de evaluación

    # Cargar y preprocesar los datos
    dataset_name = config['dataset']
    X_train_val, X_test, y_train_val, y_test = load_and_preprocess_data(dataset_name)

    # Convertir los datos de prueba a tensores de PyTorch
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # Hacer predicciones
    with torch.no_grad():
        y_pred = model(X_test_tensor).argmax(dim=1)
        accuracy = (y_pred == y_test_tensor).float().mean().item()

    print(f'Replicated accuracy on test dataset: {accuracy:.4f}')

    return accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replicated_accuracy = replicate_accuracy()