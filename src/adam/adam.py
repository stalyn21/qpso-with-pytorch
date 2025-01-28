import torch
import logging
import numpy as np
import time
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, make_circles
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

from ann.model import ExtendedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data(dataset="iris"):
    dataset_dict = {
        "iris": load_iris,
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
    }
    
    if dataset in dataset_dict:
        data = dataset_dict[dataset]()
    elif dataset == "circle":
        n = 500
        X, y = make_circles(n_samples=n, factor=0.5, noise=0.05)
        Dataset = namedtuple('Dataset', ['data', 'target', 'feature_names', 'target_names'])
        data = Dataset(data=X, target=y, feature_names=['X coordinate', 'Y coordinate'], target_names=['outer circle', 'inner circle'])
    else:
        raise ValueError("Unknown dataset")
    
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.2, random_state=100)

def save_best_model(model, config, best_val_acc):
    model_path = f"./models/{config['dataset']}_best_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_acc': best_val_acc
    }, model_path)
    logging.info(f"Best model saved at: {model_path}")

def main():
    best_acc = 0
    best_model = None

    dataset_name = "breast_cancer"

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"./output/{dataset_name}_adam.output"),
            logging.StreamHandler()
        ]
    )
    
    X_trainval, X_test, y_trainval, y_test = load_and_preprocess_data(dataset_name)
    
    input_shape = X_trainval.shape[1]
    output_shape = len(np.unique(y_trainval))
    n_samples = X_trainval.shape[0]
    
    config = {
        "dataset": dataset_name,
        "inputdim": input_shape,
        "outputdim": output_shape,
        "nsamples": n_samples,
        "hiddenlayers": [input_shape * 2, (input_shape * 3) // 2, input_shape],  # 3 times the input dimension, also accepted [4, 6, 4]
        "lr": 0.001,
        "n_epochs": 100,
        "n_folds": 4,
    }
    
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    
    kf = KFold(n_splits=config["n_folds"], shuffle=True, random_state=100)
    train_results = []
    val_results = []
    test_results = []
    time_results = []
    
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
            
            if val_loss.item() < best_acc_fold:
                best_acc_fold = val_loss.item()
            
            logging.info(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f} - Best Val Loss: {best_acc_fold:.4f}")
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        time_results.append(elapsed_time)
        
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train).argmax(dim=1)
            train_acc = (train_pred == y_train.float()).float().mean().item()
            val_pred = model(X_val).argmax(dim=1)
            val_acc = (val_pred == y_val.float()).float().mean().item()
            y_pred = model(X_test).argmax(dim=1)
            accuracy = (y_pred == y_test.float()).float().mean().item()
        
        logging.info(f"Fold {fold+1}, Accuracy on training dataset: {train_acc:.4f}")
        logging.info(f"Fold {fold+1}, Accuracy on validation dataset: {val_acc:.4f}")
        logging.info(f"Fold {fold+1}, Accuracy on test dataset: {accuracy:.4f}")
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

if __name__ == "__main__":
    main()