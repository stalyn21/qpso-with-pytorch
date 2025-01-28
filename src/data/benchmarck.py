from sklearn.datasets import load_iris, load_breast_cancer, load_wine, make_circles
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(dataset='iris'):
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