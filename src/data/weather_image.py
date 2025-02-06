import cv2
import numpy as np
import mahotas
from pathlib import Path
import logging
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.manifold import Isomap, MDS
from sklearn.decomposition import PCA

class WeatherImageDataset:
    def __init__(self, root_path='./data/img/mcw', img_size=(150, 150), bins=4):
        self.root_path = Path(root_path)
        if not self.root_path.exists():
            raise ValueError(f"El directorio {root_path} no existe")

        self.img_size = img_size
        self.bins = bins
        self.classes = ['cloudy', 'rain', 'shine', 'sunrise']
        self.feature_names = ['histogram', 'haralick', 'hu_moments']

        # Verificar que existan los subdirectorios de las clases
        self.verify_directory_structure()

    def verify_directory_structure(self):
        """Verifica que exista la estructura de directorios correcta."""
        missing_dirs = []
        for class_name in self.classes:
            class_path = self.root_path / class_name
            if not class_path.exists():
                missing_dirs.append(class_name)

        if missing_dirs:
            raise ValueError(
                f"Los siguientes directorios de clase no existen: {missing_dirs}\n"
                f"La estructura esperada es:\n"
                f"{self.root_path}/\n" +
                "\n".join(f"├── {class_name}/" for class_name in self.classes)
            )

    def _fd_hu_moments(self, image):
        """Extrae características de momentos Hu."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

    def _fd_haralick(self, image):
        """Extrae características de textura Haralick."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        return haralick

    def _fd_histogram(self, image):
        """Extrae características del histograma de color."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image], [0, 1, 2], None,
                           [self.bins, self.bins, self.bins],
                           [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _extract_features(self, image):
        """Extrae y combina todas las características de una imagen."""
        image = cv2.resize(image, self.img_size)

        fv_histogram = self._fd_histogram(image)
        fv_haralick = self._fd_haralick(image)
        fv_hu_moments = self._fd_hu_moments(image)

        return np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    def load_and_preprocess_data(self, test_size=0.2, random_state=100):
        """Carga y preprocesa el dataset completo."""
        features = []
        labels = []
        processed_count = 0

        # Cargar imágenes y extraer características
        for class_idx, class_name in enumerate(self.classes):
            class_path = self.root_path / class_name
            class_images = list(class_path.glob('*'))

            if not class_images:
                print(f"Advertencia: No se encontraron imágenes en {class_path}")
                continue

            print(f"\nProcesando clase: {class_name}")
            print(f"Encontradas {len(class_images)} imágenes")

            for img_path in class_images:
                try:
                    # Cargar y procesar imagen
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"No se pudo cargar la imagen: {img_path}")
                        continue

                    # Extraer características
                    global_feature = self._extract_features(img)
                    features.append(global_feature)
                    labels.append(class_idx)
                    processed_count += 1

                    if processed_count % 10 == 0:  # Mostrar progreso cada 10 imágenes
                        print(f'Procesadas {processed_count} imágenes...')

                except Exception as e:
                    print(f'Error procesando {img_path}: {str(e)}')

        if not features:
            raise ValueError("No se pudieron procesar imágenes. Verifica que el directorio contenga imágenes válidas.")

        # Convertir a arrays numpy
        X = np.array(features)
        y = np.array(labels)

        print(f"\nTotal de imágenes procesadas: {len(features)}")
        print(f"Forma del array de características: {X.shape}")

        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Crear namedtuple para almacenar información del dataset
        DatasetInfo = namedtuple('DatasetInfo',
                            ['X_train', 'X_test', 'y_train', 'y_test',
                                'feature_names', 'target_names'])

        return DatasetInfo(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=self.feature_names,
            target_names=self.classes
        )

from sklearn.manifold import Isomap, MDS
from sklearn.decomposition import PCA

def load_and_preprocess_mcw(root_path='./data/img/mcw', test_size=0.2, random_state=100,
                           reduction_method=None, n_components=None):
    """
    Función auxiliar para cargar el dataset directamente.

    Args:
        root_path (str): Ruta al directorio de imágenes
        test_size (float): Proporción del conjunto de prueba
        random_state (int): Semilla aleatoria
        reduction_method (str): Método de reducción de características ('isomap', 'mds', 'pca')
        n_components (int): Número de componentes para la reducción (por defecto: n_features/6)
    """
    print("loading MCW dataset ....")
    try:
        dataset = WeatherImageDataset(root_path=root_path)
        data = dataset.load_and_preprocess_data(test_size, random_state)

        # Obtener las dimensiones de cada tipo de característica
        hist_size = dataset.bins ** 3  # 64 para bins=4
        haralick_size = 13
        hu_moments_size = 7

        # Separar las características
        X_train = data.X_train
        X_test = data.X_test

        # Normalizar cada tipo de característica por separado
        scaler_hist = StandardScaler()
        scaler_haralick = StandardScaler()
        scaler_hu = StandardScaler()

        # Training data
        X_train_hist = scaler_hist.fit_transform(X_train[:, :hist_size])
        X_train_haralick = scaler_haralick.fit_transform(X_train[:, hist_size:hist_size+haralick_size])
        X_train_hu = scaler_hu.fit_transform(X_train[:, -hu_moments_size:])

        # Test data
        X_test_hist = scaler_hist.transform(X_test[:, :hist_size])
        X_test_haralick = scaler_haralick.transform(X_test[:, hist_size:hist_size+haralick_size])
        X_test_hu = scaler_hu.transform(X_test[:, -hu_moments_size:])

        # Recombinar las características normalizadas
        X_train_normalized = np.hstack([X_train_hist, X_train_haralick, X_train_hu])
        X_test_normalized = np.hstack([X_test_hist, X_test_haralick, X_test_hu])

        # Reducción de características solo si se especifica un método
        if reduction_method is not None:
            if n_components is None:
                n_components = X_train_normalized.shape[1] // 6

            print("\nAplicando reducción de características...")
            print(f"Método: {reduction_method}")
            print(f"Componentes objetivo: {n_components}")

            # Seleccionar el método de reducción
            if reduction_method.lower() == 'isomap':
                reducer = Isomap(n_components=n_components, n_neighbors=min(100, X_train_normalized.shape[0]-1))
                X_train_final = reducer.fit_transform(X_train_normalized)
                X_test_final = reducer.transform(X_test_normalized)
            elif reduction_method.lower() == 'mds':
                # Para MDS, necesitamos aplicar fit_transform a ambos conjuntos
                reducer_train = MDS(n_components=n_components, random_state=random_state)
                reducer_test = MDS(n_components=n_components, random_state=random_state)
                X_train_final = reducer_train.fit_transform(X_train_normalized)
                X_test_final = reducer_test.fit_transform(X_test_normalized)
            elif reduction_method.lower() == 'pca':
                reducer = PCA(n_components=n_components)
                X_train_final = reducer.fit_transform(X_train_normalized)
                X_test_final = reducer.transform(X_test_normalized)
            else:
                raise ValueError("Método de reducción no válido. Use 'isomap', 'mds' o 'pca'")

            print("\nDataset Information:")
            print(f"Original training set shape: {X_train_normalized.shape}")
            print(f"Reduced training set shape: {X_train_final.shape}")
            print(f"Original testing set shape: {X_test_normalized.shape}")
            print(f"Reduced testing set shape: {X_test_final.shape}")
        
        else:
            X_train_final = X_train_normalized
            X_test_final = X_test_normalized

        print(f"Training class distribution: {np.unique(data.y_train, return_counts=True)}")
        print(f"Testing class distribution: {np.unique(data.y_test, return_counts=True)}")

        return X_train_final, X_test_final, data.y_train, data.y_test

    except Exception as e:
        logging.error(f"\nError al cargar el dataset: {str(e)}")
        raise e