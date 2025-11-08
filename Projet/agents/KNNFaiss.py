import numpy as np
import faiss
from scipy import stats
import time
from sklearn.preprocessing import normalize

def create_super_features(X_flat: np.ndarray) -> np.ndarray:
    """Crée les 5 'super-features' globales."""
    mean = np.mean(X_flat, axis=1)
    std = np.std(X_flat, axis=1)
    median = np.median(X_flat, axis=1)
    count_zero = np.count_nonzero(X_flat == 0, axis=1)
    count_max = np.count_nonzero(X_flat == 255, axis=1)
    return np.vstack((mean, std, median, count_zero, count_max)).T

class Agent:
    """
    Agent K-NN (Faiss) entraîné UNIQUEMENT sur les 5 "Super-Features".
    Hypothèse : Les 784 pixels sont du bruit.
    """
    def __init__(self, output_dim: int = 10, seed: int = None):
        self.k = 15 # On a peu de features, on peut prendre plus de voisins
        self.index = None
        self.y_train_labels = None
        faiss.omp_set_num_threads(2) # 2 CPUs

    def reset(self):
        self.index = None
        self.y_train_labels = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Entraîne le K-NN sur les 5 features.
        """
        # 1. Aplatir (juste pour le calcul des features)
        X_flat = X_train.reshape(X_train.shape[0], -1)
        y_flat = y_train.ravel()

        # 2. CRÉER LES SUPER-FEATURES (Ton idée)
        X_features = create_super_features(X_flat).astype('float32')
        
        # 3. Normaliser les 5 features (important pour Faiss)
        normalize(X_features, norm='l2', axis=1, copy=False)
        
        self.y_train_labels = y_flat
        d = X_features.shape[1] # Dimensions = 5

        # 4. Créer l'index Faiss (HNSW rapide)
        self.index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_L2)
        self.index.add(X_features)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Prédit avec le K-NN sur 5 features.
        """
        if self.index is None:
            raise RuntimeError("L'agent doit être entraîné.")
            
        # 1. Aplatir (pour le calcul des features)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # 2. CRÉER LES SUPER-FEATURES pour le test
        X_test_features = create_super_features(X_test_flat).astype('float32')
        
        # 3. Normaliser les 5 features
        normalize(X_test_features, norm='l2', axis=1, copy=False)
        
        # 4. Prédire (ultra-rapide)
        distances, indices = self.index.search(X_test_features, self.k)
        neighbor_labels = self.y_train_labels[indices]
        predictions, _ = stats.mode(neighbor_labels, axis=1)
        
        return predictions.ravel()