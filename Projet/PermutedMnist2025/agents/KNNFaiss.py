import numpy as np
import faiss
from scipy import stats
from sklearn.preprocessing import normalize
import os
import joblib

# Importation depuis le fichier utils
from .utils import create_super_features_v5 as create_super_features

class Agent:
    """
    Agent K-NN (Faiss) entraîné UNIQUEMENT sur les 5 "Super-Features".
    """
    def __init__(self, output_dim: int = 10, seed: int = None):
        self.k = 15
        self.index = None
        self.y_train_labels = None
        faiss.omp_set_num_threads(2)

    def reset(self):
        self.index = None
        self.y_train_labels = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Entraîne le K-NN sur les 5 features.
        """
        self.reset()
        
        X_flat = X_train.reshape(X_train.shape[0], -1)
        y_flat = y_train.ravel()

        X_features = create_super_features(X_flat).astype('float32')
        normalize(X_features, norm='l2', axis=1, copy=False)
        
        self.y_train_labels = y_flat
        d = X_features.shape[1]

        self.index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_L2)
        self.index.add(X_features)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Prédit avec le K-NN sur 5 features.
        """
        if self.index is None or self.y_train_labels is None:
            raise RuntimeError("L'agent doit être entraîné ou chargé avant prédiction.")
            
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        X_test_features = create_super_features(X_test_flat).astype('float32')
        normalize(X_test_features, norm='l2', axis=1, copy=False)
        
        distances, indices = self.index.search(X_test_features, self.k)
        neighbor_labels = self.y_train_labels[indices]
        predictions, _ = stats.mode(neighbor_labels, axis=1)
        
        return predictions.ravel()

    def save(self, path: str = "artifacts/KNNFaiss"):
        """Sauvegarde l'index Faiss et les labels."""
        try:
            os.makedirs(path, exist_ok=True)
            # Faiss index ne peut pas être "picklé", on doit utiliser sa propre méthode
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
            # On sauvegarde les labels avec joblib
            joblib.dump(self.y_train_labels, os.path.join(path, "labels.pkl"))
            print(f"Agent (Faiss) sauvegardé dans {path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'agent : {e}")

    def load(self, path: str = "artifacts/KNNFaiss"):
        """Charge un index Faiss et les labels."""
        try:
            index_path = os.path.join(path, "index.faiss")
            labels_path = os.path.join(path, "labels.pkl")

            if not (os.path.exists(index_path) and os.path.exists(labels_path)):
                raise FileNotFoundError(f"Fichiers non trouvés dans {path}")
            
            self.index = faiss.read_index(index_path)
            self.y_train_labels = joblib.load(labels_path)
            print(f"Agent (Faiss) chargé depuis {path}")
        except Exception as e:
            print(f"Erreur lors du chargement de l'agent : {e}")