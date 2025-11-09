import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from scipy import stats # Nécessaire pour les utils
import os
import joblib

# Importation depuis le fichier utils
from .utils import create_super_features_v9 as create_super_features

class Agent:
    """
    Agent ExtraTrees-Seul + 9 "Super-Features".
    """
    def __init__(self, output_dim: int = 10, seed: int = None):
        self.seed = seed
        self.model = None
        self.n_jobs = 2 # 2 CPUs
        
        self.params = {
            'n_estimators': 150,
            'max_depth': 25,
            'n_jobs': self.n_jobs,
            'random_state': self.seed,
        }

    def reset(self):
        """Reset l'agent pour une nouvelle tâche."""
        self.model = ExtraTreesClassifier(**self.params)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Entraîne l'ET sur les 784 pixels + 9 features.
        """
        self.reset()
        
        X_flat = X_train.reshape(X_train.shape[0], -1)
        y_flat = y_train.ravel()

        super_features = create_super_features(X_flat)
        X_hybrid = np.hstack((X_flat, super_features))
        
        self.model.fit(X_hybrid, y_flat)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Prédit avec l'ET hybride.
        """
        if self.model is None:
            raise RuntimeError("L'agent doit être entraîné ou chargé avant prédiction.")
            
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        super_features_test = create_super_features(X_test_flat)
        X_test_hybrid = np.hstack((X_test_flat, super_features_test))
        
        return self.model.predict(X_test_hybrid)

    def save(self, path: str = "artifacts/ExtraTreeAgent9features"):
        """Sauvegarde le modèle ExtraTrees."""
        try:
            os.makedirs(path, exist_ok=True)
            joblib.dump(self.model, os.path.join(path, "model.pkl"))
            print(f"Agent (ExtraTrees9) sauvegardé dans {path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'agent : {e}")

    def load(self, path: str = "artifacts/ExtraTreeAgent9features"):
        """Charge un modèle ExtraTrees pré-entraîné."""
        try:
            model_path = os.path.join(path, "model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modèle non trouvé : {model_path}")
            
            self.model = joblib.load(model_path)
            print(f"Agent (ExtraTrees9) chargé depuis {path}")
        except Exception as e:
            print(f"Erreur lors du chargement de l'agent : {e}")