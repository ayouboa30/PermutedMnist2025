import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import time
from scipy import stats


def create_super_features(X_flat: np.ndarray) -> np.ndarray:
    """Crée les 5 'super-features' globales."""
    
    # 1. Statistiques globales
    mean = np.mean(X_flat, axis=1)
    std = np.std(X_flat, axis=1)
    median = np.median(X_flat, axis=1)
    
    # 2. Comptage de pixels
    count_zero = np.count_nonzero(X_flat == 0, axis=1)
    count_max = np.count_nonzero(X_flat == 255, axis=1)
    
    # Retourne un tableau de (N_samples, 5 features)
    return np.vstack((mean, std, median, count_zero, count_max)).T

class Agent:
    """
    Agent ExtraTrees-Seul + 5 "Super-Features" (moyenne, std, etc.).
    """
    def __init__(self, output_dim: int = 10, seed: int = None):
        self.seed = seed
        self.model = None
        self.n_jobs = 2 # 2 CPUs
        
        # --- Paramètres ET (puissants, car on a le temps) ---
        self.params = {
            'n_estimators': 150,    # Comme notre baseline (27s)
            'max_depth': 25,
            'n_jobs': self.n_jobs,
            'random_state': self.seed,
        }

    def reset(self):
        """Reset l'agent pour une nouvelle tâche."""
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Entraîne l'ET sur les 784 pixels + 5 features.
        """
        
        # 1. Aplatir
        X_flat = X_train.reshape(X_train.shape[0], -1)
        y_flat = y_train.ravel()

        # 2. CRÉER LES SUPER-FEATURES (Ton idée)
        super_features = create_super_features(X_flat)
        
        # 3. Combiner les features
        X_hybrid = np.hstack((X_flat, super_features)) # 789 features
        
        # 4. Créer et entraîner le modèle
        self.model = ExtraTreesClassifier(**self.params)
        self.model.fit(X_hybrid, y_flat)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Prédit avec l'ET hybride.
        """
        if self.model is None:
            raise RuntimeError("L'agent doit être entraîné.")
            
        # 1. Aplatir
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # 2. CRÉER LES SUPER-FEATURES pour le test
        super_features_test = create_super_features(X_test_flat)
        
        # 3. Combiner
        X_test_hybrid = np.hstack((X_test_flat, super_features_test))
        
        # 4. Prédire
        return self.model.predict(X_test_hybrid)