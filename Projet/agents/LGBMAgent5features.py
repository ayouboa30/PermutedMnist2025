import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
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

    def __init__(self, output_dim: int = 10, seed: int = None):
        self.seed = seed
        self.model = None
        self.n_jobs = 2 # 2 CPUs
        
        # --- Paramètres équilibrés Vitesse/Précision ---
        self.params = {
            'n_estimators': 1000,   # Plafond (s'arrêtera avant)
            'learning_rate': 0.05,
            'num_leaves': 64,       # Plus rapide que 128
            'n_jobs': self.n_jobs,
            'random_state': self.seed,
            'colsample_bytree': 0.7,
            'subsample': 0.7,
        }
        self.early_stopping_rounds = 15
        self.validation_size = 0.15

    def reset(self):
        """Reset l'agent pour une nouvelle tâche."""
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Entraîne le LGBM sur les 784 pixels + 5 features.
        """
        
        # 1. Aplatir
        X_flat = X_train.reshape(X_train.shape[0], -1)
        y_flat = y_train.ravel()

        # 2. CRÉER LES SUPER-FEATURES (Ton idée)
        super_features = create_super_features(X_flat)
        
        # 3. Combiner les features
        X_hybrid = np.hstack((X_flat, super_features))

        # 4. Créer le set de validation pour l'arrêt précoce
        X_base, X_val, y_base, y_val = train_test_split(
            X_hybrid, y_flat, 
            test_size=self.validation_size, 
            random_state=self.seed, 
            stratify=y_flat
        )
        
        # 5. Créer et entraîner le modèle
        self.model = lgb.LGBMClassifier(**self.params)
        
        self.model.fit(
            X_base, 
            y_base,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        )

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Prédit avec le LGBM hybride.
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