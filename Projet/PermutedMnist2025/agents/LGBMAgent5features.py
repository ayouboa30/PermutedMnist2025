import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os
import joblib

# Importation depuis le fichier utils
from .utils import create_super_features_v5 as create_super_features


class Agent:

    def __init__(self, output_dim: int = 10, seed: int = None):
        self.seed = seed
        self.model = None
        self.n_jobs = 2
        
        self.params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 64,
            'n_jobs': self.n_jobs,
            'random_state': self.seed,
            'colsample_bytree': 0.7,
            'subsample': 0.7,
        }
        self.early_stopping_rounds = 15
        self.validation_size = 0.15

    def reset(self):
        """Reset l'agent pour une nouvelle tâche."""
        self.model = lgb.LGBMClassifier(**self.params)

    def _preprocess_features(self, X):
        """Prépare les features pour le modèle (flatten + extraction)."""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        super_features = create_super_features(X)
        X_hybrid = np.hstack((X, super_features))
        return X_hybrid

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Entraîne le LGBM sur les 784 pixels + 5 features.
        """
        self.reset()
        
        y_flat = y_train.ravel()
        X_hybrid = self._preprocess_features(X_train)

        X_base, X_val, y_base, y_val = train_test_split(
            X_hybrid, y_flat, 
            test_size=self.validation_size, 
            random_state=self.seed, 
            stratify=y_flat
        )
        
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
            raise RuntimeError("L'agent doit être entraîné ou chargé avant prédiction.")
            
        X_test_hybrid = self._preprocess_features(X_test)
        return self.model.predict(X_test_hybrid)

    def save(self, path: str = "artifacts/LGBMAgent5features"):
        """Sauvegarde le modèle LGBM."""
        try:
            os.makedirs(path, exist_ok=True)
            joblib.dump(self.model, os.path.join(path, "model.pkl"))
            print(f"Agent (LGBM5) sauvegardé dans {path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'agent : {e}")

    def load(self, path: str = "artifacts/LGBMAgent5features"):
        """Charge un modèle LGBM pré-entraîné."""
        try:
            model_path = os.path.join(path, "model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modèle non trouvé : {model_path}")
            
            self.model = joblib.load(model_path)
            print(f"Agent (LGBM5) chargé depuis {path}")
        except Exception as e:
            print(f"Erreur lors du chargement de l'agent : {e}")