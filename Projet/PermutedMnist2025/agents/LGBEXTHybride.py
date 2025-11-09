import numpy as np
import lightgbm as lgb
import os
import joblib

# Importation depuis le fichier utils
from .utils import create_super_features_lgbm_hybrid as _extract_features

class Agent:
    def __init__(self, output_dim=10, seed=None):
        self.output_dim = output_dim
        self.seed = seed if seed is not None else 42
        self.model = None
        
    def reset(self):
        # Initialise le modèle ici pour qu'il soit prêt à être entraîné
        self.model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=self.output_dim,
            n_estimators=300,
            num_leaves=80,
            learning_rate=0.05,
            max_depth=12,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=self.seed,
            n_jobs=2,
            verbose=-1,
            force_col_wise=True
        )
    
    def _preprocess_features(self, X):
        """Prépare les features pour le modèle (flatten + extraction)."""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        # Features statistiques
        features = _extract_features(X)
        
        # Combine pixels + features
        X_combined = np.hstack([X.astype(np.float32) / 255.0, features])
        return X_combined
    
    def train(self, X_train, y_train):
        """Stratégie hybride : LightGBM sur sous-échantillon + features"""
        self.reset()
        
        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
        y_train = y_train.ravel()
        n_samples = len(X_train)
        
        sample_ratio = 0.40
        sample_size = int(n_samples * sample_ratio)
        
        np.random.seed(self.seed)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        
        X_sample = X_train[sample_idx]
        y_sample = y_train[sample_idx]
        
        X_combined = self._preprocess_features(X_sample)
        self.model.fit(X_combined, y_sample)
    
    def predict(self, X_test):
        """Prédiction rapide"""
        if self.model is None:
            raise RuntimeError("L'agent doit être entraîné ou chargé avant prédiction.")
        
        X_combined_test = self._preprocess_features(X_test)
        predictions = self.model.predict(X_combined_test)
        
        return predictions.astype(np.int32)

    def save(self, path: str = "artifacts/LGBEXTHybride"):
        """Sauvegarde le modèle LGBM."""
        try:
            os.makedirs(path, exist_ok=True)
            joblib.dump(self.model, os.path.join(path, "model.pkl"))
            print(f"Agent (LGBMHybride) sauvegardé dans {path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'agent : {e}")

    def load(self, path: str = "artifacts/LGBEXTHybride"):
        """Charge un modèle LGBM pré-entraîné."""
        try:
            model_path = os.path.join(path, "model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modèle non trouvé : {model_path}")
            
            self.model = joblib.load(model_path)
            print(f"Agent (LGBMHybride) chargé depuis {path}")
        except Exception as e:
            print(f"Erreur lors du chargement de l'agent : {e}")