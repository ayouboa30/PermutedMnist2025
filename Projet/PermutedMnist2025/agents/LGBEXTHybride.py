import numpy as np
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier

class Agent:
    def __init__(self, output_dim=10, seed=None):
        self.output_dim = output_dim
        self.seed = seed if seed is not None else 42
        self.model = None
        
    def reset(self):
        self.model = None
    
    def _extract_features(self, X):
        """Extrait 5 features statistiques ultra-rapides"""
        # Flatten si nécessaire
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        # 5 features rapides et discriminantes
        features = np.column_stack([
            np.mean(X, axis=1),           # Luminosité moyenne
            np.std(X, axis=1),            # Contraste
            np.max(X, axis=1),            # Pic d'intensité
            np.sum(X > 128, axis=1),      # Nb pixels blancs
            np.sum(X > 0, axis=1)         # Nb pixels non-nuls
        ])
        
        return features
    
    def train(self, X_train, y_train):
        """Stratégie hybride : LightGBM sur sous-échantillon + features"""
        
        # Flatten
        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        y_train = y_train.ravel()
        n_samples = len(X_train)
        
        # SUB-SAMPLING : Clé de la vitesse de Milton
        # Entraîner sur 40% des données suffit pour >98% de précision
        sample_ratio = 0.40
        sample_size = int(n_samples * sample_ratio)
        
        np.random.seed(self.seed)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        
        X_sample = X_train[sample_idx]
        y_sample = y_train[sample_idx]
        
        # Features statistiques
        features = self._extract_features(X_sample)
        
        # Combine pixels + features
        X_combined = np.hstack([X_sample.astype(np.float32) / 255.0, features])
        
        # LightGBM optimisé pour vitesse ET précision
        self.model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=self.output_dim,
            n_estimators=300,           # Équilibre vitesse/précision
            num_leaves=80,              # Profondeur optimale
            learning_rate=0.05,
            max_depth=12,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=self.seed,
            n_jobs=2,                   # Utilise les 2 CPUs
            verbose=-1,
            force_col_wise=True         # Optimisation CPU
        )
        
        self.model.fit(X_combined, y_sample)
    
    def predict(self, X_test):
        """Prédiction rapide"""
        if self.model is None:
            return np.zeros(X_test.shape[0], dtype=np.int32)
        
        # Flatten
        if X_test.ndim == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Features
        features = self._extract_features(X_test)
        
        # Combine
        X_combined = np.hstack([X_test.astype(np.float32) / 255.0, features])
        
        # Prédiction
        predictions = self.model.predict(X_combined)
        
        return predictions.astype(np.int32)