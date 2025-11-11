import torch
from torch import nn
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset

class PermutationInvariantFeatures:
    """
    Features qui restent informatives malgré les permutations
    C'est le coeur de l'approche "meta-learning léger"
    """
    @staticmethod
    def extract(X_flat: np.ndarray) -> np.ndarray:
        # Statistiques d'ordre (invariantes aux permutations spatiales)
        mean = np.mean(X_flat, axis=1, keepdims=True)
        std = np.std(X_flat, axis=1, keepdims=True)
        
        # Distribution des intensités (histogramme compact)
        q10 = np.percentile(X_flat, 10, axis=1, keepdims=True)
        q25 = np.percentile(X_flat, 25, axis=1, keepdims=True)
        q50 = np.percentile(X_flat, 50, axis=1, keepdims=True)
        q75 = np.percentile(X_flat, 75, axis=1, keepdims=True)
        q90 = np.percentile(X_flat, 90, axis=1, keepdims=True)
        
        # Moments d'ordre supérieur
        centered = X_flat - mean
        skew = np.mean(centered**3, axis=1, keepdims=True) / (std**3 + 1e-6)
        kurt = np.mean(centered**4, axis=1, keepdims=True) / (std**4 + 1e-6)
        
        # Complexité/entropie approximative
        active_pixels = np.sum(X_flat > 30, axis=1, keepdims=True)
        edge_strength = np.sum(np.abs(np.diff(X_flat, axis=1)), axis=1, keepdims=True)
        
        return np.hstack([mean, std, q10, q25, q50, q75, q90, 
                          skew, kurt, active_pixels, edge_strength])


class MetaAdaptiveModel(nn.Module):
    """
    Architecture inspirée de MAML mais adaptée à la contrainte de temps
    - Couches avec initialisation spéciale pour adaptation rapide
    - Skip connections pour gradient flow
    """
    def __init__(self):
        super().__init__()
        d_pixels = 784
        d_features = 11
        d_in = d_pixels + d_features
        
        # Architecture avec skip connections (aide à l'adaptation)
        self.fc1 = nn.Linear(d_in, 512)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.2)
        
        self.fc2 = nn.Linear(512, 384)
        self.bn2 = nn.BatchNorm1d(384, momentum=0.2)
        
        self.fc3 = nn.Linear(384, 256)
        self.bn3 = nn.BatchNorm1d(256, momentum=0.2)
        
        # Skip connection
        self.skip = nn.Linear(512, 256)
        
        self.fc4 = nn.Linear(256, 10)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)
        
        # Initialisation MAML-style (variance réduite pour adaptation rapide)
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation optimisée pour l'adaptation rapide"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier avec gain réduit (facilite adaptation)
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Premier bloc
        out1 = self.dropout(self.relu(self.bn1(self.fc1(x))))
        
        # Deuxième bloc
        out2 = self.dropout(self.relu(self.bn2(self.fc2(out1))))
        
        # Troisième bloc avec skip
        out3 = self.relu(self.bn3(self.fc3(out2)))
        out3 = out3 + self.skip(out1)  # Skip connection
        out3 = self.dropout(out3)
        
        # Sortie
        return self.fc4(out3)


class Agent:
    def __init__(self, output_dim: int = 10, seed: int = None):
        self.output_dim = output_dim
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        torch.set_num_threads(2)
        self.model = None
        self.feature_mean = None
        self.feature_std = None
        
        # Hyperparams pour convergence rapide (style MAML)
        self.batch_size = 256
        
        # === CORRECTION POUR UTILISER 60s ===
        self.warmup_epochs = 4   # Phase d'adaptation plus longue
        self.main_epochs = 8     # Phase de consolidation plus longue
        # (Total = 12 époques au lieu de 6)
        # ====================================
        
        self.lr_warmup = 3e-3
        self.lr_main = 1e-3
        self.weight_decay = 1e-4
    
    def reset(self):
        self.model = MetaAdaptiveModel()
        self.feature_mean = None
        self.feature_std = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.reset()
        
        if len(y_train.shape) > 1:
            y_train = y_train.squeeze()
        
        # ... [Feature extraction et DataLoader - inchangé] ...
        
        X_flat = X_train.reshape(X_train.shape[0], -1)
        invariant_features = PermutationInvariantFeatures.extract(X_flat)
        
        X_pixels_norm = X_flat * (1.0 / 255.0)
        
        self.feature_mean = invariant_features.mean(axis=0)
        self.feature_std = invariant_features.std(axis=0) + 1e-6
        X_features_norm = (invariant_features - self.feature_mean) / self.feature_std
        
        X_hybrid = np.hstack((X_pixels_norm, X_features_norm)).astype(np.float32)
        X_train_tensor = torch.from_numpy(X_hybrid)
        y_train_tensor = torch.from_numpy(y_train).long()
        
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # === META-INSPIRED TRAINING ===
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr_warmup,
            weight_decay=self.weight_decay
        )
        
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        self.model.train()
        start = time.time()
        
        # === CORRECTION : Marge de temps (55s) ===
        # (J'utilise 55s pour laisser 5s de marge, comme suggéré par votre commentaire)
        TIME_LIMIT_SEC = 55 
        
        # WARMUP
        for epoch in range(self.warmup_epochs):
            for xb, yb in train_loader:
                optimizer.zero_grad(set_to_none=True)
                out = self.model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                optimizer.step()
            
            # === CORRECTION : Check de temps cohérent ===
            if time.time() - start > TIME_LIMIT_SEC: 
                print(f"⏱️ Temps limite ({TIME_LIMIT_SEC}s) atteint, arrêt anticipé.")
                self.model.eval() # Important
                return
        
        # Phase 2: Consolidation avec LR réduit
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr_main
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.main_epochs
        )
        
        for epoch in range(self.main_epochs):
            for xb, yb in train_loader:
                optimizer.zero_grad(set_to_none=True)
                out = self.model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            
            # === CORRECTION : Check de temps cohérent ===
            if time.time() - start > TIME_LIMIT_SEC: 
                print(f"⏱️ Temps limite ({TIME_LIMIT_SEC}s) atteint, arrêt anticipé.")
                self.model.eval() # Important
                return
        
        self.model.eval() # Assure que le modèle est en mode eval

    def load(self, path: str = "artifacts/MetaAgent"):
        """Charge un modèle (poids) et les stats de features."""
        try:
            stats_path = os.path.join(path, "feature_stats.pkl")
            weights_path = os.path.join(path, "model_weights.pth")
            
            if not (os.path.exists(stats_path) and os.path.exists(weights_path)):
                raise FileNotFoundError(f"Fichiers non trouvés dans {path}")

            # Charger les stats
            feature_stats = joblib.load(stats_path)
            self.feature_mean = feature_stats['mean']
            self.feature_std = feature_stats['std']
            
            # Charger le modèle
            self.model = MetaAdaptiveModel()
            
            # === CORRECTION : Robustesse au chargement (CPU/GPU) ===
            device = torch.device('cpu')
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
            # =======================================================
            
            self.model.eval()
            
            print(f"Agent (Meta) chargé depuis {path}")
        except Exception as e:
            print(f"Erreur lors du chargement de l'agent : {e}")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("L'agent doit être entraîné avant prédiction.")
        
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        invariant_features_test = PermutationInvariantFeatures.extract(X_test_flat)
        
        X_pixels_norm = X_test_flat * (1.0 / 255.0)
        X_features_norm = (invariant_features_test - self.feature_mean) / self.feature_std
        
        X_hybrid_test = np.hstack((X_pixels_norm, X_features_norm)).astype(np.float32)
        X_test_tensor = torch.from_numpy(X_hybrid_test)
        
        test_loader = DataLoader(X_test_tensor, batch_size=512, num_workers=0)
        
        preds = []
        self.model.eval()
        with torch.no_grad():
            for xb in test_loader:
                out = self.model(xb)
                preds.append(out.argmax(1).cpu().numpy())
        
        return np.concatenate(preds)