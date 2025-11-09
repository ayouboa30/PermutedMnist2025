import torch
from torch import nn
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.preprocessing import RobustScaler
import os
import joblib

# Importation depuis le fichier utils
from .utils import create_super_features_v19_spatial as create_super_features

# ================================================================
#  MLP v21 — 3 couches, 19 features
# ================================================================
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        hidden_sizes = [512, 384, 192]
        dropout = 0.04
        d_in = 28**2 + 19  # 784 pixels + 19 features

        layers = []
        for n in hidden_sizes:
            layers.append(nn.Linear(d_in, n))
            layers.append(nn.BatchNorm1d(n, momentum=0.3))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            d_in = n
        layers.append(nn.Linear(d_in, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ================================================================
#  AGENT MILTON v21 — 19 Features, ~55s
# ================================================================
class Agent:
    """Agent MLP (v21) — 3 couches + 19 features, optimisé CPU 2 threads."""

    def __init__(self, output_dim: int = 10, seed: int = None):
        self.output_dim = output_dim
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        torch.set_num_threads(2)
        self.model = None
        self.scaler = None

        self.batch_size = 128
        self.epochs = 7
        self.lr = 1.2e-3
        self.weight_decay = 1e-4

    def reset(self):
        self.model = Model()
        self.scaler = RobustScaler()

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.reset()

        if len(y_train.shape) > 1:
            y_train = y_train.squeeze()

        X_flat = X_train.reshape(X_train.shape[0], -1)
        super_features = create_super_features(X_flat)

        X_pixels_norm = X_flat / 255.0
        X_features_norm = self.scaler.fit_transform(super_features)

        X_hybrid = np.hstack((X_pixels_norm, X_features_norm)).astype(np.float32)
        X_train_tensor = torch.from_numpy(X_hybrid)
        y_train_tensor = torch.from_numpy(y_train).long()

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        start = time.time()

        for epoch in range(self.epochs):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if time.time() - start > 55:
                print("⏱️ Temps limite atteint (~55s), arrêt anticipé.")
                break

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("L'agent doit être entraîné ou chargé avant prédiction.")

        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        super_features_test = create_super_features(X_test_flat)

        X_pixels_norm = X_test_flat / 255.0
        X_features_norm = self.scaler.transform(super_features_test)
        X_hybrid_test = np.hstack((X_pixels_norm, X_features_norm)).astype(np.float32)

        X_test_tensor = torch.from_numpy(X_hybrid_test)
        test_loader = DataLoader(X_test_tensor, batch_size=self.batch_size * 2)

        preds = []
        self.model.eval()
        with torch.no_grad():
            for xb in test_loader:
                out = self.model(xb)
                preds.append(out.argmax(1).cpu().numpy())
        return np.concatenate(preds)

    def save(self, path: str = "artifacts/MLPv2_1"):
        """Sauvegarde le modèle (poids) et le scaler."""
        try:
            os.makedirs(path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(path, "model_weights.pth"))
            joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))
            print(f"Agent (MLPv21) sauvegardé dans {path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'agent : {e}")

    def load(self, path: str = "artifacts/MLPv2_1"):
        """Charge un modèle (poids) et un scaler pré-entraînés."""
        try:
            scaler_path = os.path.join(path, "scaler.pkl")
            weights_path = os.path.join(path, "model_weights.pth")
            
            if not (os.path.exists(scaler_path) and os.path.exists(weights_path)):
                raise FileNotFoundError(f"Fichiers non trouvés dans {path}")

            self.scaler = joblib.load(scaler_path)
            self.model = Model()
            self.model.load_state_dict(torch.load(weights_path))
            self.model.eval()
            print(f"Agent (MLPv21) chargé depuis {path}")
        except Exception as e:
            print(f"Erreur lors du chargement de l'agent : {e}")