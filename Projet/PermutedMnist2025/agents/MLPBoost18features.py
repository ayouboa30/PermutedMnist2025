import torch
from torch import nn
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.preprocessing import StandardScaler

# ================================================================
#  SUPER FEATURES v22 : Le "Paquet Complet" (18 Features)
# ================================================================
def create_super_features(X_flat: np.ndarray) -> np.ndarray:
    """Crée les 18 'super-features' globales (v22)."""
    
    # --- 1. Stats de Base (7 features) ---
    mean = np.mean(X_flat, axis=1)
    std = np.std(X_flat, axis=1)
    median = np.median(X_flat, axis=1)
    count_zero = np.count_nonzero(X_flat == 0, axis=1)
    count_max = np.count_nonzero(X_flat == 255, axis=1)
    skew = stats.skew(X_flat, axis=1)
    kurt = stats.kurtosis(X_flat, axis=1)

    # --- 2. Histogramme (8 features) ---
    hist_bins = np.apply_along_axis(
        lambda x: np.histogram(x, bins=8, range=(0, 256))[0],
        1,
        X_flat
    )
    
    # --- 3. Quantiles (3 features) ---
    q1 = np.percentile(X_flat, 25, axis=1)
    q3 = np.percentile(X_flat, 75, axis=1)
    iqr = q3 - q1 # Écart interquartile

    # Total = 7 + 8 + 3 = 18 features
    return np.hstack((
        mean[:, None], std[:, None], median[:, None], 
        count_zero[:, None], count_max[:, None], skew[:, None], kurt[:, None],
        hist_bins, 
        q1[:, None], q3[:, None], iqr[:, None]
    ))

# ================================================================
#  MLP v22 — 2 couches, 18 features
# ================================================================
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # On garde le modèle léger
        hidden_sizes = [256, 256] 
        dropout = 0.05
        d_in = 28 ** 2 + 18  # 784 pixels + 18 features

        layers = []
        for n in hidden_sizes:
            layers.append(nn.Linear(d_in, n))
            layers.append(nn.BatchNorm1d(n, momentum=0.2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            d_in = n
        layers.append(nn.Linear(d_in, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

==
class Agent:
    """Agent MLP (v22) — 2 couches + 18 features (Le Paquet Complet)."""

    def __init__(self, output_dim: int = 10, seed: int = None):
        self.output_dim = output_dim
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        torch.set_num_threads(2)
        self.model = None
        self.scaler = None

        self.batch_size = 256
        self.epochs = 7 # On garde 7 époques
        self.lr = 1e-3
        self.weight_decay = 1e-4

    def reset(self):
        self.model = Model() 
        self.scaler = StandardScaler()

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Entraîne le modèle MLP (plus léger) sur 802 features."""
        self.reset()

        if len(y_train.shape) > 1:
            y_train = y_train.squeeze()

        # --- Préparation features ---
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

            # On garde le "coupe-circuit" à 55s
            if time.time() - start > 55:
                print("⏱️ Temps limite atteint (55s), arrêt anticipé.")
                break

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Prédit les classes sur X_test."""
        if self.model is None:
            raise RuntimeError("L'agent doit être entraîné avant prédiction.")

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
        #98.4 en 55 secondes