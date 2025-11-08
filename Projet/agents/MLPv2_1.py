import torch
from torch import nn
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.preprocessing import RobustScaler

# ================================================================
#  SUPER FEATURES v21 : Histogramme + Quantiles + Spatiales
# ================================================================
def create_super_features(X_flat: np.ndarray) -> np.ndarray:
    """Crée les features globales v21 (11 histogramme/quantiles + 8 spatiales)."""
    
    # 1. Histogramme binné (8 bins)
    hist_bins = np.apply_along_axis(
        lambda x: np.histogram(x, bins=8, range=(0, 256))[0],
        1,
        X_flat
    )

    # 2. Quantiles (3 features)
    q1 = np.percentile(X_flat, 25, axis=1)
    median = np.median(X_flat, axis=1)
    q3 = np.percentile(X_flat, 75, axis=1)

    # 3. Features spatiales légères
    X_img = X_flat.reshape(-1, 28, 28)
    coords = np.indices((28, 28))
    sum_X = np.sum(X_img, axis=(1, 2)) + 1e-6  # éviter division par 0

    x_mean = np.sum(coords[1] * X_img, axis=(1, 2)) / sum_X
    y_mean = np.sum(coords[0] * X_img, axis=(1, 2)) / sum_X

    top = np.mean(X_img[:, :14, :], axis=(1, 2))
    bottom = np.mean(X_img[:, 14:, :], axis=(1, 2))
    left = np.mean(X_img[:, :, :14], axis=(1, 2))
    right = np.mean(X_img[:, :, 14:], axis=(1, 2))

    flip_h = np.mean((X_img - np.flip(X_img, axis=2))**2, axis=(1, 2))
    flip_v = np.mean((X_img - np.flip(X_img, axis=1))**2, axis=(1, 2))

    spatial_feats = np.vstack((x_mean, y_mean, top, bottom, left, right, flip_h, flip_v)).T

    # 4. Concaténer toutes les features
    return np.hstack((
        hist_bins,
        q1[:, None],
        median[:, None],
        q3[:, None],
        spatial_feats
    ))

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

        # Hyperparams calibrés
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

            if time.time() - start > 55:
                print("⏱️ Temps limite atteint (~55s), arrêt anticipé.")
                break

    def predict(self, X_test: np.ndarray) -> np.ndarray:
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
#98.6% mais en 73 secondes