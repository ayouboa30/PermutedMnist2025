"""
PyTorch MLP Agent for Permuted MNIST
Multi-layer perceptron with batch normalization
"""
import torch
from torch import nn
import numpy as np
from time import time
import os
# (Pas besoin de joblib ici car pas de scaler)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        hidden_sizes = [400, 400]
        layers = []
        d_in = 28 ** 2
        for i, n in enumerate(hidden_sizes):
            layers.append(nn.Linear(d_in, n))
            layers.append(nn.BatchNorm1d(n))
            layers.append(nn.ReLU())
            d_in = n
        layers += [nn.Linear(d_in, 10)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)


class Agent:
    """PyTorch MLP agent for MNIST classification"""
    def __init__(self, output_dim: int = 10, seed: int = None):
        self.output_dim = output_dim
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.model = Model()
        self.batch_size = 16
        self.validation_fraction = 0.2
        self.verbose = True

    def reset(self):
        """Reset the agent for a new task/simulation"""
        self.model = Model()

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        if len(y_train.shape) > 1:
            y_train = y_train.squeeze()

        N_val = int(X_train.shape[0] * self.validation_fraction)
        X_train_sub, X_val = X_train[N_val:], X_train[:N_val]
        y_train_sub, y_val = y_train[N_val:], y_train[:N_val]

        X_train_sub = torch.from_numpy(X_train_sub).float() / 255.0
        X_val = torch.from_numpy(X_val).float() / 255.0
        y_train_sub = torch.from_numpy(y_train_sub).long()
        y_val = torch.from_numpy(y_val).long()

        N = len(X_train_sub)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()

        for i_epoch in range(10):
            perm = np.random.permutation(N)
            X = X_train_sub[perm]
            Y = y_train_sub[perm]

            for i in range(0, N, self.batch_size):
                x = X[i:i + self.batch_size]
                y = Y[i:i + self.batch_size]
                optimizer.zero_grad()
                logits = self.model(x)
                loss = ce(logits, y)
                loss.backward()
                optimizer.step()

            if self.verbose and self.validation_fraction > 0:
                y_predict = self.predict(X_val.numpy() * 255.0)
                is_correct = y_predict == y_val.numpy()
                acc = np.mean(is_correct)
                print(f"epoch {i_epoch}: {acc:0.04f}%")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("L'agent doit être entraîné ou chargé avant prédiction.")
            
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float() / 255.0
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward(X_test)
        return logits.argmax(-1).detach().cpu().numpy()

    def save(self, path: str = "artifacts/MLPAgentBase"):
        """Sauvegarde les poids du modèle."""
        try:
            os.makedirs(path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(path, "model_weights.pth"))
            print(f"Agent (MLPBase) sauvegardé dans {path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'agent : {e}")

    def load(self, path: str = "artifacts/MLPAgentBase"):
        """Charge les poids d'un modèle pré-entraîné."""
        try:
            weights_path = os.path.join(path, "model_weights.pth")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Poids non trouvés : {weights_path}")
            
            self.model = Model()
            self.model.load_state_dict(torch.load(weights_path))
            self.model.eval()
            print(f"Agent (MLPBase) chargé depuis {path}")
        except Exception as e:
            print(f"Erreur lors du chargement de l'agent : {e}")