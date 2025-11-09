import torch
from torch import nn
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
import random
from typing import List, Tuple
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from permuted_mnist.env.permuted_mnist import PermutedMNISTEnv
except ImportError:
    print("Erreur: Le dossier 'permuted_mnist' est introuvable.", file=sys.stderr)
    sys.exit(1)

BUFFER_SIZE = 4400 
SAMPLES_TO_ADD_PER_TASK = 440
BATCH_SIZE = 128

class _MLPModel(nn.Module):
    def __init__(self):
        super(_MLPModel, self).__init__()
        d_in = 28 * 28
        h1 = 400
        h2 = 400
        d_out = 10
        
        self.model = nn.Sequential(
            nn.Linear(d_in, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, d_out)
        )

    def forward(self, x):
        return self.model(x)

class Agent:
    def __init__(self, output_dim: int = 10, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
        torch.set_num_threads(2)
        
        self.model = _MLPModel()
        
        self.epochs = 5
        self.lr = 1e-4
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.rehearsal_buffer: List[Tuple[np.ndarray, int]] = []
        self.task_count = 0

    def _prepare_data(self, X_data, y_data):
        X_flat = X_data.reshape(X_data.shape[0], -1)
        X_norm = (X_flat / 255.0).astype(np.float32)
        
        if len(y_data.shape) > 1:
            y_data = y_data.squeeze()
        
        X_tensor = torch.from_numpy(X_norm)
        y_tensor = torch.from_numpy(y_data).long()
        
        return X_tensor, y_tensor

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.task_count += 1
        self.model.train()
        
        X_current_tensor, y_current_tensor = self._prepare_data(X_train, y_train)
        current_dataset = TensorDataset(X_current_tensor, y_current_tensor)
        
        if not self.rehearsal_buffer:
            print(f"Tâche {self.task_count} (Tâche 1): Buffer vide. Entraînement normal.")
            train_loader = DataLoader(
                current_dataset,
                batch_size=BATCH_SIZE * 2,
                shuffle=True,
                num_workers=0
            )
            
            for epoch in range(self.epochs):
                for xb, yb in train_loader:
                    self.optimizer.zero_grad()
                    out = self.model(xb)
                    loss = self.loss_fn(out, yb)
                    loss.backward()
                    self.optimizer.step()
        
        else:
            print(f"Tâche {self.task_count}: Entraînement avec {len(self.rehearsal_buffer)} échantillons du buffer.")
            
            current_loader = DataLoader(
                current_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0
            )
            
            buffer_X = np.array([item[0] for item in self.rehearsal_buffer])
            buffer_y = np.array([item[1] for item in self.rehearsal_buffer])
            
            X_rehearsal_tensor, y_rehearsal_tensor = self._prepare_data(buffer_X, buffer_y)
            rehearsal_dataset = TensorDataset(X_rehearsal_tensor, y_rehearsal_tensor)
            
            rehearsal_loader = DataLoader(
                rehearsal_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0
            )
            
            for epoch in range(self.epochs):
                rehearsal_iter = iter(rehearsal_loader)
                
                for xb_curr, yb_curr in current_loader:
                    
                    try:
                        xb_re, yb_re = next(rehearsal_iter)
                    except StopIteration:
                        rehearsal_iter = iter(rehearsal_loader)
                        xb_re, yb_re = next(rehearsal_iter)
                        
                    xb = torch.cat((xb_curr, xb_re), dim=0)
                    yb = torch.cat((yb_curr, yb_re), dim=0)
                    
                    self.optimizer.zero_grad()
                    out = self.model(xb)
                    loss = self.loss_fn(out, yb)
                    loss.backward()
                    self.optimizer.step()

        indices = list(range(len(X_train)))
        random.shuffle(indices)
        indices_to_add = indices[:SAMPLES_TO_ADD_PER_TASK]
        
        new_samples_X = X_train[indices_to_add]
        
        if len(y_train.shape) > 1:
            y_train_squeezed = y_train.squeeze()
        else:
            y_train_squeezed = y_train
            
        new_samples_y = y_train_squeezed[indices_to_add]

        for i in range(len(new_samples_X)):
            self.rehearsal_buffer.append((new_samples_X[i], new_samples_y[i]))

        overflow = len(self.rehearsal_buffer) - BUFFER_SIZE
        if overflow > 0:
            self.rehearsal_buffer = self.rehearsal_buffer[overflow:]
        
        print(f"Buffer mis à jour. Taille actuelle: {len(self.rehearsal_buffer)} / {BUFFER_SIZE}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.model.eval()
        
        X_test_tensor, _ = self._prepare_data(X_test, np.zeros(len(X_test)))
        
        test_loader = DataLoader(X_test_tensor, batch_size=BATCH_SIZE * 2)

        preds = []
        with torch.no_grad():
            for xb in test_loader:
                out = self.model(xb)
                preds.append(out.argmax(1).cpu().numpy())
        
        return np.concatenate(preds)

if __name__ == "__main__":
    
    print("Lancement du test autonome de l'Agent d'Apprentissage Continu (Naive Rehearsal)")
    
    NUMBER_EPISODES = 10
    SEED = 42

    env = PermutedMNISTEnv(number_episodes=NUMBER_EPISODES)
    env.set_seed(SEED)
    
    agent = Agent(output_dim=10, seed=SEED)
    
    accuracies = []
    times = []
    task_num = 1
    
    while True:
        task = env.get_next_task()
        if task is None:
            break
        
        print(f"\n--- Tâche {task_num}/{NUMBER_EPISODES} ---")
        
        start_time = time.time()
        
        agent.train(task['X_train'], task['y_train'])
        predictions = agent.predict(task['X_test'])
        
        elapsed_time = time.time() - start_time
        accuracy = env.evaluate(predictions, task['y_test'])
        
        accuracies.append(accuracy)
        times.append(elapsed_time)
        
        print(f"Tâche {task_num}: Précision = {accuracy:.2%}, Temps = {elapsed_time:.4f}s")
        print(f"Précisions cumulées: {[f'{a:.2%}' for a in accuracies]}")
        
        task_num += 1
    
    print("\n" + "="*60)
    print("Évaluation de l'Apprentissage Continu Terminée")
    print(f"Précision moyenne finale: {np.mean(accuracies):.2%}")
    print(f"Temps total: {np.sum(times):.2f}s")