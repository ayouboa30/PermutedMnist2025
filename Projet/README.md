# Challenge Permuted MNIST 2025

[Python Version 3.9+](https://www.python.org/downloads/)  
[License MIT](https://opensource.org/licenses/MIT)  
Build Status: passing  

Projet développé par **Ayoub Oulad Ali** et **Mohy Mabrouk** (MS2A).

## Vue d’ensemble

Ce dépôt contient l'implémentation et l'évaluation de plusieurs agents pour le challenge **Permuted MNIST**. L'objectif est de classifier les chiffres MNIST sur des tâches où les pixels et les étiquettes sont permutés aléatoirement, avec une contrainte stricte de **60 secondes par tâche**.

Le projet inclut les livrables suivants : package Python, outils de benchmark, rapports d'analyse et une implémentation bonus de recherche.

## Fonctionnalités

- **Agents multiples** :
  - MLP (Multi-Layer Perceptron)
  - ExtraTrees
  - KNN (avec indexation [Faiss](https://github.com/facebookresearch/faiss))
  - LGBM (LightGBM)
- **Évaluation complète** :
  - Précision (accuracy)
  - Temps d'exécution (réel et CPU)
  - Pic d'utilisation mémoire (RAM)
- **Reproductibilité** : Notebooks dédiés (`run_benchmark.ipynb` et `Report.ipynb`) pour lancer les tests et générer les analyses.
- **Package installable** : Le projet est structuré comme un package Python installable via `pyproject.toml`.
- **Bonus recherche** : Implémentation du papier **"Naive Rehearsal"** pour l'apprentissage continu (Continual Learning).

## Structure du projet

```text
Projet/
├── .gitignore
├── Report.ipynb
├── RunAgents.py
├── agents/
│   ├── BestAgentMLP.py
│   ├── ExtraTreeAgent5features.py
│   ├── ExtraTreeAgent9features.py
│   ├── KNNFaiss.py
│   ├── LGBEXTHybride.py
│   ├── LGBMAgent5features.py
│   ├── MLPAgentBase.py
│   ├── MLPBoost18features.py
│   └── MLPv2_1.py
├── bonus/
│   └── implementation_papier_de_recherche/
│       ├── ContinualAgent.py
│       └── permuted_mnist/
├── brouillon/
│   └── agent.ipynb
├── permuted_mnist/
│   ├── __init__.py
│   ├── data/
│   └── env/
├── pyproject.toml
├── requirements.txt
├── run_benchmark.ipynb
└── soumissions/
```

Prérequis

Langage : Python 3.9+

Gestionnaire de paquets : pip

Installation
# 1. Cloner le dépôt
git clone https://github.com/ayouboa30/PermutedMnist2025.git

# 2. Aller dans le dossier du projet
cd PermutedMnist2025/Projet

# 3. Installer les dépendances
pip install -r requirements.txt

# (Optionnel) Installer le package localement
pip install .

 Utilisation
Étape 1 – Lancer le benchmark

Le notebook run_benchmark.ipynb exécute tous les agents via RunAgents.py et sauvegarde leurs performances dans soumissions/.

Étape 2 – Analyser les résultats

Le notebook Report.ipynb charge les fichiers .json des résultats et génère automatiquement tableaux et graphiques d’analyse.

Feuille de Route

 Tâche 1 – Implémenter l’environnement et le script RunAgents.py

 Tâche 2 – Développer plusieurs algorithmes de benchmark

 Tâche 3 – Ajouter le suivi du CPU et de la RAM

 Tâche 4 – Créer le notebook de benchmark

 Tâche 5 – Rédiger le rapport d’analyse complet

 Tâche 6 – Ajouter le packaging Python (pyproject.toml, requirements.txt)

 Tâche 7 – Implémenter le bonus (papier de recherche ET CI/CD)

Licence

Ce projet est distribué sous licence MIT.
Voir le fichier LICENSE
 pour plus de détails.

Réalisé par Ayoub Oulad Ali & Mohy Mabrouk — MS2A

