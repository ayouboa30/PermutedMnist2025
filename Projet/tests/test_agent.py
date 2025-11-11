import pytest
import numpy as np
import torch
import os

# Assurez-vous que l'agent est importable
# (Ceci suppose que votre 'pip install .' a fonctionné)
from PermutedMnist2025.agents.MetaAdapModBESTT import Agent, PermutationInvariantFeatures, MetaAdaptiveModel

# Variable globale pour la taille des features
# (Basé sur votre code : 784 pixels + 11 features)
D_PIXELS = 784
D_FEATURES = 11
D_IN = D_PIXELS + D_FEATURES
N_CLASSES = 10

@pytest.fixture
def dummy_data():
    """Crée de fausses données pour les tests."""
    # Petit batch de 5 images
    X_train = np.random.randint(0, 256, size=(5, D_PIXELS)).astype(np.float32)
    y_train = np.random.randint(0, N_CLASSES, size=(5,)).astype(np.int64)
    X_test = np.random.randint(0, 256, size=(3, D_PIXELS)).astype(np.float32)
    return X_train, y_train, X_test

def test_agent_init():
    """Teste si l'agent s'initialise correctement."""
    agent = Agent()
    assert agent.output_dim == N_CLASSES
    assert agent.model is None
    assert agent.feature_mean is None
    assert agent.feature_std is None

def test_agent_reset():
    """Teste si le reset recrée bien le modèle."""
    agent = Agent()
    agent.reset()
    assert agent.model is not None
    assert isinstance(agent.model, MetaAdaptiveModel)

def test_feature_extraction_shape():
    """Teste la forme de sortie de l'extracteur de features."""
    X_flat = np.random.rand(10, D_PIXELS) # 10 échantillons
    features = PermutationInvariantFeatures.extract(X_flat)
    # Vérifie si le nombre de features est correct (11 dans votre code)
    assert features.shape == (10, D_FEATURES)

def test_model_forward_pass_shape():
    """Teste si le modèle PyTorch a les bonnes dimensions d'entrée/sortie."""
    model = MetaAdaptiveModel()
    model.eval() # Mettre en mode eval pour les tests (BN, Dropout)
    
    # Batch de 2
    dummy_input = torch.rand(2, D_IN) 
    with torch.no_grad():
        output = model(dummy_input)
    
    # La sortie doit être (batch_size, n_classes)
    assert output.shape == (2, N_CLASSES)

def test_train_smoke_test(dummy_data):
    """
    Test "Smoke" : Vérifie si l'agent.train() s'exécute sans crasher.
    C'est un test crucial.
    """
    agent = Agent()
    X_train, y_train, _ = dummy_data
    
    # On réduit le nombre d'époques pour que le test soit rapide
    agent.warmup_epochs = 1
    agent.main_epochs = 1
    
    try:
        agent.train(X_train, y_train)
    except Exception as e:
        pytest.fail(f"agent.train() a levé une exception inattendue: {e}")
    
    # Après train, le modèle et les stats doivent être définis
    assert agent.model is not None
    assert agent.feature_mean is not None
    assert agent.feature_std is not None
    assert agent.feature_mean.shape == (D_FEATURES,)

def test_predict_shape_after_train(dummy_data):
    """Teste si agent.predict() retourne la bonne forme après un train."""
    agent = Agent()
    X_train, y_train, X_test = dummy_data
    
    # Entraînement rapide
    agent.warmup_epochs = 1
    agent.main_epochs = 1
    agent.train(X_train, y_train)
    
    # Prédiction
    predictions = agent.predict(X_test)
    
    assert predictions.shape == (X_test.shape[0],) # (3,)
    assert predictions.dtype == np.int64

def test_save_load_cycle(dummy_data, tmp_path):
    """
    Test VITAL : Vérifie si un agent sauvegardé et rechargé 
    donne les mêmes prédictions.
    `tmp_path` est une fixture de pytest qui crée un dossier temporaire.
    """
    agent1 = Agent()
    X_train, y_train, X_test = dummy_data
    
    # Entraîner agent1
    agent1.warmup_epochs = 1
    agent1.main_epochs = 1
    agent1.train(X_train, y_train)
    
    # Prédictions d'agent1
    preds1 = agent1.predict(X_test)
    
    # Sauvegarde
    save_dir = os.path.join(tmp_path, "test_model_save")
    agent1.save(save_dir)
    
    # Vérifier que les fichiers sont créés
    assert os.path.exists(os.path.join(save_dir, "model_weights.pth"))
    assert os.path.exists(os.path.join(save_dir, "feature_stats.pkl"))
    
    # Créer un NOUVEL agent et charger
    agent2 = Agent()
    agent2.load(save_dir)
    
    # Les stats et le modèle doivent être chargés
    assert agent2.model is not None
    assert agent2.feature_mean is not None
    
    # Faire les mêmes prédictions
    preds2 = agent2.predict(X_test)
    
    # Les prédictions doivent être identiques !
    np.testing.assert_array_equal(preds1, preds2)