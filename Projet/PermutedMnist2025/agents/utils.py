# agents/utils.py
import numpy as np
from scipy import stats

"""
Ce fichier centralise toutes les fonctions de création de features 
pour éviter la duplication de code dans les agents.
"""

def create_super_features_v5(X_flat: np.ndarray) -> np.ndarray:
    """Crée les 5 'super-features' globales (mean, std, median, zero_count, max_count)."""
    mean = np.mean(X_flat, axis=1)
    std = np.std(X_flat, axis=1)
    median = np.median(X_flat, axis=1)
    count_zero = np.count_nonzero(X_flat == 0, axis=1)
    count_max = np.count_nonzero(X_flat == 255, axis=1)
    return np.vstack((mean, std, median, count_zero, count_max)).T


def create_super_features_v9(X_flat: np.ndarray) -> np.ndarray:
    """Crée les 9 'super-features' globales (v5 + quantiles + moments)."""
    # 1. Statistiques de base (v5)
    mean = np.mean(X_flat, axis=1)
    std = np.std(X_flat, axis=1)
    median = np.median(X_flat, axis=1)
    count_zero = np.count_nonzero(X_flat == 0, axis=1)
    count_max = np.count_nonzero(X_flat == 255, axis=1)
    
    # 2. Percentiles
    q1 = np.percentile(X_flat, 25, axis=1)
    q3 = np.percentile(X_flat, 75, axis=1)
    
    # 3. Moments de distribution (Asymétrie et Aplatissement)
    skewness = stats.skew(X_flat, axis=1)
    kurt = stats.kurtosis(X_flat, axis=1)
    
    return np.vstack((
        mean, std, median, count_zero, count_max, q1, q3, skewness, kurt
    )).T


def create_super_features_v11_hist_q(X_flat: np.ndarray) -> np.ndarray:
    """Crée les features globales v20 (Histogramme 8 bins + 3 Quantiles)."""
    # 1. Histogramme binné (8 features)
    hist_bins = np.apply_along_axis(
        lambda x: np.histogram(x, bins=8, range=(0, 256))[0],
        1,
        X_flat
    )
    
    # 2. Quantiles (3 features)
    q1 = np.percentile(X_flat, 25, axis=1)
    median = np.median(X_flat, axis=1)
    q3 = np.percentile(X_flat, 75, axis=1)

    return np.hstack((
        hist_bins, 
        q1[:, None],
        median[:, None], 
        q3[:, None]
    ))


def create_super_features_v18_all(X_flat: np.ndarray) -> np.ndarray:
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

    return np.hstack((
        mean[:, None], std[:, None], median[:, None], 
        count_zero[:, None], count_max[:, None], skew[:, None], kurt[:, None],
        hist_bins, 
        q1[:, None], q3[:, None], iqr[:, None]
    ))


def create_super_features_v19_spatial(X_flat: np.ndarray) -> np.ndarray:
    """Crée les features globales v21 (11 hist/quantiles + 8 spatiales)."""
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

    return np.hstack((
        hist_bins,
        q1[:, None],
        median[:, None],
        q3[:, None],
        spatial_feats
    ))


def create_super_features_lgbm_hybrid(X_flat: np.ndarray) -> np.ndarray:
    """Extrait 5 features statistiques ultra-rapides pour LGBMHybride"""
    return np.column_stack([
        np.mean(X_flat, axis=1),      # Luminosité moyenne
        np.std(X_flat, axis=1),       # Contraste
        np.max(X_flat, axis=1),       # Pic d'intensité
        np.sum(X_flat > 128, axis=1), # Nb pixels blancs
        np.sum(X_flat > 0, axis=1)    # Nb pixels non-nuls
    ])