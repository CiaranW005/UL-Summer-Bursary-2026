import numpy as np
import joblib

from sklearn.decomposition import PCA

from pathlib import Path
from collections.abc import Callable

PCAResult = tuple[np.ndarray, PCA]

def compute_pca(cls_tokens: np.ndarray) -> PCAResult:
    """Project embeddings into two principal components"""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(cls_tokens)

    return coords, pca

def compute_pca_3d(cls_tokens: np.ndarray) -> PCAResult:
    """Project embeddings into three principal components"""
    pca3 = PCA(n_components=3)
    coorsds_3d = pca3.fit_transform(cls_tokens)

    return  coorsds_3d, pca3

def load_or_compute(
        path: Path, 
        model_path: Path, 
        fn: Callable[[np.ndarray], PCAResult], 
        cls_tokens: np.ndarray
    ) -> PCAResult:

    """Load cached PCA results or compute and save them"""
    if path.exists() and model_path.exists():
        # TODO: compare path and model make times so it recomputes them if the model is newer
        print("Loading ", path)
        return np.load(path), joblib.load(model_path)
    
    print("Computing ", path)
    coords, model = fn(cls_tokens)
    
    np.save(path, coords)
    joblib.dump(model, model_path)
    return coords, model

