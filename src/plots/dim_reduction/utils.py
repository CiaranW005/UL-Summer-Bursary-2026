import numpy as np
import joblib

from sklearn.decomposition import PCA

def compute_pca(cls_tokens):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(cls_tokens)
    return coords, pca

def load_or_compute(path, model_path, fn):
    if path.exists() and model_path.exists():
        print("Loading ", path)
        return np.load(path), joblib.load(model_path)
    
    print("Computing ", path)
    coords, model = fn()
    
    np.save(path, coords)
    joblib.dump(model, model_path)
    return coords, model