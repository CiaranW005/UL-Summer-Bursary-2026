import numpy as np
import joblib

from sklearn.decomposition import PCA

def compute_pca(cls_tokens):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(cls_tokens)
    return coords, pca

def compute_pca_3d(cls_tokens):
    pca3 = PCA(n_components=3)
    coorsds_3d = pca3.fit_transform(cls_tokens)

    return  coorsds_3d, pca3

def load_or_compute(path, model_path, fn, cls_tokens):
    if path.exists() and model_path.exists():
        print("Loading ", path)
        return np.load(path), joblib.load(model_path)
    
    print("Computing ", path)
    coords, model = fn(cls_tokens)
    
    np.save(path, coords)
    joblib.dump(model, model_path)
    return coords, model

