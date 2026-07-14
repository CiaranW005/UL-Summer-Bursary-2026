import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from .dim_reduction.utils import *

def plot_ellipsoid(meta, output_dir=None, cache_dir=None):
    img_dir = output_dir / "mahlanobis_fit" 
    os.makedirs(img_dir, exist_ok=True)

    category = "screw"

    train_mask = meta["split"] == "train"
    train_meta = meta[train_mask]
    test_meta = meta[~train_mask]

    train_cat_mask = train_meta["category"] == category

    defect_test_cat_mask = (test_meta["category"] == category) & (test_meta["type"] != "good")

    pca_2d, pca = load_or_compute(cache_dir / "cls_pca.npy", cache_dir / "pca.joblib", compute_pca)
    pca_train = pca_2d[train_mask]
    pca_test = pca_2d[~train_mask]

    train_2d = pca_train[train_cat_mask]
    test_2d = pca_test[defect_test_cat_mask]

    center = train_2d.mean(axis=0)

    cov = np.cov(train_2d.T)
    cov += np.eye(cov.shape[0]) * 1e-6
    cov_inv = np.linalg.pinv(cov)

    diff = train_2d - center
    d2 = np.einsum("ij, jk, ik->i", diff, cov_inv, diff)

    ch2 = d2.max()

    eigvals, eigvecs = np.linalg.eigh(cov)

    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    width = 2 * np.sqrt(eigvals[0] * ch2)
    height = 2 * np.sqrt(eigvals[1] * ch2)

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    _, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x=train_2d[:, 0], y=train_2d[:, 1], label="Normal")
    ax.scatter(x=test_2d[:, 0], y=test_2d[:, 1], label="Defective")

    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        edgecolor="red",
        facecolor="none",
        linewidth=2
    )

    ax.add_patch(ellipse)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{img_dir}/{category}.svg")
    plt.show()