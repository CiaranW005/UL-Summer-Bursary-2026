import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

import faiss

from pathlib import Path
from .dim_reduction.utils import *

def plot_ellipsoid(
        masks,     
        output_dir: Path, 
        cache_dir: Path 
        ):
    
    img_dir = output_dir / "mahlanobis_fit" 
    os.makedirs(img_dir, exist_ok=True)

    pca_2d, _ = load_or_compute(cache_dir / "cls_pca.npy", cache_dir / "pca.joblib", compute_pca)
    pca_train = pca_2d[masks.train_mask]
    pca_test = pca_2d[masks.test_mask]

    train_2d = pca_train[masks.train_category_mask]
    test_2d = pca_test[masks.defect_test_mask]

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
    plt.savefig(f"{img_dir}/{masks.category}.svg")
    plt.show()

def plot_K(masks, cls_tokens, output_dir: Path, cache_dir: Path):
    img_dir = output_dir / "k_rate"
    os.makedirs(img_dir, exist_ok=True)

    train_emb = cls_tokens[masks.train_mask]
    cat_emb = train_emb[masks.train_category_mask]
    
    pca_2d, _ = load_or_compute(cache_dir / "cls_pca.npy", cache_dir / "pca.joblib", compute_pca)
    pca_train = pca_2d[masks.train_mask]
    pca_cat = pca_train[masks.train_category_mask]

    print("Cat length: ", len(cat_emb))

    fig, axes = plt.subplots(1, 3, figsize=(10, 6))

    for i, frac in enumerate([0.025, 0.05, 0.10]):
        K = max(2, int(frac * len(cat_emb)))

        index = faiss.IndexFlatL2(cat_emb.shape[1])
        index.add(cat_emb.astype("float32"))

        D, I = index.search(
            cat_emb.astype("float32"),
            k=K+1 # Nearest is itself
        )

        D = D[:, 1:]    # Remove self
        I = I[:, 1:]

        avg_knn = D.mean(axis=1)

        most_compact_idx = avg_knn.argmin()
        neighbours = I[most_compact_idx]

        center_2d = pca_cat[most_compact_idx]

        neighbours_2d = pca_cat[neighbours]
        radius_2d = np.linalg.norm(neighbours_2d - center_2d, axis=1).max()

        ax = axes[i]
        ax.scatter(pca_cat[:, 0], pca_cat[:, 1], alpha=0.4, label="Uncovered Embeddings")
        ax.scatter(neighbours_2d[:, 0], neighbours_2d[:, 1], alpha=0.9, label="Covered Embeddings")
        ax.scatter(center_2d[0], center_2d[1], marker="x", s=120, label="Centroid")

        circle = Circle(
            center_2d,
            radius_2d,
            fill=False,
            linewidth=2
        )

        ax.add_patch(circle)
        ax.set_title(f"K={K} ({frac*100:.1f}%)")
        ax.set_aspect("equal", adjustable="box")

        print("frac:", frac)
        print("K:", K)
        print("most compact idx:", most_compact_idx)
        print("avg KNN dist:", avg_knn[most_compact_idx])

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(0.4125, 0.15)
    )

    plt.tight_layout()
    plt.savefig(f"{img_dir}/{masks.category}.svg", bbox_inches="tight")
    plt.show()

def plot_growth_rates(masks, cls_tokens, output_dir: Path, cache_dir: Path):
    img_dir = output_dir/ "growth_rate"
    os.makedirs(img_dir, exist_ok=True)

    train_emb = cls_tokens[masks.train_mask]
    cat_emb = train_emb[masks.train_category_mask]

    pca_2d, pca = load_or_compute(cache_dir / "cls_pca.npy", cache_dir / "pca.joblib", compute_pca)
    pca_train = pca_2d[masks.train_mask]
    pca_cat = pca_train[masks.train_category_mask]

    print("Cat length: ", len(cat_emb))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    K = max(2, int(0.05 * len(cat_emb)))

    index = faiss.IndexFlatL2(cat_emb.shape[1])
    index.add(cat_emb.astype("float32"))

    D, I = index.search(
        cat_emb.astype("float32"),
        k=K+1 # Nearest is itself
    )

    D = D[:, 1:]    # Remove self
    I = I[:, 1:]

    avg_knn = D.mean(axis=1)

    most_compact_idx = avg_knn.argmin()
    neighbours = I[most_compact_idx]

    center = cat_emb[most_compact_idx]
    radius = np.sqrt(D[most_compact_idx, -1])

    distances = np.linalg.norm(cat_emb - center, axis=1)

    covered_mask = distances <= radius
    covered_idx = np.where(covered_mask)[0]


    for i, growth in enumerate([1.00, 1.025, 1.05, 1.10]):
        ax = axes[i]

        j = 1
        while True:
            print("RUN: ", j)
            centroid = cat_emb[covered_idx].mean(axis=0)

            radius = np.linalg.norm(cat_emb[covered_idx] - centroid, axis=1).max()
            radius *= growth

            distances = np.linalg.norm(cat_emb - centroid, axis=1)
            new_covered_idx = np.where(distances <= radius)[0]

            if np.array_equal(np.sort(new_covered_idx), np.sort(covered_idx)):
                break

            covered_idx = new_covered_idx
            j += 1

        centroid = cat_emb[covered_idx].mean(axis=0)
        radius = np.linalg.norm(cat_emb[covered_idx] - centroid, axis=1).max()

        centroid_2d = pca.transform(centroid.reshape(1, -1))[0]
        center_2d = pca_cat[most_compact_idx]

        neighbours_2d = pca_cat[neighbours]
        covered_2d = pca_cat[covered_idx]

        radius_2d = np.linalg.norm(covered_2d - centroid_2d, axis=1).max()

        ax.scatter(pca_cat[:, 0], pca_cat[:, 1], alpha=0.4, label="Uncovered Embeddings")
        ax.scatter(covered_2d[:, 0], covered_2d[:, 1], alpha=0.8, color="green", label="New Covered Embeddings")
        ax.scatter(neighbours_2d[:, 0], neighbours_2d[:, 1], alpha=0.9, color="orange", label="Old Covered Embeddings")
        ax.scatter(center_2d[0], center_2d[1], marker="x", s=120, color="red", label="Old Centre")
        ax.scatter(centroid_2d[0], centroid_2d[1], marker="o", s=120, color="pink", label="New Centre")

        circle = Circle(
            centroid_2d,
            radius_2d,
            fill=False,
            linewidth=2
        )

        ax.add_patch(circle)
        ax.set_title(f"Growth={growth}")
        ax.set_aspect("equal", adjustable="box")

        print("K:", K)

        print("Neighbours Shape: ", neighbours_2d.shape)
        print("Covered Shape: ", covered_2d.shape)

        print("Growth: ", growth)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.3725, -0.1)
    )

    plt.tight_layout()
    plt.savefig(f"{img_dir}/{masks.category}.svg", bbox_inches="tight")
    plt.show()

def plot_sphere_cover(masks, spheres, output_dir : Path, cache_dir : Path):
    img_dir = output_dir / "sphere_plot"
    os.makedirs(img_dir, exist_ok=True)

    pca_2d, pca = load_or_compute(cache_dir / "cls_pca.npy", cache_dir / "pca.joblib", compute_pca)
    pca_train = pca_2d[masks.train_mask]
    pca_cat = pca_train[masks.train_category_mask]

    fig, ax = plt.subplots(figsize=(18, 8))

    ax.scatter(pca_cat[:, 0], pca_cat[:, 1], alpha=0.4)

    for i, sphere in enumerate(spheres):
        centroid = sphere.center
        covered_idx = sphere.covered_idx

        centroid_2d = pca.transform(centroid.reshape(1, -1))[0]
        covered_2d = pca_cat[covered_idx]

        radius_2d = np.linalg.norm(covered_2d - centroid_2d, axis=1).max()

        ax.scatter(covered_2d[:, 0], covered_2d[:, 1], alpha=0.8, color="green", label="Covered Embeddings" if i == 0 else None)
        ax.scatter(centroid_2d[0], centroid_2d[1], marker="o", s=120, color="pink", label="Sphere Centres" if i == 0 else None)

        circle = Circle(
            centroid_2d,
            radius_2d,
            fill=False,
            linewidth=2
        )

        ax.add_patch(circle)

    ax.set_title("Cloud of Spheres plot")
    ax.set_aspect("equal", adjustable="box")

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.48, 0.88)
    )

    plt.savefig(f"{img_dir}/{masks.category}.svg")
    plt.show()