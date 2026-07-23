import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import plotly.graph_objects as go

import faiss

from collections.abc import Sequence
from typing import cast

from pathlib import Path
from .dim_reduction.utils import *
from .utils import sphere_surface

from ..algorithims.types import CategoryMasks, Hypersphere

# TODO: Refactor to stop repeating a lot of the code
def plot_ellipsoid(
        cls_tokens: np.ndarray,
        masks: CategoryMasks,     
        output_dir: Path, 
        cache_dir: Path 
        ):
    
    img_dir = output_dir / "mahlanobis_fit" 
    os.makedirs(img_dir, exist_ok=True)

    pca_2d, _ = load_or_compute(cache_dir / "cls_pca.npy", cache_dir / "pca.joblib", compute_pca, cls_tokens)
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

def plot_K(
        masks: CategoryMasks, 
        cls_tokens: np.ndarray, 
        output_dir: Path, 
        cache_dir: Path
    ):
    img_dir = output_dir / "k_rate"
    os.makedirs(img_dir, exist_ok=True)

    train_emb = cls_tokens[masks.train_mask]
    cat_emb = train_emb[masks.train_category_mask]
    
    pca_2d, _ = load_or_compute(cache_dir / "cls_pca.npy", cache_dir / "pca.joblib", compute_pca, cls_tokens)
    pca_train = pca_2d[masks.train_mask]
    pca_cat = pca_train[masks.train_category_mask]

    print("Cat length: ", len(cat_emb))

    fig, axes = plt.subplots(1, 3, figsize=(10, 6))

    for i, frac in enumerate([0.025, 0.05, 0.10]):
        k = max(2, int(frac * len(cat_emb)))

        index = faiss.IndexFlatL2(cat_emb.shape[1])
        index.add(cat_emb.astype("float32"))

        dists, nbrs = index.search(
            cat_emb.astype("float32"),
            k=k+1 # Nearest is itself
        )

        dists = dists[:, 1:]    # Remove self
        nbrs = nbrs[:, 1:]

        avg_knn = dists.mean(axis=1)

        most_compact_idx = avg_knn.argmin()
        neighbours = nbrs[most_compact_idx]

        center_2d = cast(tuple[float, float], pca_cat[most_compact_idx])

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
        ax.set_title(f"K={k} ({frac*100:.1f}%)")
        ax.set_aspect("equal", adjustable="box")

        print("frac:", frac)
        print("K:", k)
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

def plot_growth_rates(
        masks: CategoryMasks, 
        cls_tokens: np.ndarray, 
        output_dir: Path, 
        cache_dir: Path
        ):
    
    img_dir = output_dir/ "growth_rate"
    os.makedirs(img_dir, exist_ok=True)

    train_emb = cls_tokens[masks.train_mask]
    cat_emb = train_emb[masks.train_category_mask]

    pca_2d, pca = load_or_compute(cache_dir / "cls_pca.npy", cache_dir / "pca.joblib", compute_pca, cls_tokens)
    pca_train = pca_2d[masks.train_mask]
    pca_cat = pca_train[masks.train_category_mask]

    print("Cat length: ", len(cat_emb))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    K = max(2, int(0.05 * len(cat_emb)))

    index = faiss.IndexFlatL2(cat_emb.shape[1])
    index.add(cat_emb.astype("float32"))

    dists, nbrs = index.search(
        cat_emb.astype("float32"),
        k=K+1 # Nearest is itself
    )

    dists = dists[:, 1:]    # Remove self
    nbrs = nbrs[:, 1:]

    avg_knn = dists.mean(axis=1)

    most_compact_idx = avg_knn.argmin()
    neighbours = nbrs[most_compact_idx]

    center = cat_emb[most_compact_idx]
    radius = np.sqrt(dists[most_compact_idx, -1])

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

def plot_sphere_cover(
        cls_tokens: np.ndarray, 
        masks: CategoryMasks, 
        spheres: Sequence[Hypersphere], 
        output_dir : Path, 
        cache_dir : Path
        ):
    img_dir = output_dir / "sphere_plot"
    os.makedirs(img_dir, exist_ok=True)

    pca_2d, pca = load_or_compute(cache_dir / "cls_pca.npy", cache_dir / "pca.joblib", compute_pca, cls_tokens)
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


def plot_3d_sphere_cover(
        cls_tokens: np.ndarray, 
        masks: CategoryMasks, 
        spheres: Sequence[Hypersphere], 
        output_dir : Path, 
        cache_dir : Path
        ):
    rng = np.random.default_rng(42)

    pca_3d, _ = load_or_compute(cache_dir / "cls_pca_3d.npy", cache_dir / "pca_3d.joblib", compute_pca_3d, cls_tokens)
    pca_train = pca_3d[masks.train_mask]
    pca_cat = pca_train[masks.train_category_mask]

    colours = [
        f"rgb({r},{g},{b})"
        for r, g, b in rng.integers(30, 255, size=(len(spheres), 3))
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=pca_cat[:, 0],
        y=pca_cat[:, 1],
        z=pca_cat[:, 2],
        mode="markers",
        marker=dict(size=3, opacity=0.35),
        name="all screw train"
    ))

    for i, s in enumerate(spheres):
        idx = s.covered_idx
        pts = pca_cat[idx]

        center_3d = pts.mean(axis=0)
        radius_3d = np.linalg.norm(pts - center_3d).max()

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=4, color=colours[i]),
            name=f"sphere {i+1}",
            legendgroup=f"sphere{i}"
        ))

        fig.add_trace(go.Scatter3d(
            x=[center_3d[0]],
            y=[center_3d[1]],
            z=[center_3d[2]],
            mode="markers",
            marker=dict(size=7, symbol="x", color=colours[i]),
            name=f"center {i+1}",
            showlegend=False,
            legendgroup=f"sphere{i}"
        ))

        xs, ys, zs = sphere_surface(center_3d, radius_3d, resolution=20)

        fig.add_trace(go.Surface(
            x=xs,
            y=ys,
            z=zs,
            opacity=0.12,
            showscale=False,
            name=f"Boundary {i+1}",
            legendgroup=f"sphere{i}",
            showlegend=False
        ))

    fig.update_layout(
        title="3D PCA multi-sphere assignment view",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3"
        ),
        width=900,
        height=750,
        updatemenus=[
        dict(
            type="buttons",
            buttons=[
                dict(
                    label="Show All",
                    method="update",
                    args=[{"visible": [True] * len(fig.data)}]
                ),
                dict(
                    label="Hide All",
                    method="update",
                    args=[{"visible": ["legendonly"] * len(fig.data)}]
                )
            ],
            x=1.15,
            y=0
        )
        ]
    )

    fig.show()
    fig.write_image(output_dir / "3d_hypersphere_plot.svg")