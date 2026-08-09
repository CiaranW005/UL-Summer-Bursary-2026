import torch

import numpy as np

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def cluster_metrics(z: torch.Tensor | np.ndarray, categories: torch.Tensor | np.ndarray)-> dict:
    if isinstance(z, torch.Tensor):
        z = z.cpu().numpy()

    if isinstance(categories, torch.Tensor):
        categories = categories.cpu().numpy()

    sil = silhouette_score(z, categories)
    db = davies_bouldin_score(z, categories)
    ch = calinski_harabasz_score(z, categories)

    unique_labels = np.unique(categories)

    intra_distances = []
    centroids = []

    for label in unique_labels:
        points = z[categories == label]

        centroid = points.mean(axis=0)
        centroids.append(centroid)

        distances = np.linalg.norm(
            points - centroid,
            axis=1
        )

        intra_distances.extend(distances)

    centroids = np.asarray(centroids)
    intra_distance = np.mean(intra_distances)

    # compares each centroid against every other centroid(Must remove duplication)
    diff = centroids[:, None, :] - centroids[None, :, : ]
    centroid_distances = np.linalg.norm(diff, axis=1)

    upper = np.triu_indices(len(centroids), k=1)
    inter_distance = centroid_distances[upper].mean()

    return {
        "silhouette": float(sil),
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
        "intra_distance": float(intra_distance),
        "inter_distance": float(inter_distance),
    }


