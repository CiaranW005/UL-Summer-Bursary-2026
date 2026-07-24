import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from pathlib import Path
from collections.abc import Sequence

from .types import DinoModel

from .extract_embs import get_layer_embeddings, get_embeddings
from .build_faiss import build_index

def save_dino_layers(
    dino: DinoModel,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    output_dir: Path,
) -> None:
    layer_indices = list(range(len(dino.blocks)))

    embeddings_by_layer = get_layer_embeddings(
        dino=dino,
        loader=loader,
        device=device,
        layers=layer_indices,
    )

    for layer_index, embeddings in embeddings_by_layer.items():
        output_path = output_dir / f"layer_{layer_index + 1:02d}_cls.pt"

        torch.save(embeddings, output_path)

        print(
            f"Saved layer {layer_index + 1:02d}: "
            f"{tuple(embeddings.shape)} -> {output_path}"
        )

def save_standard_embeddings(
    dino: DinoModel,
    models: Sequence[nn.Module],
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    output_dir: Path,
) -> None:
    base_cls, projected_cls, patches = get_embeddings(
        dino=dino,
        models=models,
        loader=loader,
        device=device,
    )

    print(f"CLS tokens shape: {projected_cls.shape}")
    print(f"Patches shape: {patches.shape}")

    torch.save(projected_cls, output_dir / "cls.pt")
    torch.save(patches, output_dir / "patch.pt")

    build_index(embs=projected_cls)

    print_embedding_changes(
        base_cls=base_cls,
        projected_cls=projected_cls,
    )

def print_embedding_changes(
    base_cls: np.ndarray,
    projected_cls: np.ndarray,
) -> None:
    correction = projected_cls - base_cls

    base_norm = np.linalg.norm(
        base_cls,
        axis=1,
        keepdims=True,
    )

    projected_norm = np.linalg.norm(
        projected_cls,
        axis=1,
        keepdims=True,
    )

    relative_change = (
        np.linalg.norm(correction, axis=1)
        / np.clip(base_norm.squeeze(1), 1e-8, None)
    )

    cosine_similarity = (
        np.sum(base_cls * projected_cls, axis=1)
        / (
            base_norm.squeeze(1)
            * projected_norm.squeeze(1)
            + 1e-8
        )
    )

    norm_ratio = (projected_norm / np.clip(base_norm, 1e-8, None))

    projected_rescaled = projected_cls * (base_norm/ np.clip(projected_norm, 1e-8, None))

    directional_change = (
        np.linalg.norm(
            projected_rescaled - base_cls,
            axis=1,
        )
        / np.clip(base_norm.squeeze(1), 1e-8, None)
    )

    print("Mean relative change:", relative_change.mean())
    print("Median relative change:", np.median(relative_change))
    print("Mean cosine similarity:", cosine_similarity.mean())

    print("Mean projected/base norm ratio:", norm_ratio.mean())
    print("Median projected/base norm ratio:", np.median(norm_ratio))

    print("Mean directional-only change:", directional_change.mean())
    print("Median directional-only change:", np.median(directional_change))