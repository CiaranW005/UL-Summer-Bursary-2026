#python -m src.data.main

import os
import torch
import torch.nn as nn

from typing import Literal
from pathlib import Path
import numpy as np

from ..config.paths import MODELS, EMBEDS_DIR, FAISS_DIR
from ..fine_tune.model import ProjectionHead

from .create_metadata import create_metadata
from .init_db import build_metadata_db
from .preprocess import preprocess
from .extract_embs import get_embeddings
from .build_faiss import build_index

EmbeddingMode = Literal[
    "dino",
    "category_head",
    "anomaly_head"
]

EMBEDDING_MODE: EmbeddingMode = "anomaly_head"

def get_latest_checkpoint(dir: Path) -> Path:
    if not dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {dir}"
        )
    
    checkpoints = [
        path for path in dir.iterdir()
        if path.is_file()
    ]

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {dir}"
        )

    return max(
        checkpoints,
        key=lambda path: path.stat().st_mtime
    )

def load_projection_head(
        checkpoint_dir: Path,
        device : torch.device,
    ) -> nn.Module:
    model_path = get_latest_checkpoint(checkpoint_dir)

    print("Loading Checkpoint:", model_path)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    parameters = checkpoint["model_info"]["parameters"]

    model = ProjectionHead(
        dim=parameters["model_dim"],
        hidden_dim=parameters["hidden_dim"],
        norm_type=parameters["model_normaliser"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model

def load_embedding_model(
        mode : EmbeddingMode,
        device: torch.device
) -> list[nn.Module]:
    if mode == "dino":
        return []

    if mode == "category_head":
        return [load_projection_head(
            checkpoint_dir=MODELS / "category_head",
            device=device
        )]

    if mode == "anomaly_head":
        category_head = load_projection_head(
            checkpoint_dir=MODELS / "category_head",
            device=device
        )

        anomaly_head  = load_projection_head(
            checkpoint_dir=MODELS / "anomaly_head",
            device=device
        )

        return [category_head, anomaly_head]

    raise ValueError(
        f"Unknown embedding mode: {mode}"
    )


def output_directory(
    mode: EmbeddingMode,
) -> Path:
    return EMBEDS_DIR / f"{mode}_embeds"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = output_directory(EMBEDDING_MODE)

    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)

    dino = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14"
        )
    
    dino.to(device)
    dino.eval()

    embedding_models = load_embedding_model(
        mode=EMBEDDING_MODE,
        device=device
    )

    create_metadata()
    build_metadata_db()

    loader = preprocess()

    print(f"Images: {len(loader.dataset)}")

    base_cls, projected_cls, patches = get_embeddings(dino=dino, models=embedding_models, loader=loader, device=device)

    print(f"cls_tokens Shape: {projected_cls.shape}")
    print(f"Patches Shape: {patches.shape}")

    torch.save(projected_cls, output_dir / "cls.pt")
    torch.save(patches, output_dir / "patch.pt")

    build_index(embs=projected_cls)

    correction = projected_cls - base_cls

    relative_change = (
        np.linalg.norm(correction, axis=1)
        / np.clip(np.linalg.norm(base_cls, axis=1), 1e-8, None)
    )

    cosine_similarity = (
        np.sum(base_cls * projected_cls, axis=1)
        / (
            np.linalg.norm(base_cls, axis=1)
            * np.linalg.norm(projected_cls, axis=1)
            + 1e-8
        )
    )

    print("Mean relative change:", relative_change.mean())
    print("Median relative change:", np.median(relative_change))
    print("Mean cosine similarity:", cosine_similarity.mean())

    base_norm = np.linalg.norm(base_cls, axis=1, keepdims=True)
    projected_norm = np.linalg.norm(projected_cls, axis=1, keepdims=True)

    norm_ratio = projected_norm / np.clip(base_norm, 1e-8, None)

    print("Mean projected/base norm ratio:", norm_ratio.mean())
    print("Median projected/base norm ratio:", np.median(norm_ratio))

    projected_rescaled = projected_cls * (
    base_norm / np.clip(projected_norm, 1e-8, None)
    )

    directional_change = (
        np.linalg.norm(projected_rescaled - base_cls, axis=1)
        / np.clip(base_norm.squeeze(1), 1e-8, None)
    )

    print("Mean directional-only change:", directional_change.mean())
    print("Median directional-only change:", np.median(directional_change))