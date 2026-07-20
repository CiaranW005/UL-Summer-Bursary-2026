#python -m src.data.main

import os
import torch

import numpy as np

from ..config.paths import MODEL_DIR, EMBEDS_DIR, FAISS_DIR
from ..fine_tune.model import ProjectionHead

from .create_metadata import create_metadata
from .init_db import build_metadata_db
from .preprocess import preprocess
from .extract_embs import get_embeddings
from .build_faiss import build_index

if __name__ == "__main__":
    base_dir = EMBEDS_DIR / "cont_embeds"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    dino = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14"
        )
    
    dino.to(device)

    if any(MODEL_DIR.iterdir()):
        model_path = max(MODEL_DIR.iterdir(), key=lambda p: p.stat().st_mtime)
        print(model_path)
        
        checkpoint = torch.load(model_path, weights_only=True)
        parameters = checkpoint["model_info"]["parameters"]

        model = ProjectionHead(dim=parameters["model_dim"], hidden_dim=parameters["hidden_dim"], norm_type=parameters["model_normaliser"])
        model.load_state_dict(checkpoint["model_state_dict"])

        model.to(device)
    else:
        model = None

    create_metadata()
    build_metadata_db()
    loader = preprocess()

    print(f"Images: {len(loader.dataset)}")

    base_cls, projected_cls, patches = get_embeddings(dino=dino, model=model, loader=loader, device=device)

    print(f"cls_tokens Shape: {projected_cls.shape}")
    print(f"Patches Shape: {patches.shape}")

    # torch.save(projected_cls, base_dir / "cls.pt")
    # torch.save(patches, base_dir / "patch.pt")

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