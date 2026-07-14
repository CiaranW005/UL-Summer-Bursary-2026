#python -m src.data.main

import os
import torch

from ..config.paths import MODEL_DIR, EMBEDS_DIR, FAISS_DIR

from .create_metadata import create_metadata
from .init_db import build_metadata_db
from .preprocess import preprocess
from .extract_embs import get_embeddings
from .build_faiss import build_index

if __name__ == "__main__":
    base_dir = EMBEDS_DIR / "base_embeds"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    if any(MODEL_DIR.iterdir()):
        model_path = max(MODEL_DIR.iterdir(), key=lambda p: p.stat().st_mtime)

        model = torch.load(model_path)
    else:
        model = torch.hub.load(
                    "facebookresearch/dinov2",
                    "dinov2_vits14"
                )

    model = model.to(device)

    create_metadata()
    build_metadata_db()
    loader = preprocess()

    print(f"Images: {len(loader.dataset)}")

    cls_tokens, patches = get_embeddings(model=model, loader=loader, device=device)

    print(f"cls_tokens Shape: {cls_tokens.shape}")
    print(f"Patches Shape: {patches.shape}")

    torch.save(cls_tokens, base_dir / "cls.pt")
    torch.save(patches, base_dir / "patch.pt")

    build_index(embs=cls_tokens)