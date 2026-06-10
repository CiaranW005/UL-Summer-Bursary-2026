import os
import torch

from ..config import MODEL_DIR, EMBEDS_DIR, FAISS_DIR

from .create_metadata import create_metadata
from .init_db import build_metadata_db
from .preprocess import preprocess
from .extract_embs import get_embeddings
from .build_faiss import build_index

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EMBEDS_DIR, exist_ok=True)
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

    embeddings, indices = get_embeddings(model=model, loader=loader, device=device)

    print(f"Embeddings Shape: {embeddings.shape}")
    torch.save(embeddings, EMBEDS_DIR / "raw_embeds.pt")

    build_index(embs=embeddings)