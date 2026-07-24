#python -m src.data.main

import os
import torch
import torch.nn as nn

from typing import Literal
from pathlib import Path

from typing import cast 

from ..config.paths import MODELS, EMBEDS_DIR, FAISS_DIR
from ..fine_tune.model import ProjectionHead

from .create_metadata import create_metadata
from .init_db import build_metadata_db
from .preprocess import preprocess
from .save_embeds import save_dino_layers, save_standard_embeddings

from .types import DinoModel

EmbeddingMode = Literal[
    "dino",
    "dino_layers",
    "category_head",
    "anomaly_head"
]

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

    if mode == "dino_layers":
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

MODE_OPTIONS: dict[str, EmbeddingMode] = {
    "1": "dino",
    "2": "dino_layers",
    "3": "category_head",
    "4": "anomaly_head",
}

def select_embedding_mode() -> EmbeddingMode:
    print("\nSelect embedding mode:")
    print("1. DINO final layer")
    print("2. DINO intermediate layers")
    print("3. Category projection head")
    print("4. Anomaly projection head")

    while True:
        selection = input("Mode [1-4]: ").strip()

        mode = MODE_OPTIONS.get(selection)

        if mode is not None:
            print(f"Selected mode: {mode}")
            return mode

        print("Invalid selection. Enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EMBEDDING_MODE = select_embedding_mode()
    output_dir = output_directory(EMBEDDING_MODE)

    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)

    dino = cast(DinoModel, torch.hub.load( # pyright: ignore[reportUnknownMemberType]
            repo_or_dir="facebookresearch/dinov2",
            model="dinov2_vits14"
        ))
    
    dino.to(device)
    dino.eval()

    projection_models = load_embedding_model(
        mode=EMBEDDING_MODE,
        device=device
    )

    create_metadata()
    build_metadata_db()

    loader = preprocess()

    if EMBEDDING_MODE == "dino_layers":
        save_dino_layers(dino, loader, device, output_dir)
    else:
        save_standard_embeddings(dino, projection_models, loader, device, output_dir)