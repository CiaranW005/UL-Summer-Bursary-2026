#python -m src.data.main

import os
import torch

from typing import Literal
from pathlib import Path

import json

from ..config.paths import MODELS, EMBEDS_DIR, FAISS_DIR, DB_PATH
from ..model_selection import get_checkpoints, load_inference_models, EmbeddingPipeline

from .create_metadata import create_metadata
from .init_db import build_metadata_db
from .preprocess import preprocess
from .save_embeds import save_dino_layers, save_standard_embeddings, save_adapter_block_embeddings

EmbeddingMode = Literal[
    "dino",
    "dino_layers",
    "dino_fine_tune",
    "dino_adapter_block",
    "category_head",
    "anomaly_head"
]

def select_checkpoint(
        checkpoints: list[Path],
        *,
        model_name: str,
) -> Path | None:
    if not checkpoints:
        raise FileNotFoundError(
            f"No {model_name} checkpoints were found"
        )

    print(f"\nSelect {model_name}:")

    option_num = 1

    for i, checkpoint in enumerate(checkpoints, start=option_num):
        with open(checkpoint / "metadata.json", "r") as f:
            MODEL_INFO = json.load(f)

        notes = MODEL_INFO["notes"]
        
        print(f"{i}: {checkpoint.name}")
        for j, note in enumerate(notes):
            label = "Notes:" if j == 0 else ""
            print(f"\t{label:<12}{note}")

    max_option = len(checkpoints)

    while True:
        raw_selection = input(f"Selection [1-{max_option}]: ")

        try:
            selection = int(raw_selection)
        except ValueError:
            print("Enter a valid number")
            continue

        if not 1 <= selection <= max_option:
            print(f"Enter a number from 1 to {max_option}")
            continue


        checkpoint_idx = selection - 1
        return checkpoints[checkpoint_idx]

def select_pipeline(
        mode: EmbeddingMode,
        device: torch.device
) -> EmbeddingPipeline:
    if mode in {"dino", "dino_layers"}:
      selected_path = None

    elif mode == "dino_fine_tune":
        selected_path = select_checkpoint(
            get_checkpoints(MODELS / "dino_fine_tune"),
            model_name="fine-tuned DINO",
        )

    elif mode == "dino_adapter_block":
        selected_path = select_checkpoint(
            get_checkpoints(MODELS / "dino_adapter_block"),
            model_name="DINO anomaly-adapter block",
        )

    elif mode == "category_head":
        selected_path = select_checkpoint(
             get_checkpoints(MODELS / "category_head"),
            model_name="Category Projection Head",
        )

    elif mode == "anomaly_head":
        selected_path = select_checkpoint(
            get_checkpoints(MODELS / "anomaly_head"),
            model_name="Anomaly Projection Head"
        )
    else:
        raise ValueError(f"Unsupported Embedding Mode: {mode}")
    
    return load_inference_models(model_dir=selected_path, device=device)


MODE_OPTIONS: dict[str, EmbeddingMode] = {
    "1": "dino",
    "2": "dino_layers",
    "3": "dino_fine_tune",
    "4": "dino_adapter_block",
    "5": "category_head",
    "6": "anomaly_head",
}

def select_embedding_mode() -> EmbeddingMode:
    print("\nSelect embedding mode:")
    print("1. DINO final layer")
    print("2. DINO intermediate layers")
    print("3. DINO final layer fine-tuned")
    print("4. DINO anomaly-adapter block")
    print("5. Category projection head")
    print("6. Anomaly projection head")

    while True:
        selection = input("Mode [1-6]: ")

        mode = MODE_OPTIONS.get(selection)

        if mode is not None:
            print(f"Selected mode: {mode}")
            return mode

        print("Invalid selection. Enter 1, 2, 3, 4, 5 or 6.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EMBEDDING_MODE = select_embedding_mode()
    PIPELINE = select_pipeline(EMBEDDING_MODE, device=device)

    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(EMBEDS_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)

    if not DB_PATH.exists():
        create_metadata()
        build_metadata_db()

    loader = preprocess()

    if EMBEDDING_MODE == "dino_layers":
        save_dino_layers(PIPELINE.dino_stage, loader, device, EMBEDS_DIR)
    elif EMBEDDING_MODE == "dino_adapter_block":
        save_adapter_block_embeddings(PIPELINE, loader, device, EMBEDS_DIR)
    else:
        save_standard_embeddings(PIPELINE, loader, device, EMBEDS_DIR)