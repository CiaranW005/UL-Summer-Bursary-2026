from pathlib import Path
from typing import cast

import torch
import torch.nn as nn 

from src.data.types import DinoModel
from src.fine_tune.model import ProjectionHead

from .metadata import load_checkpoint

def load_dino(
    checkpoint_path: Path | None,
    device: torch.device,
) -> DinoModel:
    dino = cast(
        DinoModel,
        torch.hub.load(  # pyright: ignore[reportUnknownMemberType]
            repo_or_dir="facebookresearch/dinov2",
            model="dinov2_vits14",
        ),
    )

    if checkpoint_path is None:
        print("Using pretrained DINOv2 ViT-S/14")
    else:
        print("Loading DINO:", checkpoint_path)

        checkpoint = load_checkpoint(
            checkpoint_path,
            device=device,
        )

        dino.load_state_dict(
            checkpoint["model_state_dict"]
        )

    dino.to(device)
    dino.eval()

    for p in dino.parameters():
        p.requires_grad = False

    return dino


def load_projection_head(
    checkpoint_path: Path,
    device: torch.device,
) -> ProjectionHead:
    print("Loading projection head:", checkpoint_path)

    checkpoint = load_checkpoint(
        checkpoint_path,
        device=device,
    )

    parameters = checkpoint[
        "model_info"
    ]["parameters"]

    model = ProjectionHead(
        dim=parameters["model_dim"],
        hidden_dim=parameters["hidden_dim"],
        norm_type=parameters["model_normaliser"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model

def load_inference_models(
    dino_path: Path | None,
    category_head_path: Path | None,
    device: torch.device,
) -> list[nn.Module]:
    dino = load_dino(
        checkpoint_path=dino_path,
        device=device,
    )
    
    models: list[nn.Module] = [dino]

    if category_head_path is not None:
        category_head = load_projection_head(
            checkpoint_path=category_head_path,
            device=device,
        )

        models.append(category_head)

    return models