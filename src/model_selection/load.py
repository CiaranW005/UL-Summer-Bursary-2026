from pathlib import Path
from typing import cast, Any

import torch

from src.data.types import DinoModel
from src.fine_tune.model import ProjectionHead

from .types import EmbeddingPipeline, PipelineStage

def load_checkpoint(
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    return torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

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
    checkpoint: dict[str, Any] | None,
    device: torch.device,
) -> ProjectionHead:
    print("Loading projection head:", checkpoint_path)

    if checkpoint is None:
        checkpoint = load_checkpoint(
            checkpoint_path,
            device=device,
        )

    parameters = checkpoint["parameters"]

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
    model_dir: Path | None,
    device: torch.device
) -> EmbeddingPipeline:

    if model_dir is None:
        return EmbeddingPipeline(
            stages=[PipelineStage(
                name="dino",
                model=load_dino(None, device=device),
                path=None
            )]
        )

    model_path = model_dir / "model.pt"

    if "dino_fine_tune" in model_path.parts:
        return EmbeddingPipeline(
            stages=[PipelineStage(
                name="dino",
                model=load_dino(checkpoint_path=model_path, device=device),
                path=model_path
            )]
        )

    is_category_head = "category_head" in model_path.parts
    is_anomaly_head = "anomaly_head" in model_path.parts

    if not is_category_head and not is_anomaly_head:
        raise ValueError(
            f"Unknown model checkpoint location: {model_path}"
        )

    checkpoint = load_checkpoint(model_path, device=device)

    parent_models = checkpoint["parent_models"]
    dino_path = parent_models.get("dino")

    pipeline = EmbeddingPipeline(
        stages=[PipelineStage(
            name="dino",
            model=load_dino(checkpoint_path=dino_path, device=device),
            path=Path(dino_path)
        )]
    )

    if is_anomaly_head:
        category_head_path = parent_models.get("category_head")

        # When the anomaly head is trained using fine tuned dino
        if category_head_path is not None:
            category_head_path = Path(category_head_path)

            pipeline.stages.append(PipelineStage(
                name="category_head",
                model=load_projection_head(
                    checkpoint_path=category_head_path,
                    checkpoint=None,
                    device=device
                ),
                path=category_head_path
            ))

    pipeline.stages.append(PipelineStage(
        name= "category_head" if is_category_head else "anomaly_head",
        model=load_projection_head(
            checkpoint_path=model_path,
            checkpoint=checkpoint,
            device=device
        ),
        path=model_path
    ))

    return pipeline