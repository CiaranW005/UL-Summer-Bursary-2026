import torch
from torch.utils.data import DataLoader

import numpy as np

from pathlib import Path
from typing import cast

from .types import DinoModel

from .extract_embs import get_layer_embeddings, get_embeddings, get_adapter_block_embeddings
from .build_faiss import build_index

from ..model_selection.types import EmbeddingPipeline, PipelineStage 

def get_stage_embedding_dir(
    embed_dir: Path,
    stage: PipelineStage,
) -> Path:
    if stage.path is None:
        return embed_dir / stage.name / "pretrained"

    return embed_dir / stage.name / stage.path.parent.name

def save_dino_layers(
    dino_stage: PipelineStage,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    embed_dir: Path,
) -> None:
    dino = cast(DinoModel, dino_stage.model)
    layer_indices = list(range(len(dino.blocks)))

    stage_dir = get_stage_embedding_dir(embed_dir=embed_dir, stage=dino_stage)
    stage_dir = stage_dir / "layers"
    stage_dir.mkdir(parents=True, exist_ok=True)

    embeddings_by_layer = get_layer_embeddings(
        dino=dino,
        loader=loader,
        device=device,
        layers=layer_indices,
    )

    for layer_index, embeddings in embeddings_by_layer.items():
        output_path = stage_dir / f"layer_{layer_index + 1:02d}_cls.pt"

        torch.save(embeddings, output_path)

        print(
            f"Saved layer {layer_index + 1:02d}: "
            f"{tuple(embeddings.shape)} -> {output_path}"
        )

def save_adapter_block_embeddings(
    pipeline: EmbeddingPipeline,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    embed_dir: Path,
) -> None:
    cls_embeds = get_adapter_block_embeddings(
        pipeline=pipeline, 
        loader=loader, 
        device=device
    )

    stage_dir = get_stage_embedding_dir(
        embed_dir=embed_dir,
        stage=pipeline.dino_stage
    )
    stage_dir.mkdir(parents=True, exist_ok=True)

    cls_path = stage_dir / "cls.pt"

    if not cls_path.exists():
        torch.save(cls_embeds, cls_path)

def save_standard_embeddings(
    pipeline: EmbeddingPipeline,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    embed_dir: Path,
) -> None:

    cls_embeds, patches = get_embeddings(
        pipeline=pipeline,
        loader=loader,
        device=device,
    )

    embed_dir.mkdir(parents=True, exist_ok=True)
    for stage in pipeline:
        stage_dir = get_stage_embedding_dir(
            embed_dir=embed_dir,
            stage=stage
        )

        stage_dir.mkdir(parents=True, exist_ok=True)

        cls_path = stage_dir / "cls.pt"

        if not cls_path.exists():
            torch.save(cls_embeds[stage.name], cls_path)

        # Patches come directly from DINO, so save them only once.
        if stage.name == "dino":
            patches_path = stage_dir / "patches.pt"

            if not patches_path.exists():
                torch.save(patches, patches_path)

    build_index(embs=cls_embeds[pipeline[-1].name])
