import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from collections.abc import Sequence

import numpy as np

from .types import Embeddings, DinoModel
from ..model_selection.types import EmbeddingPipeline

@torch.inference_mode()
def get_embeddings(
        pipeline: EmbeddingPipeline,
        loader: DataLoader[torch.Tensor], 
        device: torch.device
        ) -> tuple[dict[str, np.ndarray], np.ndarray]:

    pipeline.eval()

    embeds = Embeddings(
        cls={stage.name: [] for stage in pipeline},
        patches=[]
    )
   
    for images in tqdm(loader):
        images = images.to(device)

        features = pipeline.dino.forward_features(images)

        current_cls = features["x_norm_clstoken"]
        patches = features["x_norm_patchtokens"]

        embeds.cls["dino"].append(current_cls.cpu())

        for stage in pipeline.heads:
            current_cls = stage.model(current_cls)
            embeds.cls[stage.name].append(current_cls.cpu())

        embeds.patches.append(patches.cpu())

    cls_embeds = {
        name : torch.cat(batches).numpy().astype("float32")
        for name, batches in embeds.cls.items()
    }

    patches = torch.cat(embeds.patches).numpy().astype("float32")
    return cls_embeds, patches

@torch.inference_mode()
def get_adapter_block_embeddings(
        pipeline: EmbeddingPipeline,
        loader: DataLoader[torch.Tensor], 
        device: torch.device
        ) -> np.ndarray:
    """
    Custom extractor for the adpater block as it require dinos pre_norm tokens
    """

    pipeline.eval()

    cls_embeds = []
    for images in tqdm(loader):
        images = images.to(device)


        cls_embeds.append(pipeline.dino(images))

    return torch.cat(cls_embeds).cpu().numpy().astype("float32")

@torch.inference_mode()
def get_layer_embeddings(
    dino: DinoModel,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    layers: Sequence[int]
)-> dict[int, torch.Tensor]:

    cls_by_layer: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in layers
    }

    for images in tqdm(loader):
        images = images.to(device)

        outputs =  dino.get_intermediate_layers(
            images,
            n=layers,
            reshape=False,
            return_class_token=True,
            norm=True
        )

        for layer, (_, cls_token) in zip(layers, outputs, strict=True):
            cls_by_layer[layer].append(cls_token.cpu())

    return {
        layer: torch.cat(batches, dim=0)
        for layer, batches in cls_by_layer.items()
    }
        