import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from collections.abc import Sequence

import numpy as np

from .types import Embeddings, DinoModel

@torch.inference_mode()
def get_embeddings(
        dino: DinoModel, 
        loader: DataLoader[torch.Tensor], 
        device: torch.device, 
        models: Sequence[nn.Module]
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    for model in models:
        model.eval()

    embeds = Embeddings()
   
    for images in tqdm(loader):
        images = images.to(device)
        
        features = dino.forward_features(images)
        batch_base = features["x_norm_clstoken"]
        batch_patches = features["x_norm_patchtokens"]

        batch_projected = batch_base
        for model in models:
            batch_projected = model(batch_projected)

        embeds.base_cls.append(batch_base.cpu())
        embeds.projected_cls.append(batch_projected.cpu())
        embeds.patches.append(batch_patches.cpu())
    
    base_cls = torch.cat(embeds.base_cls).numpy().astype("float32")
    projected_cls = torch.cat(embeds.projected_cls).numpy().astype("float32")

    patches = torch.cat(embeds.patches).numpy().astype("float32")
    
    return base_cls, projected_cls, patches

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
        