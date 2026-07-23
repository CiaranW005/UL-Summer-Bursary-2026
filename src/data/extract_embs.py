import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from dataclasses import dataclass, field

from typing import TypedDict, cast

import numpy as np

@dataclass
class Embeddings:
    base_cls: list[torch.Tensor] = field(default_factory=lambda: [])
    projected_cls: list[torch.Tensor] = field(default_factory=lambda: [])
    patches: list[torch.Tensor] = field(default_factory=lambda: [])

class DinoFeatures(TypedDict):
    x_norm_clstoken : torch.Tensor
    x_norm_patchtokens: torch.Tensor

def get_embeddings(
        dino: nn.Module, 
        loader: DataLoader[torch.Tensor], 
        device: torch.device, 
        models: list[nn.Module]
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    for model in models:
        model.eval()

    embeds = Embeddings()

    with torch.no_grad():
        for images in tqdm(loader):
            images = images.to(device)
            
            features = cast(DinoFeatures, dino.forward_features(images)) # type: ignore[attr-defined] 
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