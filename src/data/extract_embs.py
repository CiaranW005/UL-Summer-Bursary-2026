import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from dataclasses import dataclass, field

@dataclass
class Embeddings:
    base_cls: list[torch.Tensor] = field(default_factory=list)
    projected_cls: list[torch.Tensor] = field(default_factory=list)
    patches: list[torch.Tensor] = field(default_factory=list)

def get_embeddings(
        dino: nn.Module, 
        loader: DataLoader, 
        device: torch.device, 
        models: list[nn.Module]):

    for model in models:
        model.eval()

    embeds = Embeddings()

    with torch.no_grad():
        for images in tqdm(loader):
            images = images.to(device)
            
            features = dino.forward_features(images)
            base_cls = features["x_norm_clstoken"]
            patches = features["x_norm_patchtokens"]

            projected_cls = base_cls
            for model in models:
                projected_cls = model(projected_cls)

            embeds.base_cls.append(base_cls.cpu())
            embeds.projected_cls.append(projected_cls.cpu())
            embeds.patches.append(patches.cpu())
    
    base_cls = torch.cat(embeds.base_cls).numpy().astype("float32")
    projected_cls = torch.cat(embeds.projected_cls).numpy().astype("float32")

    patches = torch.cat(embeds.patches).numpy().astype("float32")
    
    return base_cls, projected_cls, patches