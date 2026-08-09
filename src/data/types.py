import torch
import torch.nn as nn

from typing import TypedDict
from dataclasses import dataclass
from collections.abc import Sequence
class ImageRecord(TypedDict):
    path : str
    category: str
    split: str
    type: str
    label: int

@dataclass
class Embeddings:
    cls: dict[str, list[torch.Tensor]]
    patches: list[torch.Tensor]
    
class DinoFeatures(TypedDict):
    x_norm_clstoken : torch.Tensor
    x_norm_patchtokens: torch.Tensor
    x_prenorm: torch.Tensor

class DinoModel(nn.Module):
    norm : nn.Module
    blocks: Sequence[nn.Module]

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> DinoFeatures:
        raise NotImplementedError

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        raise NotImplementedError
