import torch
import torch.nn as nn

from typing import TypedDict, Protocol, Self
from dataclasses import dataclass, field
from collections.abc import Sequence, Mapping

class ImageRecord(TypedDict):
    path : str
    category: str
    split: str
    type: str
    label: int

@dataclass
class Embeddings:
    base_cls: list[torch.Tensor] = field(default_factory=lambda: [])
    projected_cls: list[torch.Tensor] = field(default_factory=lambda: [])
    patches: list[torch.Tensor] = field(default_factory=lambda: [])

class DinoFeatures(TypedDict):
    x_norm_clstoken : torch.Tensor
    x_norm_patchtokens: torch.Tensor

class DinoModel(Protocol):
    blocks: Sequence[nn.Module]

    def to(self, device: torch.device) -> Self:
        ...

    def eval(self) -> Self:
        ...

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> DinoFeatures:
        ...

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False
    ) -> object:
        ...
    
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        ...
