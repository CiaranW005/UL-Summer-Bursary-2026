import torch 
from dataclasses import dataclass

type Sample = tuple[
    torch.Tensor,
    torch.Tensor,
    int,
    int | None
]

@dataclass
class EmbeddingBatch:
    proj_view1: torch.Tensor
    proj_view2: torch.Tensor

    categories: torch.Tensor

    org_view1: torch.Tensor | None = None
    org_view2: torch.Tensor | None = None

    negatives : torch.Tensor | None = None
