from __future__ import annotations

import torch 
import pandas as pd

from typing import TYPE_CHECKING, TypedDict, NotRequired
from dataclasses import dataclass

if TYPE_CHECKING:
    from .losses.combined_loss import CombinedLoss

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
    negative_labels: torch.Tensor | None = None

class ModelParameters(TypedDict):
    seed: int 
    num_workers: int
    pin_memory: bool

    samples_per_category: int
    batch_size: int
    epochs: int

    model_dim: int
    hidden_dim: int
    mlp_factor : NotRequired[int]
    dropout: float

    use_residual: bool
    learning_rate: float
    weight_decay: float

    model_normaliser: str
    model_name: str

class ModelInfo(TypedDict):
    timestamp: NotRequired[str]
    model_type: str
    losses: NotRequired[dict[str, float]]
    notes: list[str]
    git_commit: str
    parameters: ModelParameters
    parent_models: dict[str, str | None]
    negative_indices: NotRequired[list[int]]

@dataclass
class TrainingObjects:
    model: torch.nn.Module
    criterion: CombinedLoss
    optimizer: torch.optim.Optimizer
    device: torch.device

    negatives: torch.Tensor | None = None
    negative_labels: torch.Tensor | None = None

@dataclass
class TrainingResults:
    history: pd.DataFrame
    best_state: dict[str, torch.Tensor] | None
    best_val_loss: float
    best_epoch: int | None
    error: Exception | None = None
