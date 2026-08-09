import torch
import torch.nn as nn

from dataclasses import dataclass
from pathlib import Path

from typing import cast

from ..data.types import DinoModel

@dataclass
class PipelineStage:
    name: str
    model : nn.Module
    path: Path | None = None

@dataclass
class EmbeddingPipeline:
    stages: list[PipelineStage]

    def __post_init__(self) -> None:
        if self.stages and self.stages[0].name != "dino":
            raise ValueError("DINO must be the first pipeline stage")

    def __iter__(self):
        return iter(self.stages)

    def __len__(self):
        return len(self.stages)

    def __getitem__(self, idx: int) -> PipelineStage:
        return self.stages[idx]

    def get(self, name: str) -> PipelineStage | None:
        return next((stage for stage in self.stages if stage.name == name), None)

    def to(self, device: torch.device | str) -> "EmbeddingPipeline":
        for stage in self.stages:
            stage.model.to(device)
        return self

    def eval(self) -> "EmbeddingPipeline":
        for stage in self.stages:
            stage.model.eval()
        return self
    
    @property
    def dino(self) -> DinoModel:
        return cast(DinoModel, self.dino_stage.model)

    @property
    def dino_stage(self) -> PipelineStage:
        return self.stages[0]

    @property
    def heads(self) -> list[PipelineStage]:
        return self.stages[1:]
