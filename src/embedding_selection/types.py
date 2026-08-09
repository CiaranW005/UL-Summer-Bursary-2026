import torch

import numpy as np

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PipelineStage:
    name: str
    path: Path

@dataclass
class EmbeddingPipeline:
    stages: list[PipelineStage]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("No embedding stages are in the pipeline")

    def __iter__(self) -> Iterator[tuple[PipelineStage, np.ndarray]]:
        for stage in self.stages:
            yield stage, torch.load(stage.path / "cls.pt", weights_only=False)

    def __len__(self):
        return len(self.stages)

    def __getitem__(self, idx: int) -> PipelineStage:
        return self.stages[idx]

    def get(self, name: str) -> PipelineStage | None:
        return next((stage for stage in self.stages if stage.name == name), None)