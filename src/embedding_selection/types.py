import torch

import json
import numpy as np

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class PipelineStage:
    name: str
    model_path: Path | None
    embed_path: Path

@dataclass
class EmbeddingPipeline:
    stages: list[PipelineStage]
    neg_indices: np.ndarray = field(init=False, default_factory=lambda: np.empty(0, dtype=np.int64))

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("No embedding stages are in the pipeline")

        neg_set: set[int] = set()
        for stage in self.stages:
            if stage.model_path is not None:
                with open(stage.model_path / "metadata.json", "r") as f:
                    meta = json.load(f)

                neg_set.update(meta.get("negative_indices", []))

        self.neg_indices = np.asarray(list(neg_set), dtype=np.int64)

    def __iter__(self) -> Iterator[tuple[PipelineStage, np.ndarray]]:
        for stage in self.stages:
            cls = torch.load(stage.embed_path / "cls.pt", weights_only=False)

            keep = torch.ones(len(cls), dtype=torch.bool)
            keep[self.neg_indices] = False

            yield stage, cls[keep]

    def __len__(self):
        return len(self.stages)

    def __getitem__(self, idx: int) -> PipelineStage:
        return self.stages[idx]

    def get(self, name: str) -> PipelineStage | None:
        return next((stage for stage in self.stages if stage.name == name), None)