import torch.nn as nn
from enum import StrEnum

class LossName(StrEnum):
    CATEGORY_CONTRASTIVE = "category_contrastive"
    PRESERVATION = "preservation"
    VICREG = "vicreg"
    ANOMALY_CATEGORY_CONTRASTIVE = "anomaly_category_contrastive"

class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.collect_metrics: bool = False
        
class WeightedLoss(nn.Module):
    def __init__(self, name: str, loss: BaseLoss, weight: float = 1.0) -> None:
        super().__init__()

        self.name = name
        self.loss = loss
        self.weight = weight

    def forward(self, *args, **kwargs):
        loss, metrics = self.loss(*args, **kwargs)

        return self.weight * loss, metrics

    @property
    def collect_metrics(self) -> bool:
        return self.loss.collect_metrics

    @collect_metrics.setter
    def collect_metrics(self, value: bool) -> None:
        self.loss.collect_metrics = value

class LossCollection:
    def __init__(self, losses: list[WeightedLoss]) -> None:
        self._losses = {
            loss.name : loss
            for loss in losses
        }
        self._collect_metrics: bool = False

    def __contains__(self, name: str) -> bool:
        return name in self._losses

    def __getitem__(self, name: str) -> WeightedLoss:
        return self._losses[name]
    
    def __iter__(self):
        return iter(self._losses.values())

    def __len__(self):
        return len(self._losses)

    @property
    def collect_metrics(self) -> bool:
        return self._collect_metrics

    @collect_metrics.setter
    def collect_metrics(self, value: bool) -> None:
        self._collect_metrics = value

        for loss in self._losses.values():
            loss.collect_metrics = value

    def to_dict(self) -> dict[str, float]:
        return {
            name : loss.weight
            for name, loss in self._losses.items()
        }
    