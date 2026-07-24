import torch
import torch.nn as nn

from .supervisedcontloss import AnomalyCategoryContLoss

from ...types import EmbeddingBatch

class CombinedLoss(nn.Module):
    def __init__(
            self, 
            enabled: dict[str, bool],
            weights: dict[str, float],
            negatives: list[torch.Tensor],
            negative_labels: list[int]
        ):
        super().__init__()

        self.losses = nn.ModuleDict()
        self.weights = weights

        self.negatives = negatives
        self.negative_labels = negative_labels

        if enabled.get("cont_loss", False):
            self.losses["cont_loss"] = AnomalyCategoryContLoss(temperature=0.1)

        if len(self.losses) == 0:
            raise ValueError("At least one loss must be enabled.")

    def forward(self, batch: EmbeddingBatch) -> tuple[torch.Tensor, dict[str, int]]:
        total_loss = batch.proj_view1.new_tensor(0.0)
        components: dict[str, int] = {}

        if "cont_loss" in self.losses:
            projected = torch.cat([batch.proj_view1, batch.proj_view2], dim=0)
            categories = torch.cat([batch.categories, batch.categories], dim=0)

            # TODO: Return cosines for logging
            category_loss, _ = self.losses["cont_loss"](projected, categories, batch.negatives, self.negative_labels)

            total_loss = total_loss + self.weights.get("category_cont", 1.0) * category_loss

            components["cont_loss"] = category_loss.detach().item()
            

        return total_loss, components
