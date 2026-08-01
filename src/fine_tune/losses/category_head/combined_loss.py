import torch
import torch.nn as nn

from .category_loss import CategoryContLoss
from .preserve import PreservationLoss

from ...types import EmbeddingBatch

class CombinedLoss(nn.Module):
    def __init__(
            self, 
            enabled: dict[str, bool],
            weights: dict[str, float]
        ):
        super().__init__()

        self.losses = nn.ModuleDict()
        self.weights = weights

        if enabled.get("category_cont", False):
            self.losses["category_cont"] = CategoryContLoss(temperature=0.1)

        if enabled.get("preservation", False):
            self.losses["preservation"] = PreservationLoss(sim_weight=1.0, norm_weight=1.0, eps=1e-8)

        if len(self.losses) == 0:
            raise ValueError("At least one loss must be enabled.")

    def forward(self, batch: EmbeddingBatch) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = batch.proj_view1.new_tensor(0.0)
        components: dict[str, float]= {}

        if "category_cont" in self.losses:
            projected = torch.cat([batch.proj_view1, batch.proj_view2], dim=0)
            categories = torch.cat([batch.categories, batch.categories], dim=0)

            # TODO: return cosines to log
            category_loss, cosines = self.losses["category_cont"](projected, categories)

            total_loss = total_loss + self.weights.get("category_cont", 1.0) * category_loss

            components["category_cont"] = category_loss.detach().item()
            
        if "preservation" in self.losses:
            preserve_z1 = self.losses["preservation"](batch.org_view1, batch.proj_view1)
            preserve_z2 = self.losses["preservation"](batch.org_view2, batch.proj_view2)

            preserve_loss = 0.5 * (preserve_z1 + preserve_z2)

            total_loss = total_loss + self.weights.get("preservation", 1.0) * preserve_loss
            components["preservation"] = preserve_loss.detach().item()
        
        components["total"] = total_loss.detach().item()

        return total_loss, components
