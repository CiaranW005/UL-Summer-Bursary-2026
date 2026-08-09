from __future__ import annotations

import torch
import torch.nn as nn

from typing import TYPE_CHECKING

from .types import LossName

if TYPE_CHECKING:
    from .types import LossCollection
    from ..types import EmbeddingBatch

class CombinedLoss(nn.Module):
    def __init__(
            self, 
            losses: LossCollection
        ):
        super().__init__()

        self.losses = losses

        if len(self.losses) == 0:
            raise ValueError("At least one loss must be enabled.")

    @property
    def collect_metrics(self) -> bool:
        return self.losses.collect_metrics

    @collect_metrics.setter
    def collect_metrics(self, value: bool) -> None:
        self.losses.collect_metrics = value

    def forward(self, batch: EmbeddingBatch) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = batch.proj_view1.new_tensor(0.0)
        metrics: dict[str, float]= {}

        if (LossName.CATEGORY_CONTRASTIVE in self.losses
            and self.losses[LossName.CATEGORY_CONTRASTIVE] != 0
        ):
                projected = torch.cat([batch.proj_view1, batch.proj_view2], dim=0)
                categories = torch.cat([batch.categories, batch.categories], dim=0)

                category_loss, cosines = self.losses[LossName.CATEGORY_CONTRASTIVE](projected, categories)
                total_loss += category_loss

                metrics.update({
                    "loss/category_contrastive": category_loss.detach().item(),
                    **cosines
                })

        if (LossName.ANOMALY_CATEGORY_CONTRASTIVE in self.losses
            and self.losses[LossName.ANOMALY_CATEGORY_CONTRASTIVE] != 0
        ):
            projected = torch.cat([batch.proj_view1, batch.proj_view2], dim=0)
            categories = torch.cat([batch.categories, batch.categories], dim=0)

            anom_cat_loss, cosines = self.losses[LossName.ANOMALY_CATEGORY_CONTRASTIVE](projected, categories, batch.negatives, batch.negative_labels)
            total_loss += anom_cat_loss

            metrics.update({
                "loss/anomaly_category_contrastive": anom_cat_loss.detach().item(),
                **cosines
            })
            
        if (LossName.PRESERVATION in self.losses
            and self.losses[LossName.PRESERVATION].weight != 0
        ):
            preserve_z1, metrics_z1 = self.losses[LossName.PRESERVATION](batch.org_view1, batch.proj_view1)
            preserve_z2, metrics_z2 = self.losses[LossName.PRESERVATION](batch.org_view2, batch.proj_view2)

            preserve_loss = 0.5 * (preserve_z1 + preserve_z2)
            total_loss += preserve_loss

            metrics.update({
                "loss/preservation": preserve_loss.detach().item(),
                **{
                    f"preservation/{key}": 0.5 * (
                        metrics_z1[key] + metrics_z2[key]
                    )
                    for key in metrics_z1
                }
            })

        if (LossName.VICREG in self.losses
            and self.losses[LossName.VICREG].weight != 0
        ):
            vicreg, components = self.losses[LossName.VICREG](batch.proj_view1, batch.proj_view2)
            total_loss += vicreg

            metrics.update({
                "loss/vicreg": vicreg.detach().item(),
                **components
            })
            
        metrics["loss/total"] = total_loss.detach().item()
    
        return total_loss, metrics
