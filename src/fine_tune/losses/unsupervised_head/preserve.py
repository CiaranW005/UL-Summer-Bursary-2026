import torch
import torch.nn.functional as F

from typing import cast

from ..types import BaseLoss

class PreservationLoss(BaseLoss):
    """
    Preserve information from the base representation.

    Encourages:
    - pairwise similarity structure to remain similar
    - embedding norms to remain similar

    Also reports diagnostics describing how the representation moved.
    """
    def __init__(self, 
                sim_weight: float = 1.0, 
                norm_weight :float = 1.0, 
                eps : float = 1e-8
        ):
        super().__init__()

        self.sim_weight = sim_weight
        self.norm_weight = norm_weight

        self.eps = eps

    def forward(self, 
                org_embeds: torch.Tensor, 
                proj_embeds: torch.Tensor
            )-> tuple[torch.Tensor, dict[str, float]]:
        org_embeds = org_embeds.detach()

        org_norm = cast(torch.Tensor,
            org_embeds.norm(dim=-1, keepdim=True) 
        ).clamp_min(self.eps)

        proj_norm = cast(torch.Tensor,
            proj_embeds.norm(dim=-1, keepdim=True) 
        ).clamp_min(self.eps)
    
        org_unit = org_embeds / org_norm
        proj_unit = proj_embeds / proj_norm

        org_sim = org_unit @ org_unit.T
        proj_sim = proj_unit @ proj_unit.T

        # how similar the two directions are remove magnitude
        similarity_loss = F.mse_loss(proj_sim, org_sim)

        # how muc bigger/smaller proj vs org
        norm_ratio = proj_norm / org_norm

        norm_loss = F.mse_loss(norm_ratio, torch.ones_like(norm_ratio))

        loss = (
            self.sim_weight * similarity_loss +
            self.norm_weight * norm_loss
        )

        metrics = {}
        if self.collect_metrics:
            metrics = self.get_diagnostics(
                org_embeds,
                org_norm,
                proj_embeds,
                proj_norm
            )

            metrics.update({
                "similarity_loss": similarity_loss.item(),
                "norm_loss": norm_loss.item()
            })

        return loss, metrics

    def get_diagnostics(
            self, 
            org_embeds: torch.Tensor, 
            org_norm: torch.Tensor,
            proj_embeds: torch.Tensor,
            proj_norm: torch.Tensor,
        ) -> dict[str, float]:
        with torch.no_grad():
            cosine_sim = F.cosine_similarity(
                            org_embeds,
                            proj_embeds,
                            dim=-1,
                            eps=self.eps
                        )
            
            relative_change = (
                (proj_embeds - org_embeds).norm(dim=-1) 
                / org_norm.squeeze(-1)
            )

            norm_ratio_flat = (proj_norm / org_norm).squeeze(-1)

            projected_rescaled = (
                proj_embeds * (org_norm / proj_norm)
            )

            directional_change = (
               (projected_rescaled - org_embeds).norm(dim=-1)
               / org_norm.squeeze(-1)
            )

            metrics = {
                # Original vs projected direction
                "cosine/mean": cosine_sim.mean().item(),
                "cosine/median": cosine_sim.median().item(),
                "cosine/std": cosine_sim.std().item(),

                # Total representation movement
                "relative_change/mean": relative_change.mean().item(),
                "relative_change/median": relative_change.median().item(),
                "relative_change/std": relative_change.std().item(),

                # Magnitude change
                "norm_ratio/mean": norm_ratio_flat.mean().item(),
                "norm_ratio/median": norm_ratio_flat.median().item(),
                "norm_ratio/std": norm_ratio_flat.std().item(),

                # Movement after removing magnitude effects
                "directional_change/mean": directional_change.mean().item(),
                "directional_change/median": directional_change.median().item(),
                "directional_change/std": directional_change.std().item(),
            }

            return metrics


            

