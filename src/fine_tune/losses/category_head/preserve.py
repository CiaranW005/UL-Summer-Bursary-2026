import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import cast
"""
Used to preserve how much info from base model(Dino)
Based on how similar embeddings are to the old ones
"""
class PreservationLoss(nn.Module):
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
            )-> torch.Tensor:
        org_embeds = org_embeds.detach()

        org_norm = cast(torch.Tensor,
            org_embeds.norm(dim=-1, keepdim=True) # pyright: ignore[reportUnknownMemberType]
        ).clamp_min(self.eps)

        proj_norm = cast(torch.Tensor,
            proj_embeds.norm(dim=-1, keepdim=True) # pyright: ignore[reportUnknownMemberType]
        ).clamp_min(self.eps)
    
        org_unit = org_embeds / org_norm
        proj_unit = proj_embeds / proj_norm

        org_sim = org_unit @ org_unit.T
        proj_sim = proj_unit @ proj_unit.T

        similarity_loss = F.mse_loss(proj_sim, org_sim)

        norm_ratio = proj_norm / org_norm

        norm_loss = F.mse_loss(norm_ratio, torch.ones_like(norm_ratio))

        return (
            self.sim_weight * similarity_loss
            + self.norm_weight * norm_loss
        )
