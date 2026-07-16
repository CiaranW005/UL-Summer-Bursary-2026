import torch.nn as nn
import torch.nn.functional as F

"""
Used to preserve how much info from base model(Dino)
Based on how similar embeddings are to the old ones
"""
class PreservationLoss(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, org_embeds, proj_embeds):
        org_sim = F.normalize(org_embeds, dim=-1)
        org_sim = org_sim @ org_sim.T

        proj_sim = F.normalize(proj_embeds, dim=-1)
        proj_sim = proj_sim @ proj_sim.T

        return F.mse_loss(
            proj_sim,
            org_sim.detach()
        )


    