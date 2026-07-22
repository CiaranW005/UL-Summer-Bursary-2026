import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyCategoryContLoss(nn.Module):
    def __init__(self, temperature = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, 
                norm_embeds, norm_labels,
                anom_embeds, anom_labels
                ):
        norm_embeds = F.normalize(norm_embeds, dim=-1)
        anom_embeds = F.normalize(anom_embeds, dim=-1)

        norm_sim = norm_embeds @ norm_embeds.T
        anom_sim = norm_embeds @ anom_embeds.T

        batch_size = norm_embeds.shape[0]

        self_mask = torch.eye(
            batch_size,
            dtype=torch.bool,
            device=norm_embeds.device
        )

        positive_mask = (
            norm_labels[:, None] == norm_labels[None, :]
        ) & ~self_mask


        other_category_mask = (
            norm_labels[:, None] != norm_labels[None, :]
        )

        negative_mask = (
            norm_labels[:, None] == anom_labels[None, :]
        ) 

        norm_logits = norm_sim / self.temperature
        anom_logits = anom_sim / self.temperature

        norm_logits = norm_logits.masked_fill(self_mask, float("-inf"))
        anom_logits = anom_logits.masked_fill(~negative_mask, float("-inf"))

        all_logits = torch.cat(
            [norm_logits, anom_logits],
            dim=1
        )

        log_denominator = torch.logsumexp(
            all_logits,
            dim=1,
            keepdim=True
        )

        positive_log_prob = (
            norm_logits - log_denominator
        ).masked_fill(~positive_mask, 0.0)

        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0

        if not valid.any():
            return norm_embeds.sum() * 0.0

        mean_pos_log_prob = (
            positive_log_prob.sum(dim=1) / positive_count.clamp_min(1)
        )

        with torch.no_grad():
            same_sim = norm_sim[positive_mask].mean()

            other_normal_sim = norm_sim[
                other_category_mask
            ].mean()

            if negative_mask.any():
                matched_anomaly_sim = anom_sim[negative_mask].mean()
            else:
                matched_anomaly_sim = torch.tensor(float("nan"), device=norm_embeds.device)

            print(
                f"same_normal={same_sim.item():.4f}, "
                f"other_normal={other_normal_sim.item():.4f}, "
                f"matched_anomaly={matched_anomaly_sim.item():.4f}"
            )

        return -mean_pos_log_prob[valid].mean()