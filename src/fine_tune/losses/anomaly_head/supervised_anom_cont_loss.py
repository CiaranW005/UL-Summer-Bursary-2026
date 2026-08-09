import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import cast

from ..types import BaseLoss

class CategoryAnomalyContrastiveLoss(BaseLoss):
    """
    Contrastive loss that learns category-specific normal embeddings while
    treating matched anomalies as negatives.

    Normal samples from the same category are pulled together. Normal samples
    from different categories and anomalies belonging to the corresponding
    category contribute to the denominator, encouraging the model to separate
    anomalous representations and further separate normal representations.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, 
                norm_embeds: torch.Tensor, 
                norm_labels: torch.Tensor,
                anom_embeds: torch.Tensor, 
                anom_labels: torch.Tensor
            )-> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        norm_embeds = F.normalize(norm_embeds, dim=-1)
        anom_embeds = F.normalize(anom_embeds, dim=-1)

        # Pairwise cosine similarities between normal samples and between 
        # normal and anomalous samples
        norm_sim = norm_embeds @ norm_embeds.T
        anom_sim = norm_embeds @ anom_embeds.T

        batch_size = norm_embeds.shape[0]

        self_mask = torch.eye(
            batch_size,
            dtype=torch.bool,
            device=norm_embeds.device
        )

        # Positive pairs are normal samples from the same category, excluding self-comparisons
        positive_mask = (
            norm_labels[:, None] == norm_labels[None, :]
        ) & ~self_mask

        # Normal samples belonging to different categories
        other_category_mask = (
            norm_labels[:, None] != norm_labels[None, :]
        )

        # Anomalies belonging to the same category as each normal sample
        matched_anomaly = (
            norm_labels[:, None] == anom_labels[None, :]
        ) 

        norm_logits = norm_sim / self.temperature
        anom_logits = anom_sim / self.temperature

        # Exclude self-comparisons and unrelated anomaly categories from the contrastive denominator.
        norm_logits = norm_logits.masked_fill(self_mask, float("-inf"))
        anom_logits = anom_logits.masked_fill(~matched_anomaly, float("-inf"))

        all_logits = torch.cat(
            [norm_logits, anom_logits],
            dim=1
        )

        # Compute the InfoNCE normalised term over normal samples and matched anomalies
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
            unique, counts = cast(torch.Tensor, torch.unique(norm_labels, return_counts=True)) 

            raise RuntimeError(
                "No positive pairs found.\n"
                f"Batch size: {len(norm_labels)}\n"
                f"Unique labels: {unique.tolist()}\n" 
                f"Counts: {counts.tolist()}"
            )

        # Avergae the log-probability across all positive pairs for each anchor
        mean_pos_log_prob = (
            positive_log_prob.sum(dim=1) / positive_count.clamp_min(1)
        )

        # Log cosine similarities to monitor representation during training
        cosines = {}
        if self.collect_metrics:
            with torch.no_grad():
                same_sim = norm_sim[positive_mask].mean()

                # How similar categories are to other categories
                other_normal_sim = norm_sim[
                    other_category_mask
                ].mean()

                if matched_anomaly.any():
                    matched_anomaly_sim = anom_sim[matched_anomaly].mean()
                else:
                    matched_anomaly_sim = torch.tensor(float("nan"), device=norm_embeds.device)

                cosines = {
                    "anomaly_contrastive/normal": same_sim,
                    "anomaly_contrastive/other_normal": other_normal_sim,
                    "anomaly_contrastive/matched_anomaly": matched_anomaly_sim
                }

        return -mean_pos_log_prob[valid].mean(), cosines
    