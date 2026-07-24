import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoryContLoss(nn.Module):
    def __init__(self, temperature = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings,
        labels
    ):
        embeddings = F.normalize(embeddings, dim=-1)

        similarities = embeddings @ embeddings.T
        logits = similarities / self.temperature

        batch_size = embeddings.shape[0]

        self_mask = torch.eye(
            batch_size,
            dtype=torch.bool,
            device=embeddings.device
        )

        positive_mask = labels[:, None] == labels[None, :]
        positive_mask = positive_mask & ~self_mask
        
        negative_mask = labels[:, None] != labels[None, :]

        with torch.no_grad():
            same_sim = similarities[positive_mask].mean()
            diff_sim = similarities[negative_mask].mean()

            print(
                f"same={same_sim.item():.4f}, "
                f"different={diff_sim.item():.4f}, "
                f"gap={(same_sim - diff_sim).item():.4f}"
            )
        logits = logits.masked_fill(self_mask, float("-inf"))

        log_prob = logits - torch.logsumexp(
            logits,
            dim=1,
            keepdim=True
        )

        positive_count = positive_mask.sum(dim=1)

        valid = positive_count > 0

        mean_pos_log_prob = (
            log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)
            / positive_count.clamp_min(1)
        )

        if not valid.any():
            return embeddings.sum() * 0.0
        
        return -mean_pos_log_prob[valid].mean()
    