import torch
import torch.nn.functional as F

from ..types import BaseLoss

class VICRegLoss(BaseLoss):
    def __init__(
            self,
            invar_weight: float = 25.0,
            var_weight: float = 25.0,
            cov_weight: float = 1.0,
            var_target = 1.0,
            eps: float = 1e-4
    )-> None:
        super().__init__()

        self.invar_weight = invar_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

        self.var_target = var_target
        self.eps = eps

    def forward(
            self,
            z1: torch.Tensor,
            z2: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        invar_loss = F.mse_loss(z1, z2)

        var_loss = self._variance_loss(z1) + self._variance_loss(z2)
        cov_loss = self._covaraince_loss(z1) + self._covaraince_loss(z2)

        loss = (
            self.invar_weight * invar_loss +
            self.var_weight * var_loss +
            self.cov_weight * cov_loss
        )

        metrics = {}
        if self.collect_metrics:
            metrics = {
                "vicreg/invariance": invar_loss.item(),
                "vicreg/variance": var_loss.item(),
                "vicreg/covraince": cov_loss.item()
            }

        return loss, metrics

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0, unbiased=True) + self.eps)

        return F.relu(self.var_target - std).mean()

    def _covaraince_loss(self, z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape

        diff = z - z.mean(dim=0)

        cov = (diff.T @ diff) / (n - 1)

        off_diag = cov.flatten()[:-1].view(
            d - 1, d + 1
        )[:, 1:].flatten()

        return off_diag.square().sum() / d