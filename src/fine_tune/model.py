import torch
from torch import nn

class ProjectionHead(nn.Module):
    def __init__(
            self, 
            dim: int = 384, 
            hidden_dim: int = 768, 
            norm_type : str | None = None,
            dropout: float = 0.0,
            residual: bool = False
        ) -> None:
        super().__init__()

        self.residual = residual

        self.norm_type = norm_type
        if norm_type == "batch":
            norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layer":
            norm = nn.LayerNorm(hidden_dim)
        elif norm_type is None:
            norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            norm,
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return x + self.net(x)
        return self.net(x)
    