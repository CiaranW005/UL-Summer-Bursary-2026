from torch import nn

class ProjectionHead(nn.Module):
    def __init__(self, dim=384, hidden_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)