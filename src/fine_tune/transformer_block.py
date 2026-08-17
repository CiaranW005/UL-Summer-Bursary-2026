from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from typing import TYPE_CHECKING
from .models.dino import Block, Attention, Mlp

if TYPE_CHECKING:
    from ..model_selection.types import DinoModel

class AnomalyDinoBlock(nn.Module):
    """
    A single DINOv2-style transformer block used as a trainable adapter.

    The frozen DINO backbone produces token embeddings which are refined by
    this additional block before the CLS token is used for anomaly learning.
    """
    def __init__(
            self,
            dim: int = 384,
            num_heads: int = 6,
            mlp_ratio: float = 4.0, #ratio between hidden dim and input dim of MLP 384 * 4 = 1536
            drop: float = 0.0,  # dropout
            drop_path: float = 0.0 # attention branch dropout (stochastic depth)
    )-> None:
        super().__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,  # give each linear layer a learnable bias (Query, Key, Value)
            proj_bias=True, #  bias for the output projection after self-attention
            ffn_bias=True, # each MLP layer has a learnable bias
            drop=drop,
            attn_drop=drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            act_layer=nn.GELU,
            attn_class=Attention, # memory-efficient attention implementation
            ffn_layer=Mlp,  # Meta's feed-forward network implementation
            init_values=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DinoBlockExtension(nn.Module):
    def __init__(
            self,
            dino: DinoModel,
            use_outer_residual: bool = False,

            dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.dino = dino
        self.residual = use_outer_residual

        self.adapter = AnomalyDinoBlock(drop=dropout) # FIXME: Initialize with appropriate parameters so its configureable
        self.norm = nn.LayerNorm(384, eps=1e-6) # FIXME: Use those initalised paramter for the correct dimension even though it will more than likely stay as 384

    def train(self, mode: bool = True)-> "DinoBlockExtension":
        super().train(mode)

        self.adapter.train(mode)

        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.dino.forward_features(images)
            tokens = features["x_prenorm"]

        adapted_tokens = self.adapter(tokens)
        adapted_tokens = self.norm(adapted_tokens)

        dino_cls = tokens[:, 0]
        cls = adapted_tokens[:, 0]

        if self.residual:
            return dino_cls + cls
        return cls 