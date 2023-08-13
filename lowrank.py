import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float("Inf"))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class LowRankDense(nn.Module):
    def __init__(self, in_shape, out_shape, rank=4, initializer=None) -> None:
        super().__init__()

        self.initializer = initializer

        self.lora_a = nn.Parameter(torch.empty((in_shape, rank), dtype=torch.float32))

        self.lora_b = nn.Parameter(torch.empty((rank, out_shape), dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        if self.initializer is not None:
            self.initializer(self.lora_a)
            self.initializer(self.lora_b)
        else:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_b, a=math.sqrt(5))

    def forward(self, x):
        return x @ self.lora_a @ self.lora_b


class LowRankMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        q_rank: int = 4,
        k_rank: int = 4,
        v_rank: int = 4,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)

        self.query_proj = LowRankDense(d_model, self.d_head * num_heads, q_rank)
        self.key_proj = LowRankDense(d_model, self.d_head * num_heads, k_rank)
        self.value_proj = LowRankDense(d_model, self.d_head * num_heads, v_rank)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)

        query = (
            query.permute(2, 0, 1, 3)
            .contiguous()
            .view(batch_size * self.num_heads, -1, self.d_head)
        )
        key = (
            key.permute(2, 0, 1, 3)
            .contiguous()
            .view(batch_size * self.num_heads, -1, self.d_head)
        )
        value = (
            value.permute(2, 0, 1, 3)
            .contiguous()
            .view(batch_size * self.num_heads, -1, self.d_head)
        )

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        context = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = (
            context.permute(1, 2, 0, 3)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_head)
        )

        return context


class LowRankSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        q_rank: int = 4,
        k_rank: int = 4,
        v_rank: int = 4,
    ):
        super().__init__()
        self.model = LowRankMultiHeadAttention(
            d_model, num_heads, q_rank, k_rank, v_rank
        )
        
    def forward(self, x, mask=None):
        return self.model(x, x, x, mask)


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LowRankFeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, rank=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            LowRankDense(dim, hidden_dim, rank),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            LowRankDense(hidden_dim, dim, rank),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LowRankTransformerEncoder(torch.nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                LowRankSelfAttention(dim, num_heads=heads),
                            )
                        ),
                        Residual(
                            PreNorm(
                                dim, LowRankFeedForward(dim, mlp_dim, dropout=dropout)
                            )
                        ),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x
