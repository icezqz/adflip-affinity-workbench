import torch
import torch.nn as nn


class LightAttention(nn.Module):
    """
    Light Attention pooling over residues.
    Interface:
      - x: (B, D, N)
      - mask: (B, N) boolean or 0/1 where True/1 means valid residue
      - returns pooled: (B, D)
    """

    def __init__(
        self,
        embeddings_dim: int,
        kernel_size: int = 9,
        conv_dropout: float = 0.25,
    ):
        super().__init__()
        self.feature_convolution = nn.Conv1d(
            embeddings_dim,
            embeddings_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.attention_convolution = nn.Conv1d(
            embeddings_dim,
            embeddings_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype != torch.bool:
            mask = mask > 0
        m = mask.unsqueeze(1)  # (B,1,N)

        o = self.feature_convolution(x)
        o = self.dropout(o)

        attn = self.attention_convolution(x)
        attn = attn.masked_fill(~m, -1e9)
        weights = self.softmax(attn)  # (B,D,N)
        pooled = torch.sum(o * weights, dim=-1)  # (B,D)
        return pooled


class AffinityHead(nn.Module):
    def __init__(self, dim, hidden_dims=(256, 128), use_lightattn=True,
                 lightattn_kernel_size=9, lightattn_dropout=0.25):
        super().__init__()
        self.use_lightattn = use_lightattn
        self.pool = LightAttention(dim, lightattn_kernel_size, lightattn_dropout) if use_lightattn else None

        sizes = [dim, *hidden_dims, 1]
        layers = []
        for i, (in_d, out_d) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(in_d, out_d))
            if i != len(sizes) - 2:   # 不是最后一层才加 ReLU
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, residue_x, mask):
        if self.use_lightattn:
            pooled = self.pool(residue_x.transpose(1, 2), mask)  # (B,D)
        else:
            m = mask.float().unsqueeze(-1) if mask.dtype == torch.bool else mask.unsqueeze(-1)
            pooled = (residue_x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.mlp(pooled)  # (B,1)


