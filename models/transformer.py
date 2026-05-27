import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Multi-head self-attention
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k     = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B,H,L,dk)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)             # (B,H,L,L)
        attn   = self.dropout(torch.softmax(scores, dim=-1))
        out    = (attn @ V).transpose(1, 2).contiguous().view(B, L, D)       # (B,L,D)
        return self.W_o(out)


# ─────────────────────────────────────────────────────────────────────────────
# Position-wise feed-forward network
# ─────────────────────────────────────────────────────────────────────────────
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Single transformer block — post-norm as given in the problem:
#   H^(l) = LayerNorm( H^(l-1) + MultiHead(H^(l-1)) )
#   H^(l) = LayerNorm( H^(l)   + FFN(H^(l))          )
# ─────────────────────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn   = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))   # Eq. (2)
        x = self.norm2(x + self.ffn(x))    # Eq. (3)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Stack of transformer blocks with learned positional encoding
# ─────────────────────────────────────────────────────────────────────────────
class TransformerStack(nn.Module):
    """
    H^(0) = Z + PE                      (Eq. 1)
    H^(l) = TransformerBlock(H^(l-1))   (Eqs. 2-3)
    """
    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        d_ff:     int,
        n_layers: int,
        seq_len:  int,
        dropout:  float = 0.1,
    ):
        super().__init__()
        # Learned positional encoding — shape (1, seq_len, d_model)
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: (B, seq_len, d_model)
        x = Z + self.pe          # H^(0)
        for block in self.blocks:
            x = block(x)
        return x                 # (B, seq_len, d_model)
