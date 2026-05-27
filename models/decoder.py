import torch
import torch.nn as nn
from .transformer import TransformerStack
from config import Config


class RXDecoder(nn.Module):
    """
    Runs ONCE after all T rounds (Hint 2).

    Input:
      • received : (B, T, 4)  — all noisy signals y(1)…y(T)

    For each symbol position i the decoder sees the vector
      [y_i(1), y_i(2), …, y_i(T)]  ∈  R^T
    giving 4 tokens of dimension T.

    Architecture:
        per-position raw  →  pre-MLP  →  TransformerStack  →  classifier
                                                                  ↓
                                                       logits (B, 4, vocab_size)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # ── Pre-MLP: T received values per position → d_model ───────────────
        self.pre_mlp = nn.Sequential(
            nn.Linear(config.T, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        # ── Transformer ──────────────────────────────────────────────────────
        self.transformer = TransformerStack(
            d_model  = config.d_model,
            n_heads  = config.n_heads,
            d_ff     = config.d_ff,
            n_layers = config.n_layers,
            seq_len  = config.seq_len,
            dropout  = config.dropout,
        )

        # ── Classification head: d_model → vocab_size ────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.vocab_size),
        )

    def forward(self, received: torch.Tensor) -> torch.Tensor:
        """
        received : (B, T, 4)
        returns  : (B, 4, vocab_size)  — logits for each symbol position
        """
        # Rearrange to (B, 4, T): one token per symbol position
        y = received.permute(0, 2, 1)          # (B, 4, T)

        # Pre-MLP  →  (B, 4, d_model)
        Z = self.pre_mlp(y)

        # Transformer  →  (B, 4, d_model)
        H = self.transformer(Z)

        # Classification  →  (B, 4, vocab_size)
        logits = self.classifier(H)

        return logits
