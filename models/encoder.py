import torch
import torch.nn as nn
from .transformer import TransformerStack
from config import Config


class TXEncoder(nn.Module):
    """
    At round t the encoder receives:
      • messages     : (B, 4)        — original symbols (fixed across rounds)
      • tx_history   : (B, 4, T-1)  — coded symbols sent in rounds 1…t-1  (zero-padded)
      • fb_history   : (B, 4, T-1)  — feedback y received in rounds 1…t-1 (zero-padded)

    It produces:
      • x_t          : (B, 4)        — power-normalised coded symbols

    Architecture (Hint 3):
        raw input → pre-MLP → TransformerStack → post-MLP → power normalisation
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # ── Symbol embedding  m_i ∈ {0,...,7} → R^{d_emb} ──────────────────
        self.symbol_embed = nn.Embedding(config.vocab_size, config.d_emb)

        # ── Pre-MLP: maps raw token → d_model ───────────────────────────────
        # raw dim per token = d_emb  +  (T-1) tx history  +  (T-1) fb history
        raw_dim = config.d_emb + 2 * (config.T - 1)
        self.pre_mlp = nn.Sequential(
            nn.Linear(raw_dim, config.d_model),
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

        # ── Post-MLP: penultimate latent → 1 coded symbol per position ───────
        self.post_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(
        self,
        messages:   torch.Tensor,   # (B, 4)        int64
        tx_history: torch.Tensor,   # (B, 4, T-1)   float
        fb_history: torch.Tensor,   # (B, 4, T-1)   float
    ) -> torch.Tensor:              # (B, 4)         float

        # 1. Symbol embedding  →  (B, 4, d_emb)
        sym = self.symbol_embed(messages)           # (B, 4, d_emb)

        # 2. Concatenate histories  →  (B, 4, raw_dim)
        raw = torch.cat([sym, tx_history, fb_history], dim=-1)

        # 3. Pre-MLP  →  Z(t) ∈ (B, 4, d_model)
        Z = self.pre_mlp(raw)

        # 4. Transformer  →  (B, 4, d_model)
        H = self.transformer(Z)

        # 5. Post-MLP  →  (B, 4, 1)  →  (B, 4)
        x = self.post_mlp(H).squeeze(-1)

        # 6. Per-sample power normalisation so that ||x||² = 1
        #    This guarantees E[||x(t)||²] = 1  ≤  1  ✓
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x = x / norm

        return x    # (B, 4),  ||x[i]||² = 1 for every sample i
