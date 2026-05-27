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
      • round_idx    : int           — current round index (0-indexed)

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

        # ── Round embedding: which round t ∈ {0,...,T-1} are we in? ─────────
        # Critical: without this the encoder cannot distinguish rounds
        self.round_embed = nn.Embedding(config.T, config.d_emb)

        # ── Pre-MLP: maps raw token → d_model ───────────────────────────────
        # raw dim = d_emb (symbol) + d_emb (round) + (T-1) tx hist + (T-1) fb hist
        raw_dim = 2 * config.d_emb + 2 * (config.T - 1)
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
        round_idx:  int = 0,        # current round (0-indexed)
    ) -> torch.Tensor:              # (B, 4)         float

        B = messages.size(0)

        # 1. Symbol embedding  →  (B, 4, d_emb)
        sym = self.symbol_embed(messages)

        # 2. Round embedding  →  (B, 4, d_emb)
        #    Same round for all positions, broadcast across seq_len
        r_idx = torch.tensor(round_idx, device=messages.device)
        rnd   = self.round_embed(r_idx).unsqueeze(0).unsqueeze(0)  # (1, 1, d_emb)
        rnd   = rnd.expand(B, self.config.seq_len, -1)             # (B, 4, d_emb)

        # 3. Concatenate everything  →  (B, 4, raw_dim)
        raw = torch.cat([sym, rnd, tx_history, fb_history], dim=-1)

        # 4. Pre-MLP  →  Z(t) ∈ (B, 4, d_model)
        Z = self.pre_mlp(raw)

        # 5. Transformer  →  (B, 4, d_model)
        H = self.transformer(Z)

        # 6. Post-MLP  →  (B, 4, 1)  →  (B, 4)
        x = self.post_mlp(H).squeeze(-1)

        # 7. Per-sample power normalisation: ||x||² = 1  →  E[||x(t)||²] = 1 ✓
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x = x / norm

        return x