import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from models import TXEncoder, RXDecoder


class CommSystem(nn.Module):
    """
    Full interactive communication system for T rounds.

    Forward pass:
      1.  For t = 1 … T:
            a. TX Encoder produces x(t) from (messages, tx_history, fb_history)
            b. AWGN channel:  y(t) = x(t) + ε,  ε ~ N(0, σ²I)
            c. Feedback (Hint 1):  f(t) = y(t)   [noiseless relay]
            d. Update histories for the next round
      2.  RX Decoder maps all {y(t)} → logits (B, 4, 8)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config  = config
        self.sigma   = config.sigma2 ** 0.5
        self.encoder = TXEncoder(config)
        self.decoder = RXDecoder(config)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        """
        messages : (B, 4)  int64  ∈ {0,...,7}
        returns  : (B, 4, vocab_size)  logits
        """
        B      = messages.size(0)
        T      = self.config.T
        device = messages.device

        # Running histories — always shape (B, 4, T-1), zero-padded on the right
        # We rebuild them from lists to keep the computation graph intact
        # (avoids in-place ops on tensors that need gradients)
        x_list: list[torch.Tensor] = []   # coded symbols  x(1)…x(t-1)
        y_list: list[torch.Tensor] = []   # feedback        y(1)…y(t-1)
        received_list: list[torch.Tensor] = []

        for t in range(T):

            # ── Build history tensors (B, 4, T-1) ────────────────────────────
            if t == 0:
                tx_hist = torch.zeros(B, self.config.seq_len, T - 1, device=device)
                fb_hist = torch.zeros(B, self.config.seq_len, T - 1, device=device)
            else:
                # Stack what we have so far  →  (B, 4, t)
                tx_hist = torch.stack(x_list, dim=2)
                fb_hist = torch.stack(y_list, dim=2)
                # Pad right to T-1 with zeros
                pad = T - 1 - t
                if pad > 0:
                    tx_hist = F.pad(tx_hist, (0, pad))
                    fb_hist = F.pad(fb_hist, (0, pad))

            # ── TX Encoder  →  x(t) ∈ (B, 4), ||x||²=1 ──────────────────────
            x_t = self.encoder(messages, tx_hist, fb_hist, round_idx=t)

            # ── AWGN channel  →  y(t) = x(t) + ε ────────────────────────────
            noise = torch.randn_like(x_t) * self.sigma
            y_t   = x_t + noise

            # ── Store for histories and final decoding ────────────────────────
            x_list.append(x_t)
            y_list.append(y_t)           # feedback = y(t)  (Hint 1)
            received_list.append(y_t)

        # ── Stack all received signals  →  (B, T, 4) ─────────────────────────
        received = torch.stack(received_list, dim=1)   # (B, T, 4)

        # ── RX Decoder (runs once)  →  (B, 4, vocab_size) ────────────────────
        logits = self.decoder(received)

        return logits

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, messages: torch.Tensor) -> torch.Tensor:
        """Returns hard symbol predictions  (B, 4)  int64."""
        logits = self.forward(messages)
        return logits.argmax(dim=-1)

    # ─────────────────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)