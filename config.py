from dataclasses import dataclass
import torch


@dataclass
class Config:
    # ── Message ──────────────────────────────────────────────
    vocab_size: int = 8          # alphabet {1,...,8}, we use 0-indexed internally
    seq_len:    int = 4          # 4 symbols per message

    # ── Channel ──────────────────────────────────────────────
    T:      int   = 4            # communication rounds
    sigma2: float = 0.25         # AWGN noise variance  →  σ = 0.5

    # ── Model ────────────────────────────────────────────────
    d_model:  int   = 64
    n_heads:  int   = 4
    n_layers: int   = 2
    d_ff:     int   = 128
    d_emb:    int   = 32         # symbol embedding dim (fed into pre-MLP)
    dropout:  float = 0.1

    # ── Training ─────────────────────────────────────────────
    batch_size: int   = 256
    lr:         float = 3e-4
    epochs:     int   = 150
    train_size: int   = 64_000
    val_size:   int   = 8_000
    grad_clip:  float = 1.0

    # ── Misc ─────────────────────────────────────────────────
    seed:       int  = 42
    device:     str  = "cuda"
    save_path:  str  = "model.pt"

    def get_device(self) -> torch.device:
        return torch.device(self.device if torch.cuda.is_available() else "cpu")
