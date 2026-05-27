import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from data import get_dataloaders
from system import CommSystem


# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, device, config, train: bool):
    model.train(train)
    total_loss   = 0.0
    total_correct = 0
    total_symbols = 0

    with torch.set_grad_enabled(train):
        for messages in loader:
            messages = messages.to(device)          # (B, 4)

            logits = model(messages)                # (B, 4, 8)

            # Cross-entropy over all 4 × B symbol predictions
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                messages.view(-1),
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            preds = logits.argmax(dim=-1)           # (B, 4)
            total_correct  += (preds == messages).sum().item()
            total_symbols  += messages.numel()
            total_loss     += loss.item()

    avg_loss = total_loss / len(loader)
    ser      = 1.0 - total_correct / total_symbols   # Symbol Error Rate
    return avg_loss, ser


# ─────────────────────────────────────────────────────────────────────────────
def train(config: Config | None = None):
    if config is None:
        config = Config()

    torch.manual_seed(config.seed)
    device = config.get_device()
    print(f"Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_dl, val_dl = get_dataloaders(config)

    # ── Model ────────────────────────────────────────────────────────────────
    model = CommSystem(config).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # ── Optimiser + Scheduler ────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_ser = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_ser": [], "val_ser": []}

    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_ser = run_epoch(model, train_dl, optimizer, device, config, train=True)
        va_loss, va_ser = run_epoch(model, val_dl,   optimizer, device, config, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_ser"].append(tr_ser)
        history["val_ser"].append(va_ser)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:4d}/{config.epochs} | "
                f"Train loss {tr_loss:.4f}  SER {tr_ser:.4f} | "
                f"Val loss {va_loss:.4f}  SER {va_ser:.4f} | "
                f"LR {lr_now:.2e}"
            )

        # Save best checkpoint
        if va_ser < best_val_ser:
            best_val_ser = va_ser
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_ser": va_ser, "config": config},
                config.save_path,
            )

    print(f"\nBest val SER: {best_val_ser:.4f}  (checkpoint → {config.save_path})")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
