import argparse
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
    total_loss    = 0.0
    total_correct = 0
    total_symbols = 0

    with torch.set_grad_enabled(train):
        for messages in loader:
            messages = messages.to(device)          # (B, 4)
            logits   = model(messages)              # (B, 4, 8)

            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                messages.view(-1),
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            preds          = logits.argmax(dim=-1)
            total_correct  += (preds == messages).sum().item()
            total_symbols  += messages.numel()
            total_loss     += loss.item()

    return total_loss / len(loader), 1.0 - total_correct / total_symbols


# ─────────────────────────────────────────────────────────────────────────────
def train(config: Config):
    torch.manual_seed(config.seed)
    device = config.get_device()
    print(f"Device     : {device}")

    train_dl, val_dl = get_dataloaders(config)

    model = CommSystem(config).to(device)
    print(f"Parameters : {model.count_parameters():,}")
    print(f"T={config.T}  sigma2={config.sigma2}  d_model={config.d_model}\n")

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)

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
            print(
                f"Epoch {epoch:4d}/{config.epochs} | "
                f"Train loss {tr_loss:.4f}  SER {tr_ser:.4f} | "
                f"Val loss {va_loss:.4f}  SER {va_ser:.4f} | "
                f"LR {scheduler.get_last_lr()[0]:.2e}"
            )

        if va_ser < best_val_ser:
            best_val_ser = va_ser
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_ser": va_ser, "config": config, "history": history},
                config.save_path,
            )

    print(f"\nBest val SER : {best_val_ser:.4f}  →  {config.save_path}")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train interactive AWGN comm system")
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--d_model",     type=int,   default=None)
    p.add_argument("--n_layers",    type=int,   default=None)
    p.add_argument("--T",           type=int,   default=None)
    p.add_argument("--sigma2",      type=float, default=None)
    p.add_argument("--save_path",   type=str,   default=None)
    p.add_argument("--seed",        type=int,   default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args   = parse_args()
    config = Config()

    # Override config with any CLI args that were passed
    for key, val in vars(args).items():
        if val is not None:
            setattr(config, key, val)

    train(config)
