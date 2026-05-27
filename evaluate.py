import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import Config
from data import MessageDataset
from system import CommSystem
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: CommSystem, config: Config, n_test: int = 10_000):
    """
    Returns a dict with:
      ser           – overall Symbol Error Rate
      per_pos_ser   – SER per symbol position (length 4)
      confusion     – (8, 8) confusion matrix (row=true, col=pred)
    """
    device = config.get_device()
    model.eval()

    ds  = MessageDataset(n_test, config)
    dl  = DataLoader(ds, batch_size=512, shuffle=False)

    all_true, all_pred = [], []

    for messages in dl:
        messages = messages.to(device)
        preds    = model.predict(messages)
        all_true.append(messages.cpu())
        all_pred.append(preds.cpu())

    true = torch.cat(all_true)   # (N, 4)
    pred = torch.cat(all_pred)   # (N, 4)

    ser         = (true != pred).float().mean().item()
    per_pos_ser = (true != pred).float().mean(dim=0).tolist()   # list of 4

    # Confusion matrix (aggregate over all positions)
    conf = torch.zeros(config.vocab_size, config.vocab_size, dtype=torch.long)
    for t, p in zip(true.view(-1).tolist(), pred.view(-1).tolist()):
        conf[t][p] += 1

    return {"ser": ser, "per_pos_ser": per_pos_ser, "confusion": conf.numpy()}


# ─────────────────────────────────────────────────────────────────────────────
def plot_results(history: dict, eval_results: dict, save_path: str = "results.png"):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    epochs = range(1, len(history["train_loss"]) + 1)

    # 1 — Training & validation loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title("Cross-entropy Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # 2 — Symbol Error Rate curves
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(epochs, history["train_ser"], label="Train SER")
    ax2.semilogy(epochs, history["val_ser"],   label="Val SER")
    ax2.set_title("Symbol Error Rate (log scale)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("SER")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # 3 — Per-position SER bar chart
    ax3 = fig.add_subplot(gs[0, 2])
    pos_labels = [f"Pos {i+1}" for i in range(4)]
    ax3.bar(pos_labels, eval_results["per_pos_ser"], color="steelblue")
    ax3.set_title("SER per Symbol Position")
    ax3.set_ylabel("SER"); ax3.set_ylim(0, max(eval_results["per_pos_ser"]) * 1.3 + 1e-6)
    ax3.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(eval_results["per_pos_ser"]):
        ax3.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

    # 4 — Confusion matrix
    ax4 = fig.add_subplot(gs[1, :2])
    conf = eval_results["confusion"].astype(float)
    conf_norm = conf / conf.sum(axis=1, keepdims=True).clip(min=1)
    im = ax4.imshow(conf_norm, cmap="Blues", vmin=0, vmax=1)
    ax4.set_title("Normalised Confusion Matrix")
    ax4.set_xlabel("Predicted symbol")
    ax4.set_ylabel("True symbol")
    ticks = list(range(8))
    ax4.set_xticks(ticks); ax4.set_xticklabels([str(i+1) for i in ticks])
    ax4.set_yticks(ticks); ax4.set_yticklabels([str(i+1) for i in ticks])
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    # 5 — Summary text box
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    summary = (
        f"Final Results\n"
        f"─────────────────\n"
        f"Overall SER : {eval_results['ser']:.4f}\n\n"
        f"Per-position SER\n"
        + "\n".join(
            f"  Position {i+1} : {v:.4f}"
            for i, v in enumerate(eval_results["per_pos_ser"])
        )
        + f"\n\nBest val SER  : {min(history['val_ser']):.4f}"
        + f"\n(epoch {np.argmin(history['val_ser'])+1})"
    )
    ax5.text(
        0.1, 0.95, summary,
        transform=ax5.transAxes, fontsize=11,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.suptitle(
        f"Interactive AWGN Communication  |  T={4}, σ²=0.25",
        fontsize=13, fontweight="bold"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, config: Config) -> CommSystem:
    device = config.get_device()
    ckpt   = torch.load(checkpoint_path, map_location=device)
    model  = CommSystem(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  (val SER={ckpt['val_ser']:.4f})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config  = Config()
    model   = load_model(config.save_path, config)
    results = evaluate(model, config)
    print(f"Test SER: {results['ser']:.4f}")
    print(f"Per-pos : {[f'{s:.4f}' for s in results['per_pos_ser']]}")
