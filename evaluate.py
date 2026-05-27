import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from config import Config
from data import MessageDataset
from system import CommSystem


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: CommSystem, config: Config, n_test: int = 10_000):
    device = config.get_device()
    model.eval()

    ds = MessageDataset(n_test, config)
    dl = DataLoader(ds, batch_size=512, shuffle=False)

    all_true, all_pred = [], []
    for messages in dl:
        messages = messages.to(device)
        preds    = model.predict(messages)
        all_true.append(messages.cpu())
        all_pred.append(preds.cpu())

    true = torch.cat(all_true)   # (N, 4)
    pred = torch.cat(all_pred)

    ser         = (true != pred).float().mean().item()
    per_pos_ser = (true != pred).float().mean(dim=0).tolist()

    conf = torch.zeros(config.vocab_size, config.vocab_size, dtype=torch.long)
    for t, p in zip(true.view(-1).tolist(), pred.view(-1).tolist()):
        conf[t][p] += 1

    return {"ser": ser, "per_pos_ser": per_pos_ser, "confusion": conf.numpy()}


# ─────────────────────────────────────────────────────────────────────────────
def plot_results(history: dict, eval_results: dict, config: Config, save_path: str = "results.png"):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title("Cross-entropy Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(epochs, history["train_ser"], label="Train SER")
    ax2.semilogy(epochs, history["val_ser"],   label="Val SER")
    ax2.set_title("Symbol Error Rate (log scale)"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("SER")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar([f"Pos {i+1}" for i in range(4)], eval_results["per_pos_ser"], color="steelblue")
    ax3.set_title("SER per Symbol Position"); ax3.set_ylabel("SER")
    ax3.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(eval_results["per_pos_ser"]):
        ax3.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

    ax4 = fig.add_subplot(gs[1, :2])
    conf_norm = eval_results["confusion"].astype(float)
    conf_norm = conf_norm / conf_norm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax4.imshow(conf_norm, cmap="Blues", vmin=0, vmax=1)
    ax4.set_title("Normalised Confusion Matrix")
    ax4.set_xlabel("Predicted symbol"); ax4.set_ylabel("True symbol")
    ticks = list(range(8))
    ax4.set_xticks(ticks); ax4.set_xticklabels([str(i+1) for i in ticks])
    ax4.set_yticks(ticks); ax4.set_yticklabels([str(i+1) for i in ticks])
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(gs[1, 2]); ax5.axis("off")
    summary = (
        f"Final Results\n─────────────────\n"
        f"Overall SER : {eval_results['ser']:.4f}\n\n"
        f"Per-position SER\n"
        + "\n".join(f"  Position {i+1} : {v:.4f}" for i, v in enumerate(eval_results["per_pos_ser"]))
        + f"\n\nBest val SER : {min(history['val_ser']):.4f}"
        + f"\n(epoch {np.argmin(history['val_ser'])+1})"
    )
    ax5.text(0.1, 0.95, summary, transform=ax5.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.suptitle(f"Interactive AWGN  |  T={config.T}, σ²={config.sigma2}", fontsize=13, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, config: Config) -> CommSystem:
    device = config.get_device()
    ckpt   = torch.load(checkpoint_path, map_location=device)
    model  = CommSystem(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint  epoch={ckpt['epoch']}  val_SER={ckpt['val_ser']:.4f}")
    return model, ckpt.get("history", {})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to .pt file (default: config.save_path)")
    p.add_argument("--n_test",     type=int, default=10_000)
    p.add_argument("--plot",       type=str, default="results.png")
    args = p.parse_args()

    config = Config()
    ckpt_path = args.checkpoint or config.save_path

    model, history = load_model(ckpt_path, config)
    results = evaluate(model, config, n_test=args.n_test)

    print(f"\nTest SER  : {results['ser']:.4f}")
    print(f"Per-pos   : {[f'{s:.4f}' for s in results['per_pos_ser']]}")

    if history:
        plot_results(history, results, config, save_path=args.plot)
