import argparse
import torch

from config import Config
from train import train
from evaluate import evaluate, plot_results, load_model


def parse_args():
    p = argparse.ArgumentParser(description="Interactive AWGN Communication System")
    p.add_argument("--mode",       type=str, default="train",
                   choices=["train", "eval", "all"],
                   help="train | eval | all  (default: train)")

    # Config overrides
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--d_model",    type=int,   default=None)
    p.add_argument("--n_layers",   type=int,   default=None)
    p.add_argument("--T",          type=int,   default=None)
    p.add_argument("--sigma2",     type=float, default=None)
    p.add_argument("--save_path",  type=str,   default=None)
    p.add_argument("--seed",       type=int,   default=None)

    # Eval-only
    p.add_argument("--checkpoint", type=str,   default=None)
    p.add_argument("--n_test",     type=int,   default=10_000)
    p.add_argument("--plot",       type=str,   default="results.png")
    return p.parse_args()


def main():
    args   = parse_args()
    config = Config()

    for key in ["epochs", "batch_size", "lr", "d_model", "n_layers",
                "T", "sigma2", "save_path", "seed"]:
        val = getattr(args, key)
        if val is not None:
            setattr(config, key, val)

    history = {}

    if args.mode in ("train", "all"):
        _, history = train(config)

    if args.mode in ("eval", "all"):
        ckpt_path = args.checkpoint or config.save_path
        model, ckpt_history = load_model(ckpt_path, config)
        if not history:
            history = ckpt_history

        results = evaluate(model, config, n_test=args.n_test)
        print(f"\nTest SER : {results['ser']:.4f}")
        print(f"Per-pos  : {[f'{s:.4f}' for s in results['per_pos_ser']]}")

        if history:
            plot_results(history, results, config, save_path=args.plot)


if __name__ == "__main__":
    main()
