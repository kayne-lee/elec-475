"""
ELEC 475: Lab 2 - Plotting & Visualization Utilities
"""
import os
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "./plots/training_curves.png",
    model_name: str = "Model"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r--s', label='Val Loss')
    plt.title(f"{model_name.upper()} - Cross-Entropy Loss", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'g--^', label='Val Accuracy')
    plt.title(f"{model_name.upper()} - Top-1 Accuracy (%)", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training curves successfully saved to: {save_path}")

def plot_ablation_summary(
    ablation_results: Dict[str, Dict[str, List[float]]],
    save_path: str = "./plots/ablation_summary.png",
    title: str = "ELEC 475: Master Ablation Study (Validation Accuracy)"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    styles = ['-', '--', '-.', ':', '-', '--', '-.']

    for idx, (model_label, data) in enumerate(ablation_results.items()):
        val_accs = data.get("val_accs", [])
        epochs = range(1, len(val_accs) + 1)
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        plt.plot(epochs, val_accs, linestyle=style, marker='o', color=color, label=model_label, linewidth=2)

    plt.title(title, fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Master ablation summary plot saved to: {save_path}")

def parse_training_log(log_path: str) -> Dict[str, List[float]]:
    """Recovers per-epoch history from the epoch table printed by train.py."""
    history = {"train_losses": [], "train_accs": [], "val_losses": [], "val_accs": [], "val_f1s": [], "epoch_times": []}
    with open(log_path) as f:
        for line in f:
            cols = [c.strip() for c in line.split("|")]
            if len(cols) == 8 and cols[0].isdigit():
                history["train_losses"].append(float(cols[1]))
                history["train_accs"].append(float(cols[2]))
                history["val_losses"].append(float(cols[3]))
                history["val_accs"].append(float(cols[4]))
                history["val_f1s"].append(float(cols[5]))
                history["epoch_times"].append(float(cols[6]))
    return history

# CIFAR-10 training logs plotted by --plot-ablation (written by commands.txt)
ABLATION_LOGS = {
    "1. Base LeNet-5": "logs/lenet.log",
    "2. Modernized LeNet": "logs/lenet_modern.log",
    "3. AlexNet (No Norm)": "logs/alexnet_none.log",
    "4. AlexNet + LRN": "logs/alexnet_lrn.log",
    "5. AlexNet + BatchNorm": "logs/alexnet_bn.log",
    "6. AlexNet + BN + Aug": "logs/alexnet_with_aug.log",
    "7. PyTorch Zoo AlexNet": "logs/zoo_alexnet.log",
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ELEC 475 Lab 2 visualization utilities")
    parser.add_argument("--plot-ablation", action="store_true", help="Plot validation accuracy of every ablation run")
    parser.add_argument("--save-path", type=str, default="./plots/ablation_summary.png")
    args = parser.parse_args()

    if args.plot_ablation:
        results = {}
        for label, path in ABLATION_LOGS.items():
            if os.path.exists(path):
                results[label] = parse_training_log(path)
            else:
                print(f"Skipping '{label}': {path} not found")
        plot_ablation_summary(results, save_path=args.save_path)
    else:
        parser.print_help()
