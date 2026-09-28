"""
╔══════════════════════════════════════════════════════════════╗
║        Plant Leaf Classification using CNN                   ║
║        ─────────────────────────────────────                  ║
║  A complete deep learning project that:                      ║
║    1. Generates a synthetic leaf dataset (10 species)        ║
║    2. Builds a CNN from scratch using NumPy                  ║
║    3. Trains with mini-batch SGD + Adam optimizer            ║
║    4. Evaluates with comprehensive metrics & visualizations  ║
║                                                              ║
║  Species: Maple, Oak, Birch, Willow, Pine,                   ║
║           Eucalyptus, Palm, Fern, Ivy, Mint                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import argparse
import numpy as np
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train
from evaluate import run_evaluation


def parse_args():
    """Parse user options for a quick or full training run."""
    parser = argparse.ArgumentParser(description='Plant leaf CNN training pipeline')
    parser.add_argument('--samples-per-class', type=int, default=60,
                        help='Number of synthetic images per species (default: 60)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Mini-batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--lr-decay', type=float, default=0.97,
                        help='Learning rate decay factor per epoch (default: 0.97)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation during training')
    return parser.parse_args()


def main():
    """Run the complete Plant Leaf Classification pipeline."""
    print(__doc__)
    args = parse_args()

    start_time = time.time()

    # ─────── Configuration ───────
    # Default values are intentionally lighter so the pipeline completes reliably on CPU.
    config = {
        'samples_per_class': args.samples_per_class,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'lr_decay': args.lr_decay,
        'seed': args.seed,
        'save_dir': 'checkpoints',
        'augment': not args.no_augment,
    }

    print(f"\nRunning with config: {config}")

    # ─────── Train ───────
    model, history, data_splits, species = train(config)

    # ─────── Evaluate ───────
    run_evaluation(model, history, data_splits, species, save_dir='results')

    total_time = time.time() - start_time
    print(f"\n  Total pipeline time: {total_time:.1f}s")
    print("\n  Project files:")
    print("    dataset/plant_leaves/  - Generated leaf images")
    print("    checkpoints/           - Saved model weights")
    print("    results/               - Evaluation plots & metrics")
    print("\n  Done! \U0001f331")


if __name__ == "__main__":
    main()
