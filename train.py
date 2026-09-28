"""
Training Pipeline for Plant Leaf CNN
=====================================
Handles:
  - Data loading and preprocessing
  - Train/validation/test split
  - Mini-batch training with Adam optimizer
  - Learning rate scheduling
  - Training history tracking
  - Model checkpointing
"""

import numpy as np
import time
import os

from dataset_generator import generate_dataset, SPECIES, IMG_SIZE
from cnn_model import LeafCNN


def preprocess(X):
    """Normalize images to [0, 1] and convert to channels-first format.

    Args:
        X: array of shape (N, H, W, 3), dtype uint8
    Returns:
        X: array of shape (N, 3, H, W), dtype float32
    """
    X = X.astype(np.float32) / 255.0
    # HWC -> CHW
    X = X.transpose(0, 3, 1, 2)
    return X


def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split dataset into train/val/test sets."""
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = rng.permutation(n)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def augment_batch(X_batch, rng):
    """Apply simple data augmentation.
    - Random horizontal flip
    - Random brightness adjustment
    """
    augmented = X_batch.copy()
    batch_size = augmented.shape[0]

    for i in range(batch_size):
        # Random horizontal flip
        if rng.random() > 0.5:
            augmented[i] = augmented[i, :, :, ::-1]

        # Random brightness
        factor = rng.uniform(0.85, 1.15)
        augmented[i] = np.clip(augmented[i] * factor, 0, 1)

    return augmented


def evaluate(model, X, y, batch_size=32):
    """Evaluate model on a dataset."""
    n = len(X)
    correct = 0
    total_loss = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X[start:end]
        y_batch = y[start:end]

        probs = model.forward(X_batch)
        loss = model.compute_loss(probs, y_batch)
        preds = np.argmax(probs, axis=1)

        correct += np.sum(preds == y_batch)
        total_loss += loss
        n_batches += 1

    accuracy = correct / n
    avg_loss = total_loss / n_batches
    return accuracy, avg_loss


def train(config=None):
    """Main training function."""
    # Default config
    if config is None:
        config = {
            'samples_per_class': 200,
            'epochs': 15,
            'batch_size': 32,
            'learning_rate': 0.001,
            'lr_decay': 0.95,
            'seed': 42,
            'save_dir': 'checkpoints',
            'augment': True,
        }

    print("\n" + "="*60)
    print("  PLANT LEAF CNN - TRAINING")
    print("="*60)
    print(f"  Config: {config}")

    # ─────── 1. Generate Dataset ───────
    print("\n[Step 1] Generating dataset...")
    X, y, species = generate_dataset(
        samples_per_class=config['samples_per_class'],
        seed=config['seed'],
        output_dir='dataset/plant_leaves'
    )
    print(f"  Total samples: {len(X)}")

    # ─────── 2. Preprocess ───────
    print("\n[Step 2] Preprocessing...")
    X = preprocess(X)
    print(f"  Shape after preprocessing: {X.shape}")
    print(f"  Dtype: {X.dtype}, Range: [{X.min():.3f}, {X.max():.3f}]")

    # ─────── 3. Split ───────
    print("\n[Step 3] Splitting dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, seed=config['seed']
    )
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Class distribution
    print("\n  Class distribution (train):")
    for i, name in enumerate(species):
        count = np.sum(y_train == i)
        print(f"    {name:15s}: {count:4d} samples")

    # ─────── 4. Build Model ───────
    print("\n[Step 4] Building CNN model...")
    np.random.seed(config['seed'])
    model = LeafCNN(num_classes=len(species))
    model.summary()

    # ─────── 5. Training Loop ───────
    print("\n[Step 5] Training...")
    rng = np.random.RandomState(config['seed'])
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_times': []
    }

    best_val_acc = 0.0
    lr = config['learning_rate']
    batch_size = config['batch_size']
    os.makedirs(config['save_dir'], exist_ok=True)

    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()

        # Shuffle training data
        perm = rng.permutation(len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0

        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Data augmentation
            if config['augment']:
                X_batch = augment_batch(X_batch, rng)

            # Forward pass
            probs = model.forward(X_batch)
            loss = model.compute_loss(probs, y_batch)

            # Backward pass
            model.backward(y_batch)

            # Update weights
            model.update_params(lr=lr)

            # Track metrics
            epoch_loss += loss
            preds = np.argmax(probs, axis=1)
            epoch_correct += np.sum(preds == y_batch)
            n_batches += 1

        # Epoch metrics
        train_loss = epoch_loss / n_batches
        train_acc = epoch_correct / len(X_train)

        # Validation
        val_acc, val_loss = evaluate(model, X_val, y_val, batch_size)

        epoch_time = time.time() - epoch_start

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)

        # Print progress
        print(f"  Epoch {epoch:2d}/{config['epochs']}  "
              f"loss={train_loss:.4f}  acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={lr:.6f}  time={epoch_time:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(os.path.join(config['save_dir'], 'best_model.pkl'))
            print(f"  >>> New best val_acc: {best_val_acc:.4f} - model saved!")

        # Learning rate decay
        lr *= config['lr_decay']

    # ─────── 6. Final Evaluation ───────
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    total_time = sum(history['epoch_times'])
    print(f"  Total training time: {total_time:.1f}s")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")

    # Load best model for test evaluation
    model.load(os.path.join(config['save_dir'], 'best_model.pkl'))
    test_acc, test_loss = evaluate(model, X_test, y_test, batch_size)
    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss:     {test_loss:.4f}")

    return model, history, (X_train, y_train, X_val, y_val, X_test, y_test), species


if __name__ == "__main__":
    model, history, data_splits, species = train()
