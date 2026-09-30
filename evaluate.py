"""
Evaluation & Visualization for Plant Leaf CNN
==============================================
Generates:
  1. Training/Validation loss curves
  2. Training/Validation accuracy curves
  3. Confusion matrix heatmap
  4. Per-class precision, recall, F1-score
  5. Sample predictions visualization
  6. Feature map visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support
)
import os

from dataset_generator import SPECIES


def plot_training_curves(history, save_dir='results'):
    """Plot training and validation loss/accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss curves ──
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # ── Accuracy curves ──
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=4)
    axes[1].plot(epochs, history['val_acc'], 'r-s', label='Val Acc', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, species_names, save_dir='results'):
    """Plot confusion matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    im1 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    axes[0].set_xticks(range(len(species_names)))
    axes[0].set_yticks(range(len(species_names)))
    axes[0].set_xticklabels(species_names, rotation=45, ha='right', fontsize=9)
    axes[0].set_yticklabels(species_names, fontsize=9)
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)

    # Add text annotations
    for i in range(len(species_names)):
        for j in range(len(species_names)):
            axes[0].text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black',
                        fontsize=8)

    # Normalized
    im2 = axes[1].imshow(cm_normalized, interpolation='nearest', cmap='Blues',
                          vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    axes[1].set_xticks(range(len(species_names)))
    axes[1].set_yticks(range(len(species_names)))
    axes[1].set_xticklabels(species_names, rotation=45, ha='right', fontsize=9)
    axes[1].set_yticklabels(species_names, fontsize=9)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)

    for i in range(len(species_names)):
        for j in range(len(species_names)):
            axes[1].text(j, i, f'{cm_normalized[i, j]:.2f}',
                        ha='center', va='center',
                        color='white' if cm_normalized[i, j] > 0.5 else 'black',
                        fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_per_class_metrics(y_true, y_pred, species_names, save_dir='results'):
    """Plot per-class precision, recall, F1 as bar charts."""
    os.makedirs(save_dir, exist_ok=True)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(species_names)),
        zero_division=0
    )

    x = np.arange(len(species_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#4CAF50', alpha=0.85)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#FF9800', alpha=0.85)

    ax.set_xlabel('Species', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(species_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=3)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(save_dir, 'per_class_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_sample_predictions(model, X_test, y_test, species_names,
                            n_samples=20, save_dir='results'):
    """Show sample predictions: correct and incorrect."""
    os.makedirs(save_dir, exist_ok=True)

    # Get predictions
    preds = []
    for start in range(0, len(X_test), 32):
        end = min(start + 32, len(X_test))
        batch_preds = model.predict(X_test[start:end])
        preds.extend(batch_preds)
    preds = np.array(preds)

    # Find correct and incorrect
    correct_mask = preds == y_test
    incorrect_mask = ~correct_mask

    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]

    # ── Plot correct predictions ──
    n_correct = min(n_samples, len(correct_indices))
    if n_correct > 0:
        fig, axes = plt.subplots(2, min(n_correct, 10), figsize=(20, 5))
        if n_correct == 1:
            axes = np.array([[axes]])
        elif n_correct <= 10:
            axes = axes.reshape(1, -1) if axes.ndim == 1 else axes

        fig.suptitle('Sample CORRECT Predictions', fontsize=16, fontweight='bold', color='green')

        for k in range(min(n_correct, 20)):
            row = k // 10
            col = k % 10
            if n_correct <= 10:
                ax = axes[0, col] if axes.ndim > 1 else axes[col]
            else:
                ax = axes[row, col]

            idx = correct_indices[k]
            img = X_test[idx].transpose(1, 2, 0)  # CHW -> HWC
            img = np.clip(img, 0, 1)

            ax.imshow(img)
            ax.set_title(f'True: {species_names[y_test[idx]]}\nPred: {species_names[preds[idx]]}',
                        fontsize=8, color='green')
            ax.axis('off')

        plt.tight_layout()
        path = os.path.join(save_dir, 'correct_predictions.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # ── Plot incorrect predictions ──
    n_incorrect = min(n_samples, len(incorrect_indices))
    if n_incorrect > 0:
        n_show = min(n_incorrect, 10)
        fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2, 3))
        if n_show == 1:
            axes = [axes]
        fig.suptitle('Sample INCORRECT Predictions', fontsize=16, fontweight='bold', color='red')

        for k in range(n_show):
            idx = incorrect_indices[k]
            img = X_test[idx].transpose(1, 2, 0)
            img = np.clip(img, 0, 1)

            axes[k].imshow(img)
            axes[k].set_title(f'True: {species_names[y_test[idx]]}\nPred: {species_names[preds[idx]]}',
                            fontsize=8, color='red')
            axes[k].axis('off')

        plt.tight_layout()
        path = os.path.join(save_dir, 'incorrect_predictions.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    else:
        print("  No incorrect predictions to show!")


def plot_dataset_samples(X, y, species_names, save_dir='results'):
    """Show sample images from each class."""
    os.makedirs(save_dir, exist_ok=True)
    n_classes = len(species_names)
    n_per_class = 5

    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 2, n_classes * 2))
    fig.suptitle('Dataset Samples (5 per species)', fontsize=16, fontweight='bold')

    for cls_id in range(n_classes):
        cls_indices = np.where(y == cls_id)[0][:n_per_class]
        for j, idx in enumerate(cls_indices):
            img = X[idx].transpose(1, 2, 0)  # CHW -> HWC
            img = np.clip(img, 0, 1)
            axes[cls_id, j].imshow(img)
            axes[cls_id, j].axis('off')
            if j == 0:
                axes[cls_id, j].set_ylabel(species_names[cls_id], fontsize=10,
                                           rotation=0, labelpad=60, va='center')

    plt.tight_layout(rect=[0.08, 0, 1, 0.96])
    path = os.path.join(save_dir, 'dataset_samples.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_dataset_distribution(y, species_names, save_dir='results'):
    """Plot class counts and the class proportions in the dataset."""
    os.makedirs(save_dir, exist_ok=True)
    counts = np.bincount(y, minlength=len(species_names))
    colors = plt.cm.Set3(np.linspace(0, 1, len(species_names)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(species_names, counts, color=colors, edgecolor='#333333', linewidth=0.5)
    axes[0].set_title('Images per Species', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of images')
    axes[0].tick_params(axis='x', rotation=40)
    axes[0].grid(axis='y', alpha=0.25)
    for index, count in enumerate(counts):
        axes[0].text(index, count, str(count), ha='center', va='bottom', fontsize=9)

    axes[1].pie(
        counts,
        labels=species_names,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        textprops={'fontsize': 9},
    )
    axes[1].set_title('Species Share of Dataset', fontsize=14, fontweight='bold')

    fig.suptitle('Dataset Class Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'dataset_distribution.png')
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_species_accuracy(y_true, y_pred, species_names, save_dir='results'):
    """Plot test accuracy (per-class recall) and test support for each species."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(species_names)))
    support = cm.sum(axis=1)
    accuracy = np.divide(
        np.diag(cm), support, out=np.zeros(len(species_names), dtype=float), where=support > 0
    )
    order = np.argsort(accuracy)
    colors = ['#2E8B57' if score >= 0.9 else '#D98E04' for score in accuracy[order]]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(np.array(species_names)[order], accuracy[order], color=colors)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel('Test accuracy (recall)')
    ax.set_title('Test Accuracy by Species', fontsize=15, fontweight='bold')
    ax.grid(axis='x', alpha=0.25)
    for bar, score, count in zip(bars, accuracy[order], support[order]):
        ax.text(score + 0.015, bar.get_y() + bar.get_height() / 2,
                f'{score:.1%}  (n={count})', va='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, 'species_accuracy.png')
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_cnn_architecture(save_dir='results'):
    """Draw the CNN layers and feature-map dimensions used by LeafCNN."""
    from matplotlib.patches import FancyBboxPatch

    os.makedirs(save_dir, exist_ok=True)
    layers = [
        ('Input', '3 x 64 x 64\nRGB image', '#DCEAF7'),
        ('Conv block 1', 'Conv 3x3, 16\nReLU + MaxPool\n16 x 31 x 31', '#B8D8F0'),
        ('Conv block 2', 'Conv 3x3, 32\nReLU + MaxPool\n32 x 14 x 14', '#A8D5BA'),
        ('Conv block 3', 'Conv 3x3, 64\nReLU + MaxPool\n64 x 6 x 6', '#F6D6A8'),
        ('Flatten', '2,304 features', '#E8D5EF'),
        ('Dense', '128 units\nReLU', '#F4C7C3'),
        ('Output', '10 units\nSoftmax probabilities', '#F3E6A2'),
    ]
    fig, ax = plt.subplots(figsize=(17, 5.5))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    box_width = 2.0
    box_height = 2.2
    positions = np.linspace(0.25, 14.75, len(layers))

    for index, (title, details, color) in enumerate(layers):
        x = positions[index]
        box = FancyBboxPatch(
            (x, 1.85), box_width, box_height,
            boxstyle='round,pad=0.08,rounding_size=0.12',
            facecolor=color, edgecolor='#34495E', linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + box_width / 2, 3.55, title, ha='center', va='center',
                fontsize=11, fontweight='bold')
        ax.text(x + box_width / 2, 2.75, details, ha='center', va='center', fontsize=9)
        if index < len(layers) - 1:
            ax.annotate('', xy=(positions[index + 1] - 0.08, 2.95),
                        xytext=(x + box_width + 0.08, 2.95),
                        arrowprops={'arrowstyle': '->', 'color': '#34495E', 'lw': 1.5})

    ax.text(8.5, 4.85, 'Plant Leaf CNN Architecture', ha='center', va='center',
            fontsize=18, fontweight='bold')
    ax.text(8.5, 0.9, 'Valid 3x3 convolutions; 2x2 max pooling with stride 2; '
            '319,914 trainable parameters', ha='center', va='center', fontsize=10)
    path = os.path.join(save_dir, 'cnn_architecture.png')
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def print_classification_report(y_true, y_pred, species_names):
    """Print sklearn classification report."""
    print("\n" + "="*60)
    print("  CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(
        y_true, y_pred,
        target_names=species_names,
        digits=4,
        zero_division=0
    )
    print(report)


def run_evaluation(model, history, data_splits, species_names, save_dir='results'):
    """Run complete evaluation pipeline."""
    X_train, y_train, X_val, y_val, X_test, y_test = data_splits

    print("\n" + "="*60)
    print("  EVALUATION & VISUALIZATION")
    print("="*60)

    # 1. Training curves
    print("\n[1] Plotting training curves...")
    plot_training_curves(history, save_dir)

    # 2. Dataset samples
    print("\n[2] Plotting dataset samples...")
    plot_dataset_samples(X_train, y_train, species_names, save_dir)
    plot_dataset_distribution(
        np.concatenate([y_train, y_val, y_test]), species_names, save_dir
    )
    plot_cnn_architecture(save_dir)

    # 3. Get test predictions
    print("\n[3] Computing test predictions...")
    test_preds = []
    for start in range(0, len(X_test), 32):
        end = min(start + 32, len(X_test))
        batch_preds = model.predict(X_test[start:end])
        test_preds.extend(batch_preds)
    test_preds = np.array(test_preds)

    # 4. Confusion matrix
    print("\n[4] Plotting confusion matrix...")
    plot_confusion_matrix(y_test, test_preds, species_names, save_dir)

    # 5. Per-class metrics
    print("\n[5] Plotting per-class metrics...")
    plot_per_class_metrics(y_test, test_preds, species_names, save_dir)
    plot_species_accuracy(y_test, test_preds, species_names, save_dir)

    # 6. Classification report
    print_classification_report(y_test, test_preds, species_names)

    # 7. Sample predictions
    print("\n[7] Plotting sample predictions...")
    plot_sample_predictions(model, X_test, y_test, species_names,
                           n_samples=10, save_dir=save_dir)

    print("\n" + "="*60)
    print(f"  All results saved to: {save_dir}/")
    print("="*60)
