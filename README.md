# 🌿 Plant Leaf Classification using CNN

A complete deep learning project that classifies plant species from leaf images using a **Convolutional Neural Network (CNN) built entirely from scratch in NumPy**.

## 📋 Project Overview

| Component | Description |
|-----------|-------------|
| **Dataset** | Synthetically generated leaf images (10 species, 64×64 RGB) |
| **Model** | Custom CNN with 3 conv blocks + 2 dense layers |
| **Framework** | Pure NumPy (no TensorFlow/PyTorch dependency) |
| **Optimizer** | Adam with learning rate decay |
| **Augmentation** | Random flips and brightness adjustments |

## 🌱 Species (10 Classes)

| # | Species | Leaf Characteristics |
|---|---------|---------------------|
| 0 | Maple | Star/palmate shape, warm red-orange colors |
| 1 | Oak | Rounded lobed shape, dark green |
| 2 | Birch | Oval with serrated edges, light green |
| 3 | Willow | Long narrow ellipse, yellow-green |
| 4 | Pine | Needle-like thin shape, dark green |
| 5 | Eucalyptus | Elongated ellipse, blue-green tint |
| 6 | Palm | Fan shape with radiating rays, bright green |
| 7 | Fern | Compound shape with small leaflets |
| 8 | Ivy | Heart/triangular shape, dark green |
| 9 | Mint | Small oval with pronounced serrations |

## 🏗️ CNN Architecture

```
Input (3 × 64 × 64)
    │
    ├─ Conv2D(16, 3×3) → ReLU → MaxPool(2×2)    →  16 × 31 × 31
    ├─ Conv2D(32, 3×3) → ReLU → MaxPool(2×2)    →  32 × 14 × 14
    ├─ Conv2D(64, 3×3) → ReLU → MaxPool(2×2)    →  64 ×  6 ×  6
    │
    ├─ Flatten                                    →  2304
    ├─ Dense(128) → ReLU                          →  128
    └─ Dense(10) → Softmax                        →  10 (probabilities)
```

**Total Parameters: ~297,000**

## 📁 Project Structure

```
Plant_Leaf_Classification/
├── main.py                 # 🚀 Main entry point - runs full pipeline
├── dataset_generator.py    # 🎨 Synthetic leaf image generator
├── cnn_model.py            # 🧠 CNN model (Conv2D, MaxPool, Dense, etc.)
├── train.py                # 🏋️ Training pipeline with Adam optimizer
├── evaluate.py             # 📊 Evaluation, metrics & visualizations
├── README.md               # 📖 This file
│
├── dataset/plant_leaves/   # Generated after running (auto-created)
│   ├── Maple/
│   ├── Oak/
│   ├── ... (10 species folders)
│   └── metadata.json
│
├── checkpoints/            # Saved model weights (auto-created)
│   └── best_model.pkl
│
└── results/                # Evaluation outputs (auto-created)
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── per_class_metrics.png
    ├── dataset_samples.png
    ├── correct_predictions.png
    └── incorrect_predictions.png
```

## 🚀 How to Run

```bash
# Run the complete pipeline (dataset generation → training → evaluation)
python main.py
```

Or run individual components:

```bash
# Generate dataset only
python dataset_generator.py

# Train model only
python train.py

# Test model architecture
python cnn_model.py
```

## 📊 Outputs

After running, check the `results/` folder for:

1. **Training Curves** - Loss and accuracy over epochs
2. **Confusion Matrix** - Raw counts and normalized heatmaps
3. **Per-Class Metrics** - Precision, Recall, F1-Score bar charts
4. **Dataset Samples** - 5 sample images from each species
5. **Prediction Samples** - Correct and incorrect classification examples

## 🔧 Configuration

Edit the `config` dictionary in `main.py`:

```python
config = {
    'samples_per_class': 60,     # Images per species for a faster CPU-friendly run
    'epochs': 5,                 # Training epochs
    'batch_size': 32,            # Mini-batch size
    'learning_rate': 0.001,      # Initial learning rate
    'lr_decay': 0.97,            # LR decay factor per epoch
    'seed': 42,                  # Random seed
    'augment': True,             # Enable data augmentation
}
```

You can also override these at runtime:

```bash
python main.py --samples-per-class 150 --epochs 15 --learning-rate 0.001
```

## 📦 Requirements

- Python 3.8+
- NumPy
- Pillow (PIL)
- Matplotlib
- scikit-learn

## 🧪 Key Concepts Demonstrated

- **Convolution operation** (im2col optimization)
- **Backpropagation** through conv, pool, and dense layers
- **Adam optimizer** implementation
- **Data augmentation** (flips, brightness)
- **He weight initialization**
- **Learning rate scheduling**
- **Model checkpointing** (save/load best model)
- **Comprehensive evaluation** (confusion matrix, precision/recall/F1)
