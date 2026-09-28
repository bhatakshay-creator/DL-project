"""
CNN Model from Scratch using NumPy
===================================
Implements a Convolutional Neural Network with:
  - Conv2D layers (with filters, strides)
  - ReLU activation
  - MaxPooling
  - Flatten
  - Dense (fully-connected) layers
  - Softmax output
  - Cross-entropy loss
  - Adam optimizer

Architecture:
  Input (64x64x3)
    -> Conv2D(16, 3x3) -> ReLU -> MaxPool(2x2)
    -> Conv2D(32, 3x3) -> ReLU -> MaxPool(2x2)
    -> Conv2D(64, 3x3) -> ReLU -> MaxPool(2x2)
    -> Flatten
    -> Dense(128) -> ReLU
    -> Dense(10) -> Softmax
"""

import numpy as np
import pickle
import os


# ═══════════════════════════════════════════════════════════════
#  LAYER CLASSES
# ═══════════════════════════════════════════════════════════════

class Conv2D:
    """2D Convolution layer with 'valid' padding."""

    def __init__(self, num_filters, kernel_size, input_channels, stride=1):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        # He initialization
        scale = np.sqrt(2.0 / (kernel_size * kernel_size * input_channels))
        self.filters = np.random.randn(
            num_filters, input_channels, kernel_size, kernel_size
        ).astype(np.float32) * scale
        self.biases = np.zeros(num_filters, dtype=np.float32)
        # Cache for backprop
        self.input_cache = None
        # Adam optimizer state
        self.m_filters = np.zeros_like(self.filters)
        self.v_filters = np.zeros_like(self.filters)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)

    def forward(self, x):
        """Forward pass.
        Args:
            x: input of shape (batch, channels, height, width)
        Returns:
            output of shape (batch, num_filters, out_h, out_w)
        """
        self.input_cache = x
        batch, in_c, in_h, in_w = x.shape
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1

        # im2col for efficient computation
        cols = self._im2col(x, out_h, out_w)
        filters_reshaped = self.filters.reshape(self.num_filters, -1)
        output = filters_reshaped @ cols + self.biases.reshape(-1, 1)
        output = output.reshape(self.num_filters, batch, out_h, out_w)
        output = output.transpose(1, 0, 2, 3)
        return output

    def backward(self, d_out):
        """Backward pass.
        Args:
            d_out: gradient of shape (batch, num_filters, out_h, out_w)
        Returns:
            d_input: gradient w.r.t. input
        """
        x = self.input_cache
        batch, in_c, in_h, in_w = x.shape
        _, _, out_h, out_w = d_out.shape

        # Gradient w.r.t. biases
        self.d_biases = np.sum(d_out, axis=(0, 2, 3))

        # Gradient w.r.t. filters
        cols = self._im2col(x, out_h, out_w)
        d_out_reshaped = d_out.transpose(1, 0, 2, 3).reshape(self.num_filters, -1)
        self.d_filters = (d_out_reshaped @ cols.T).reshape(self.filters.shape)

        # Gradient w.r.t. input
        filters_reshaped = self.filters.reshape(self.num_filters, -1)
        d_cols = filters_reshaped.T @ d_out_reshaped
        d_input = self._col2im(d_cols, x.shape, out_h, out_w)

        return d_input

    def _im2col(self, x, out_h, out_w):
        batch, in_c, in_h, in_w = x.shape
        k = self.kernel_size
        cols = np.zeros((in_c * k * k, batch * out_h * out_w), dtype=np.float32)
        idx = 0
        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, :, i*self.stride:i*self.stride+k, j*self.stride:j*self.stride+k]
                cols[:, idx:idx+batch] = patch.reshape(batch, -1).T
                idx += batch
        return cols

    def _col2im(self, cols, x_shape, out_h, out_w):
        batch, in_c, in_h, in_w = x_shape
        k = self.kernel_size
        d_input = np.zeros(x_shape, dtype=np.float32)
        idx = 0
        for i in range(out_h):
            for j in range(out_w):
                patch = cols[:, idx:idx+batch].T.reshape(batch, in_c, k, k)
                d_input[:, :, i*self.stride:i*self.stride+k, j*self.stride:j*self.stride+k] += patch
                idx += batch
        return d_input


class MaxPool2D:
    """Max Pooling layer."""

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input_cache = None
        self.mask = None

    def forward(self, x):
        self.input_cache = x
        batch, channels, in_h, in_w = x.shape
        out_h = (in_h - self.pool_size) // self.stride + 1
        out_w = (in_w - self.pool_size) // self.stride + 1

        output = np.zeros((batch, channels, out_h, out_w), dtype=np.float32)
        self.mask = np.zeros_like(x)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                region = x[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size]
                output[:, :, i, j] = np.max(region, axis=(2, 3))
                # Create mask for backprop
                max_val = output[:, :, i, j][:, :, np.newaxis, np.newaxis]
                self.mask[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size] += \
                    (region == max_val).astype(np.float32)

        return output

    def backward(self, d_out):
        batch, channels, out_h, out_w = d_out.shape
        d_input = np.zeros_like(self.input_cache)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                mask_region = self.mask[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size]
                d_input[:, :, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size] += \
                    mask_region * d_out[:, :, i, j][:, :, np.newaxis, np.newaxis]

        return d_input


class ReLU:
    """ReLU activation."""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask

    def backward(self, d_out):
        return d_out * self.mask


class Flatten:
    """Flatten spatial dimensions."""

    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.input_shape)


class Dense:
    """Fully connected layer."""

    def __init__(self, input_dim, output_dim):
        scale = np.sqrt(2.0 / input_dim)
        self.weights = np.random.randn(input_dim, output_dim).astype(np.float32) * scale
        self.biases = np.zeros(output_dim, dtype=np.float32)
        self.input_cache = None
        # Adam state
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)

    def forward(self, x):
        self.input_cache = x
        return x @ self.weights + self.biases

    def backward(self, d_out):
        self.d_weights = self.input_cache.T @ d_out
        self.d_biases = np.sum(d_out, axis=0)
        d_input = d_out @ self.weights.T
        return d_input


class Softmax:
    """Softmax activation (used with cross-entropy loss)."""

    def __init__(self):
        self.output = None

    def forward(self, x):
        # Numerical stability
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, y_true_onehot):
        """Combined softmax + cross-entropy backward.
        Returns d_loss/d_logits = softmax_output - y_true_onehot
        """
        return (self.output - y_true_onehot) / y_true_onehot.shape[0]


# ═══════════════════════════════════════════════════════════════
#  CNN MODEL
# ═══════════════════════════════════════════════════════════════

class LeafCNN:
    """CNN for Plant Leaf Classification.

    Architecture:
        Conv2D(16, 3x3) -> ReLU -> MaxPool(2x2)
        Conv2D(32, 3x3) -> ReLU -> MaxPool(2x2)
        Conv2D(64, 3x3) -> ReLU -> MaxPool(2x2)
        Flatten
        Dense(128) -> ReLU
        Dense(10) -> Softmax
    """

    def __init__(self, num_classes=10, input_channels=3):
        self.num_classes = num_classes

        # Build layers
        self.layers = [
            # Block 1: Conv(16) -> ReLU -> MaxPool
            Conv2D(num_filters=16, kernel_size=3, input_channels=input_channels),  # 64->62
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),  # 62->31

            # Block 2: Conv(32) -> ReLU -> MaxPool
            Conv2D(num_filters=32, kernel_size=3, input_channels=16),  # 31->29
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),  # 29->14

            # Block 3: Conv(64) -> ReLU -> MaxPool
            Conv2D(num_filters=64, kernel_size=3, input_channels=32),  # 14->12
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),  # 12->6

            # Flatten
            Flatten(),  # 64*6*6 = 2304

            # Dense layers
            Dense(input_dim=64*6*6, output_dim=128),
            ReLU(),
            Dense(input_dim=128, output_dim=num_classes),
            Softmax()
        ]

        self.adam_t = 0  # Adam timestep

    def forward(self, x):
        """Forward pass through all layers.
        Args:
            x: input of shape (batch, channels, height, width)
        Returns:
            probabilities of shape (batch, num_classes)
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss(self, probs, y):
        """Cross-entropy loss."""
        batch_size = y.shape[0]
        # Clip for numerical stability
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        log_probs = -np.log(probs_clipped[np.arange(batch_size), y])
        return np.mean(log_probs)

    def backward(self, y):
        """Backward pass through all layers."""
        batch_size = y.shape[0]
        # One-hot encode targets
        y_onehot = np.zeros((batch_size, self.num_classes), dtype=np.float32)
        y_onehot[np.arange(batch_size), y] = 1.0

        # Start from softmax+CE combined backward
        grad = self.layers[-1].backward(y_onehot)

        # Propagate through remaining layers in reverse
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad)

    def update_params(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """Update parameters using Adam optimizer."""
        self.adam_t += 1
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                self._adam_update(
                    layer.filters, layer.d_filters,
                    layer.m_filters, layer.v_filters,
                    lr, beta1, beta2, eps
                )
                self._adam_update(
                    layer.biases, layer.d_biases,
                    layer.m_biases, layer.v_biases,
                    lr, beta1, beta2, eps
                )
            elif isinstance(layer, Dense):
                self._adam_update(
                    layer.weights, layer.d_weights,
                    layer.m_weights, layer.v_weights,
                    lr, beta1, beta2, eps
                )
                self._adam_update(
                    layer.biases, layer.d_biases,
                    layer.m_biases, layer.v_biases,
                    lr, beta1, beta2, eps
                )

    def _adam_update(self, param, grad, m, v, lr, beta1, beta2, eps):
        """Single Adam parameter update (in-place)."""
        m[:] = beta1 * m + (1 - beta1) * grad
        v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** self.adam_t)
        v_hat = v / (1 - beta2 ** self.adam_t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, x):
        """Predict class labels."""
        probs = self.forward(x)
        return np.argmax(probs, axis=1)

    def save(self, filepath):
        """Save model weights."""
        state = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv2D):
                state[f'layer_{i}_filters'] = layer.filters
                state[f'layer_{i}_biases'] = layer.biases
            elif isinstance(layer, Dense):
                state[f'layer_{i}_weights'] = layer.weights
                state[f'layer_{i}_biases'] = layer.biases
        state['adam_t'] = self.adam_t

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"  Model saved to {filepath}")

    def load(self, filepath):
        """Load model weights."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv2D):
                layer.filters = state[f'layer_{i}_filters']
                layer.biases = state[f'layer_{i}_biases']
            elif isinstance(layer, Dense):
                layer.weights = state[f'layer_{i}_weights']
                layer.biases = state[f'layer_{i}_biases']
        self.adam_t = state.get('adam_t', 0)
        print(f"  Model loaded from {filepath}")

    def summary(self):
        """Print model architecture summary."""
        print("\n" + "="*60)
        print("  LeafCNN Model Summary")
        print("="*60)
        total_params = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv2D):
                params = layer.filters.size + layer.biases.size
                total_params += params
                print(f"  [{i:2d}] Conv2D      filters={layer.num_filters:3d}  "
                      f"kernel={layer.kernel_size}x{layer.kernel_size}  params={params:,}")
            elif isinstance(layer, MaxPool2D):
                print(f"  [{i:2d}] MaxPool2D   pool={layer.pool_size}x{layer.pool_size}  "
                      f"stride={layer.stride}")
            elif isinstance(layer, ReLU):
                print(f"  [{i:2d}] ReLU")
            elif isinstance(layer, Flatten):
                print(f"  [{i:2d}] Flatten")
            elif isinstance(layer, Dense):
                params = layer.weights.size + layer.biases.size
                total_params += params
                print(f"  [{i:2d}] Dense       in={layer.weights.shape[0]:5d}  "
                      f"out={layer.weights.shape[1]:4d}  params={params:,}")
            elif isinstance(layer, Softmax):
                print(f"  [{i:2d}] Softmax")
        print("-"*60)
        print(f"  Total trainable parameters: {total_params:,}")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Quick test
    model = LeafCNN(num_classes=10)
    model.summary()
    # Test forward pass
    dummy = np.random.randn(2, 3, 64, 64).astype(np.float32)
    out = model.forward(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Output sums to 1? {np.allclose(out.sum(axis=1), 1.0)}")
