# 🧠 FlexiNN: Neural Networks from Scratch (But Make It Fun)

**[🇨🇳 中文版本](README_cn.md)** | **[🇬🇧 English](README.md)**

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243)
![CuPy](https://img.shields.io/badge/CuPy-Optional-003F87)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Educational-orange)

This neural network is built **purely with NumPy** — no fancy deep learning frameworks, just good old math and elbow grease. **CuPy optional** for GPU speedup.

> Heads up: This is for learning purposes only. If you're trying to train GPT-4 or predict stock prices, go grab PyTorch. Your time is too valuable for this.

---
## 📖 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [API Documentation](#-api-documentation)
- [Examples](#-examples)
- [XOR Problem](#1-xor-problem)
- [Iris Classification](#2-iris-classification)
- [Mini-Batch Training](#3-mini-batch-training-with-learning-rate-decay)
- [CNN Image Classification](#4-cnn-image-classification)
- [Performance Notes](#-performance-notes)
- [License](#-license)

---

## ✨ Features

- **Zero-Dependency Dark Magic** — Only `numpy`, all gradients calculated by hand. No `import torch` here, cowboy.
- **GPU Option** — Swap in `FNN_cupy.py` and train on CUDA. Same code, different backend (`import cupy as xp`).
- **Super Flexible** — Most settings can be overridden on the fly. Want to change learning rate mid-training? Go wild.
- **Modular Design**: Built-in layers like `Affine`, `Sigmoid`, `ReLU`, `LeakyReLU`, `Tanh`, `SoftmaxWithLoss`, `Dropout`, `Convolution`, and `Pooling`.
- **Fully Working CNN**: `Convolution` + `Pooling` implemented with `im2col`/`col2im`. Max pooling *and* average pooling. No more placeholders, baby.
- **Customizable Network**: Any number of hidden layers, any number of neurons per layer. Mix and match like a neural network DJ.
- **Activation Functions**: `relu`, `leaky_relu`, `sigmoid`, `tanh`. Default is `leaky_relu` (because dead neurons are sad).
- **Smart Weight Initialization**: He-style (`sqrt(2/fan_in)`) for ReLU-ish activations, Xavier-style for sigmoid/tanh, or bring your own `init_weights`.
- **Numerical Stability**: Sigmoid input is clipped to `[-500, 500]`, softmax subtracts the max, loss clips at `1e-12`. No more `NaN` surprises.
- **Training Controls**:
  - Full batch or mini-batch training with `epochs` and `batch_size`.
  - Learning rate decay (for when your network needs to chill out).
  - Dropout regularization (randomly fire some neurons, keep things spicy).
- **Model Persistence**: Save and load trained models to `.npz`. Loading auto-rebuilds the layer structure.

---

## 📦 Installation

Clone the repo and make sure NumPy is installed:

```bash
git clone https://github.com/linuxlemon256/Neural-Network.git
cd Neural-Network
pip install numpy
```

Want GPU acceleration? Install CuPy matching your CUDA version:

```bash
pip install cupy-cuda11x   # or cupy-cuda12x, check https://cupy.dev
```

---

## 🚀 Quick Start

```python
import numpy as xp
from FNN_numpy import MLP   # or: from FlexiNN import MLP

# Generate fake data: 100 samples, 5 features, 3 classes
X = xp.random.randn(100, 5).astype(xp.float32)
y = xp.eye(3)[xp.random.randint(0, 3, 100)].astype(xp.float32) # One-hot labels

# Create network: Input 5 → Hidden [10] → Output 3, train for 500 iterations
net = MLP(input_size=5, hidden_size=[10], output_size=3,
          learning_time=500, learning_rate=0.1)

# Train that bad boy
net.train(X, y)

# Predict some stuff
output = net.predict(X)
pred_class = xp.argmax(output, axis=1)
print("First 5 predictions:", pred_class[:5])
```

GPU version? Just change the import:

```python
from FNN_cupy import MLP   # everything else stays the same
```

---

## ⚙️ How It Works

### 1. Forward Propagation

Each layer does `X @ W + b`, then applies an activation function. The last layer uses Softmax (because probabilities are nice), everything else uses your activation of choice (default: LeakyReLU, because ReLU is too harsh on negative values).

We save intermediate results during forward pass so we can calculate gradients later. It's like leaving breadcrumbs for the backpropagation fairy.

### 2. Backward Propagation (Error Backpropagation)

We use the chain rule to send gradients backwards from the output to the input:

```
dL/dy * dy/dx = dL/dx
```

For the SoftmaxWithLoss layer, gradient is `(softmax_output - true_label) / batch_size`.

For activation layers, gradients are:
- Sigmoid: `out * (1 - out)` — classic S-shaped derivative
- ReLU: `x > 0 ? 1 : 0` — binary switch, harsh but effective
- LeakyReLU: `x > 0 ? 1 : alpha` — like ReLU but with a safety net
- Tanh: `1 - out^2` — squishes things between -1 and 1

For Affine layers, gradients are:
- `dW = x.T @ dout` — transpose trick
- `db = sum(dout, axis=0)` — just sum it up
- `dx = dout @ W.T` — reverse the multiplication

### 3. Convolution & Pooling

Conv is implemented the classic way: `im2col` unrolls each filter window into a column, turning convolution into one big matrix multiply. `col2im` puts the gradient back (yes, it's a scatter-add, no cheating).

Pooling uses max (default) or average, and remembers which neurons won (`mask`) so backprop can route gradients only to the winners.

### 4. Parameter Update

Good old gradient descent:

```
W = W - learning_rate * dW
b = b - learning_rate * db
```

Simple, elegant, and it works (most of the time).

---
## 📚 API Documentation

### `MLP` Class (Fully Connected Network)

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | `int` | - | Number of input features |
| `hidden_size` | `int` or `list` | - | Number of neurons in hidden layers (`[32, 16]` = two hidden layers) |
| `output_size` | `int` | - | Number of output classes |
| `learning_rate` | `float` | `0.1` | Gradient descent learning rate |
| `learning_time` | `int` | `1000` | Total training iterations (full batch mode) |
| `init_weights` | `float` or `None` | `None` | Weight initialization scaling (`None` = He for ReLU-ish / Xavier for sigmoid) |
| `print_every` | `int` | `100` | Print loss every N iterations |
| `print_output` | `bool` | `True` | Whether to print training progress |
| `activation` | `str` | `"leaky_relu"` | Activation function to use |
| `backpropagation` | `str` | `"error-back"` | Backpropagation method |
| `decay_rate` | `float` | `None` | Learning rate decay rate (0.0 to 1.0) |
| `dropout_rate` | `float` | `0.0` | Dropout probability (hidden layers) |

#### Core Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `train(x, t, ...)` | `None` | Train the network |
| `predict(x)` | `xp.ndarray` | Forward pass, returns softmax probabilities |
| `accuracy(x, t)` | `float` | Calculate classification accuracy (inputs = predictions, labels) |
| `save(name=None)` | `None` | Save model to `.npz` file |
| `load(name=None)` | `None` | Load model from `.npz` file (auto-rebuilds layer count) |

#### train() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `xp.ndarray` | - | Training data |
| `t` | `xp.ndarray` | - | Training labels (one-hot encoded) |
| `learning_time` | `int` | `None` | Override training iterations |
| `learning_rate` | `float` | `None` | Override learning rate |
| `batch_size` | `int` | `None` | Mini-batch size (`None` = full batch) |
| `epochs` | `int` | `100` | Number of epochs (mini-batch mode) |
| `decay_rate` | `float` | `None` | Override decay rate |
| `dropout_rate` | `float` | `None` | Override dropout rate |
| `print_every` | `int` | `None` | Override print interval |
| `print_output` | `bool` | `None` | Override print toggle |

### `CNN` Class (Convolutional Network)

#### Initialization Parameters

| Parameter |类型| Default |描述|
|-----------|------|---------|-------------|
| `input_shape` | `tuple` | - | Input image shape `(C, H, W)` |
| `output_size` | `int` | - | Number of output classes |
| `conv_w` | `xp.ndarray` | - | Conv filters, shape `(FN, C, FH, FW)` |
| `conv_b` | `xp.ndarray` | - | Conv biases, shape `(FN,)` |
| `pool_h` / `pool_w` | `int` | - | Pooling window size |
| `activation` | `str` | `"relu"` | Activation after convolution |
| `learning_rate` | `float` | `0.1` | Gradient descent learning rate |
| `learning_time` | `int` | `1000` | Total training iterations |
| `print_every` | `int` | `100` | Print loss every N iterations |
| `print_output` | `bool` | `True` | Whether to print training progress |

#### Core Methods

| Method | Returns |描述|
|--------|---------|-------------|
| `train(x, t, ...)` | `None` | Train the network |
| `predict(x)` | `xp.ndarray` | Forward pass, **returns class indices directly** (`argmax`) |
| `accuracy(x, t)` | `float` | Calculate classification accuracy |

**Note**: The CNN uses `stride=1, pad=0` convolution followed by one pooling layer, then a fully connected output layer. The last hidden dimension is auto-computed as `filter_count * pool_H * pool_W`.

---

## 🧪 Examples

### 1. XOR Problem

XOR is the classic "I can't do this with a straight line" problem. You need at least one hidden layer to solve it.

```python
import numpy as xp
from FNN_numpy import MLP

X = xp.array([[0,0], [0,1], [1,0], [1,1]]).astype(xp.float32)
y = xp.array([[1,0], [0,1], [0,1], [1,0]]).astype(xp.float32) # Two-class one-hot

# Create network: Input 2 → Hidden [4] → Output 2
net = MLP(input_size=2, hidden_size=[4], output_size=2,
          learning_time=2000, learning_rate=0.1)

# Train
net.train(X, y)

# Predict
out = net.predict(X)
predicted_classes = xp.argmax(out, axis=1)
print("Predicted classes:", predicted_classes)
```

---

### 2. Iris Classification

The classic "hello world" of machine learning datasets.

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from FNN_numpy import MLP
import numpy as xp

# Load data
iris = load_iris()
X = xp.array(iris.data, dtype=xp.float32)
y = xp.array(OneHotEncoder(sparse=False).fit_transform(iris.target.reshape(-1,1)), dtype=xp.float32)

# Create network: Input 4 → Hidden [8] → Output 3
net = MLP(input_size=4, hidden_size=[8], output_size=3,
          learning_time=800, learning_rate=0.1)

# Train
net.train(X, y)

# Predict and calculate accuracy
pred = net.predict(X)
accuracy = xp.mean(xp.argmax(pred, axis=1) == xp.array(iris.target))
print(f"Iris accuracy: {accuracy:.2%}")

# Save model
net.save("iris_model")

# Load model
net.load("iris_model")
```

---

### 3. Mini-Batch Training with Learning Rate Decay

For when your dataset is bigger than a breadbox.

```python
import numpy as xp
from FNN_numpy import MLP

# Generate fake data
X = xp.random.randn(500, 10).astype(xp.float32)
y = xp.eye(5)[xp.random.randint(0, 5, 500)].astype(xp.float32)

# Create network: Input 10 → Hidden [32, 16] → Output 5
net = MLP(input_size=10, hidden_size=[32, 16], output_size=5,
          learning_rate=0.1, dropout_rate=0.3)

# Train with mini-batches, decay learning rate by 0.99 each epoch
net.train(X, y, epochs=200, batch_size=32, decay_rate=0.99)

# Predict
output = net.predict(X)
accuracy = net.accuracy(output, y)
print(f"Accuracy: {accuracy:.4f}")
```

---

### 4. CNN Image Classification

MNIST-lite with random 8x8 grayscale images. Conv layer + ReLU + 2x2 max pooling + FC layer.

```python
import numpy as xp
from FNN_numpy import CNN

# Fake images: 50 samples, 1 channel, 8x8
X = xp.random.randn(50, 1, 8, 8).astype(xp.float32)
y = xp.eye(3)[xp.random.randint(0, 3, 50)].astype(xp.float32)

# 4 filters, 1 input channel, 3x3 kernel
conv_w = xp.random.randn(4, 1, 3, 3).astype(xp.float32) * 0.1
conv_b = xp.zeros(4).astype(xp.float32)

# 8x8 → conv 6x6 → pool 3x3 → flattened 4*3*3=36 → output 3
net = CNN(input_shape=(1, 8, 8), output_size=3,
          conv_w=conv_w, conv_b=conv_b, pool_h=2, pool_w=2,
          learning_time=200, learning_rate=0.1)

net.train(X, y)

# predict() returns class indices directly
pred = net.predict(X)
print("First 10 predictions:", pred[:10])
```

---
## 🐢 Performance Notes

Since we're using analytical gradients (backpropagation), this is way faster than numerical differentiation. Each iteration is just one forward pass + one backward pass.

- **Tiny Network (2-4-2)**
  - ~22 parameters. Think XOR problem.
  - Trains in milliseconds. Blink and you'll miss it.

- **Small Network (4-8-3)**
  - ~67 parameters. Think Iris classification.
  - Trains in seconds. Grab a coffee... wait, no, it's already done.

- **Medium Network (784-128-10)**
  - ~100,000 parameters. Think MNIST digits.
  - Trains in minutes on CPU. Time for a proper coffee break. On GPU with `FNN_cupy.py`? Time for a bathroom break instead.

**Important**: This is a teaching tool. For real-world tasks, use GPU-accelerated frameworks like PyTorch or TensorFlow. Your future self will thank you.

**Known Limitations**: The CNN is intentionally minimal — one conv layer, one pooling layer, no padding/stride options on the conv. It's enough to learn how it works, not enough to win ImageNet.

---
## 📄 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute. Go wild.

---

*Made with ❤️ by [linuxlemon256](https://github.com/linuxlemon256)*
