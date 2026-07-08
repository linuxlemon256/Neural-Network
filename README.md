# 🧠 FlexiNN: Neural Networks from Scratch (But Make It Fun)

**[🇨🇳 中文版本](README_cn.md)** | **[🇬🇧 English](README.md)**

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Educational-orange)

This neural network is built **purely with NumPy** — no fancy deep learning frameworks, just good old math and elbow grease.

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
- [Performance Notes](#-performance-notes)
- [License](#-license)

---

## ✨ Features

- **Zero-Dependency Dark Magic** — Only `numpy`, all gradients calculated by hand. No `import torch` here, cowboy.
- **Super Flexible** — Most settings can be overridden on the fly. Want to change learning rate mid-training? Go wild.
- **Modular Design**: Built-in layers like `Affine`, `Sigmoid`, `ReLU`, `LeakyReLU`, `Tanh`, `SoftmaxWithLoss`, and `Dropout`.
- **Customizable Network**: Any number of hidden layers, any number of neurons per layer. Mix and match like a neural network DJ.
- **Activation Functions**: `relu`, `leaky_relu`, `sigmoid`, `tanh`. Default is `leaky_relu` (because dead neurons are sad).
- **Training Controls**:
  - Full batch or mini-batch training with `epochs` and `batch_size`.
  - Learning rate decay (for when your network needs to chill out).
  - Dropout regularization (randomly fire some neurons, keep things spicy).
- **Model Persistence**: Save and load trained models. Because training takes time, and nobody wants to retrain from scratch.

---

## 📦 Installation

Clone the repo and make sure NumPy is installed:

```bash
git clone https://github.com/linuxlemon256/FlexiNN.git
cd FlexiNN
pip install numpy
```

---

## 🚀 Quick Start

```python
import numpy as xp
from FlexiNN import FlexiNN

# Generate fake data: 100 samples, 5 features, 3 classes
X = xp.random.randn(100, 5).astype(xp.float32)
y = xp.eye(3)[xp.random.randint(0, 3, 100)].astype(xp.float32) # One-hot labels

# Create network: Input 5 → Hidden [10] → Output 3, train for 500 iterations
net = FlexiNN(input_size=5, hidden_size=[10], output_size=3,
              learning_time=500, learning_rate=0.1)

# Train that bad boy
net.train(X, y)

# Predict some stuff
output = net.predict(X)
pred_class = xp.argmax(output, axis=1)
print("First 5 predictions:", pred_class[:5])
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

### 3. Parameter Update

Good old gradient descent:

```
W = W - learning_rate * dW
b = b - learning_rate * db
```

Simple, elegant, and it works (most of the time).

---
## 📚 API Documentation

### `FlexiNN` Class

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | `int` | - | Number of input features |
| `hidden_size` | `int` or `list` | - | Number of neurons in hidden layers |
| `output_size` | `int` | - | Number of output classes |
| `learning_time` | `int` | `1000` | Total training iterations (full batch mode) |
| `learning_rate` | `float` | `0.1` | Gradient descent learning rate |
| `print_every` | `int` | `100` | Print loss every N iterations |
| `activation` | `str` | `"leaky_relu"` | Activation function to use |
| `init_weights` | `float` or `None` | `None` | Weight initialization scaling |
| `print_output` | `bool` | `True` | Whether to print training progress |
| `backpropagation` | `str` | `"error-back"` | Backpropagation method |
| `decay_rate` | `float` | `None` | Learning rate decay rate (0.0 to 1.0) |
| `dropout_rate` | `float` | `0.0` | Dropout probability |

#### Core Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `train(x, t, ...)` | `None` | Train the network |
| `predict(x)` | `xp.ndarray` | Forward pass, returns softmax probabilities |
| `accuracy(x, t)` | `float` | Calculate classification accuracy |
| `save(name=None)` | `None` | Save model to `.npz` file |
| `load(name=None)` | `None` | Load model from `.npz` file |

#### train() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `xp.ndarray` | - | Training data |
| `t` | `xp.ndarray` | - | Training labels (one-hot encoded) |
| `learning_time` | `int` | `None` | Override training iterations |
| `learning_rate` | `float` | `None` | Override learning rate |
| `batch_size` | `int` | `None` | Mini-batch size |
| `epochs` | `int` | `100` | Number of epochs (mini-batch mode) |
| `decay_rate` | `float` | `None` | Override decay rate |
| `dropout_rate` | `float` | `None` | Override dropout rate |
| `print_every` | `int` | `None` | Override print interval |
| `print_output` | `bool` | `None` | Override print toggle |

---

## 🧪 Examples

### 1. XOR Problem

XOR is the classic "I can't do this with a straight line" problem. You need at least one hidden layer to solve it.

```python
import numpy as xp
from FlexiNN import FlexiNN

X = xp.array([[0,0], [0,1], [1,0], [1,1]]).astype(xp.float32)
y = xp.array([[1,0], [0,1], [0,1], [1,0]]).astype(xp.float32) # Two-class one-hot

# Create network: Input 2 → Hidden [4] → Output 2
net = FlexiNN(input_size=2, hidden_size=[4], output_size=2,
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
from FlexiNN import FlexiNN
import numpy as xp

# Load data
iris = load_iris()
X = xp.array(iris.data, dtype=xp.float32)
y = xp.array(OneHotEncoder(sparse=False).fit_transform(iris.target.reshape(-1,1)), dtype=xp.float32)

# Create network: Input 4 → Hidden [8] → Output 3
net = FlexiNN(input_size=4, hidden_size=[8], output_size=3,
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
from FlexiNN import FlexiNN

# Generate fake data
X = xp.random.randn(500, 10).astype(xp.float32)
y = xp.eye(5)[xp.random.randint(0, 5, 500)].astype(xp.float32)

# Create network: Input 10 → Hidden [32, 16] → Output 5
net = FlexiNN(input_size=10, hidden_size=[32, 16], output_size=5,
              learning_rate=0.1, dropout_rate=0.3)

# Train with mini-batches, decay learning rate by 0.99 each epoch
net.train(X, y, epochs=200, batch_size=32, decay_rate=0.99)

# Predict
output = net.predict(X)
accuracy = net.accuracy(output, y)
print(f"Accuracy: {accuracy:.4f}")
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
  - Trains in minutes. Time for a proper coffee break.

**Important**: This is a teaching tool. For real-world tasks, use GPU-accelerated frameworks like PyTorch or TensorFlow. Your future self will thank you.

**Known Limitations**: `Convolution` and `Pooling` classes are just placeholders (`pass`). CNNs coming soon... maybe.

---
## 📄 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute. Go wild.

---

*Made with ❤️ by [linuxlemon256](https://github.com/linuxlemon256)*
