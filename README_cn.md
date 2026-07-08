# 🧠 FlexiNN：从数学底层理解神经网络架构

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243)
![License](https://img.shields.io/badge/License-MIT-green)
![状态](https://img.shields.io/badge/Status-Educational-orange)

该神经网络完全由纯 NumPy 实现，没有复杂的深度学习框架，只有直观的数学原理。

> 该方法是用数学模型和原理实现，所以请仅在小型计算或教学上使用，以节约您宝贵的时间。

---
## 📖 目录

- [功能](#-features)
- [安装](#-installation)
- [快速开始](#-quick-start)
- [工作原理](#-how-it-works)
- [API 文档](#-api-documentation)
- [示例](#-examples)
- [异或问题](#1-xor-problem)
- [鸢尾花分类](#2-iris-classification)
- [性能说明](#-performance-notes)
- [许可证](#-license)

---

## ✨ 特性

- **零依赖的黑魔法** — 仅依赖 `numpy`，所有梯度均手动计算。
- **通融性强** — 有许多地方的值可以覆盖，当然直接使用 `self.xxx` 进行覆盖也可以。
- **模块化设计**：内置 `Affine`、`Sigmoid`、`ReLU`、`LeakyReLU`、`Tanh`、`SoftmaxWithLoss`、`Dropout` 等常用层。
- **灵活的网络配置**：支持任意数量的隐藏层，可自定义每层神经元数量。
- **多种激活函数**：支持 `relu`、`leaky_relu`、`sigmoid`、`tanh`，默认 `leaky_relu`。
- **训练控制**：
  - 批量训练（全量或小批量）支持 `epochs` 和 `batch_size`。
  - 学习率衰减（可设置衰减率）。
  - Dropout 正则化（训练时随机失活，预测时缩放）。
- **模型持久化**：支持保存和加载训练好的模型参数。

---

## 📦 安装

克隆仓库并确保已安装 NumPy：

```bash
git clone https://github.com/linuxlemon256/FlexiNN.git
cd FlexiNN
pip install numpy
```

---

## 🚀 快速开始

```python
import numpy as xp
from FlexiNN import FlexiNN

# 生成模拟数据：100 个样本，5 个特征，3 个类别
X = xp.random.randn(100, 5).astype(xp.float32)
y = xp.eye(3)[xp.random.randint(0, 3, 100)].astype(xp.float32) # 独热标签

# 创建网络：输入 5 → 隐藏层 [10] → 输出 3，训练 500 轮
net = FlexiNN(input_size=5, hidden_size=[10], output_size=3,
              learning_time=500, learning_rate=0.1)

# 训练
net.train(X, y)

# 预测
output = net.predict(X)
pred_class = xp.argmax(output, axis=1)
print("前 5 个样本的预测类别：", pred_class[:5])
```

---

## ⚙️ 工作原理

### 1. 前向传播

每一层执行线性变换 `X @ W + b`，然后通过激活函数。除了使用 Softmax 的最后一层，所有中间层都使用指定的激活函数（默认 LeakyReLU）。

前向传播时会保存每一层的中间结果（`affine` 对象、`activation` 对象、`dropout` 对象），用于反向传播时计算梯度。

### 2. 反向传播（误差反向传播）

使用链式法则将损失函数的梯度从输出层反向传播到输入层：

```
dL/dy * dy/dx = dL/dx
```

对于 SoftmaxWithLoss 层，梯度为 `(softmax_output - true_label) / batch_size`。

对于激活层，梯度为 `dout * activation_gradient`：
- Sigmoid: `out * (1 - out)`
- ReLU: `x > 0 ? 1 : 0`
- LeakyReLU: `x > 0 ? 1 : alpha`
- Tanh: `1 - out^2`

对于 Affine 层，梯度为：
- `dW = x.T @ dout`
- `db = sum(dout, axis=0)`
- `dx = dout @ W.T`

### 3. 参数更新

使用标准梯度下降更新参数：

```
W = W - learning_rate * dW
b = b - learning_rate * db
```

---
## 📚 API 文档

### `FlexiNN` 类

#### 初始化参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `input_size` | `int` | - | 输入特征的数量 |
| `hidden_size` | `int` or `list` | - | 隐藏层神经元数量，可为整数或列表 |
| `output_size` | `int` | - | 输出类别的数量 |
| `learning_time` | `int` | `1000` | 总训练迭代次数（批量模式） |
| `learning_rate` | `float` | `0.1` | 梯度下降学习率 |
| `print_every` | `int` | `100` | `train()` 中每隔多少迭代打印一次损失 |
| `activation` | `str` | `"leaky_relu"` | 定义所使用的激活函数 |
| `init_weights` | `float` or `None` | `None` | 权重初始化标准差；设为 `None` 时自动根据激活函数选择 |
| `print_output` | `bool` | `True` | 是否在训练过程中打印损失等信息 |
| `backpropagation` | `str` | `"error-back"` | 定义所使用的反向传播的方法 |
| `decay_rate` | `float` | `None` | 学习率衰减率，范围在 `0.0` 到 `1.0` |
| `dropout_rate` | `float` | `0.0` | 防止过拟合而丢弃的神经元比例 |

#### 核心方法

| 方法 | 返回值 | 描述 |
|------|--------|------|
| `train(x, t, ...)` | `None` | 训练网络，支持批量和小批量训练 |
| `predict(x)` | `xp.ndarray` | 执行前向传播，返回 softmax 概率输出 |
| `accuracy(x, t)` | `float` | 计算分类准确率，`x` 为预测输出，`t` 为真实标签 |
| `save(name=None)` | `None` | 保存模型参数到 `.npz` 文件 |
| `load(name=None)` | `None` | 从 `.npz` 文件加载模型参数 |

#### train 方法参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `x` | `xp.ndarray` | - | 训练数据 |
| `t` | `xp.ndarray` | - | 训练标签（独热编码） |
| `learning_time` | `int` | `None` | 覆盖初始化的训练次数 |
| `learning_rate` | `float` | `None` | 覆盖初始化的学习率 |
| `batch_size` | `int` | `None` | 小批量大小，`None` 为批量训练 |
| `epochs` | `int` | `100` | 训练轮数（小批量模式） |
| `decay_rate` | `float` | `None` | 覆盖初始化的衰减率 |
| `dropout_rate` | `float` | `None` | 覆盖初始化的 Dropout 率 |
| `print_every` | `int` | `None` | 覆盖初始化的打印间隔 |
| `print_output` | `bool` | `None` | 覆盖初始化的打印开关 |

---

## 🧪 示例

### 1. XOR 问题

XOR 是一个经典的线性不可分问题，需要至少一个隐藏层才能学习。

```python
import numpy as xp
from FlexiNN import FlexiNN

X = xp.array([[0,0], [0,1], [1,0], [1,1]]).astype(xp.float32)
y = xp.array([[1,0], [0,1], [0,1], [1,0]]).astype(xp.float32) # 两类 one-hot

# 创建网络：输入 2 维，隐藏层 [4]，输出 2 维
net = FlexiNN(input_size=2, hidden_size=[4], output_size=2,
              learning_time=2000, learning_rate=0.1)

# 训练
net.train(X, y)

# 预测
out = net.predict(X)
predicted_classes = xp.argmax(out, axis=1)
print("预测的类别序列:", predicted_classes)
```

---

### 2. 鸢尾花分类

使用经典的鸢尾花数据集进行训练和评估。

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from FlexiNN import FlexiNN
import numpy as xp

# 加载数据
iris = load_iris()
X = xp.array(iris.data, dtype=xp.float32)
y = xp.array(OneHotEncoder(sparse=False).fit_transform(iris.target.reshape(-1,1)), dtype=xp.float32)

# 创建网络：输入 4 维，隐藏层 [8]，输出 3 个类别
net = FlexiNN(input_size=4, hidden_size=[8], output_size=3,
              learning_time=800, learning_rate=0.1)

# 训练
net.train(X, y)

# 预测并计算准确率
pred = net.predict(X)
accuracy = xp.mean(xp.argmax(pred, axis=1) == xp.array(iris.target))
print(f"鸢尾花数据集上的准确率: {accuracy:.2%}")

# 保存模型
net.save("iris_model")

# 加载模型
net.load("iris_model")
```

---

### 3. 小批量训练与学习率衰减

```python
import numpy as xp
from FlexiNN import FlexiNN

# 生成模拟数据
X = xp.random.randn(500, 10).astype(xp.float32)
y = xp.eye(5)[xp.random.randint(0, 5, 500)].astype(xp.float32)

# 创建网络：输入 10 → 隐藏层 [32, 16] → 输出 5
net = FlexiNN(input_size=10, hidden_size=[32, 16], output_size=5,
              learning_rate=0.1, dropout_rate=0.3)

# 小批量训练，学习率每轮衰减 0.99
net.train(X, y, epochs=200, batch_size=32, decay_rate=0.99)

# 预测
output = net.predict(X)
accuracy = net.accuracy(output, y)
print(f"准确率: {accuracy:.4f}")
```

---
## 🐢 性能描述

由于使用了解析梯度（误差反向传播），该实现的计算效率远高于数值微分方法。每次训练迭代只需一次前向传播和一次反向传播。

- **微型网络 (2-4-2)**
  - 例如 XOR 问题，大约有 22 个参数。
  - 训练可以在零点几秒内完成。

- **小型网络 (4-8-3)**
  - 例如鸢尾花分类，大约有 67 个参数。
  - 训练可以在几秒内完成。

- **中等网络 (784-128-10)**  
  - 例如 MNIST 手写数字识别，参数超过 100,000 个。
  - 训练需要几至几十分钟完成。

**注意**：此代码库旨在作为教学工具，清楚展示神经网络的内部机制。对于任何实际规模的任务，强烈建议改用基于 GPU 加速的框架，如 PyTorch 或 TensorFlow。

**已知限制**：`Convolution` 和 `Pooling` 类目前为占位符（`pass`），尚未实现卷积和池化功能。

---
## 📄 许可证

本项目遵循 **MIT 许可证**。您可以自由使用、修改和分发该代码。更多详情，请参阅项目根目录中的 [LICENSE](LICENSE) 文件。
