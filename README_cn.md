# 🧠 FlexiNN：从零搭建神经网络（但要有趣）

**[🇨🇳 中文版本](README_cn.md)** | **[🇬🇧 English](README.md)**

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243)
![CuPy](https://img.shields.io/badge/CuPy-可选-003F87)
![License](https://img.shields.io/badge/License-MIT-green)
![状态](https://img.shields.io/badge/状态-教学向-orange)

这个神经网络完全用 **纯 NumPy** 实现 — 没有花里胡哨的深度学习框架，只有朴素的数学和满满的诚意。**可选 CuPy** 获得 GPU 加速。

> 友情提示：这只是个教学项目。如果你想用它训练 GPT-4 或者预测股市，建议出门左转 PyTorch。你的时间很宝贵，别浪费在这。

---
## 📖 目录

- [功能](#-功能)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [工作原理](#-工作原理)
- [API 文档](#-api-文档)
- [示例](#-示例)
- [异或问题](#1-异或问题)
- [鸢尾花分类](#2-鸢尾花分类)
- [小批量训练](#3-小批量训练与学习率衰减)
- [CNN 图像分类](#4-cnn-图像分类)
- [MNIST 基准](#5-mnist-基准500-个-epoch-内达到-985-的目标)
- [性能说明](#-性能说明)
- [许可证](#-许可证)

---

## ✨ 功能

- **零依赖黑魔法** — 只靠 `numpy`，所有梯度手动计算。调包侠退散！
- **GPU 选项** — 换成 `FNN_cupy.py` 就能在 CUDA 上训练。同样的代码，换个后端（`import cupy as xp`）。
- **极度灵活** — 大部分参数都能随时覆盖。训练到一半想改学习率？没问题，随便改。
- **模块化设计**：内置 `Affine`、`BatchNorm`、`Sigmoid`、`ReLU`、`LeakyReLU`、`Tanh`、`SoftmaxWithLoss`、`Dropout`、`Convolution`、`Pooling` 等常用层。
- **完整可用的 CNN**：`Convolution` + `Pooling` 用 `im2col`/`col2im` 实现，支持最大池化和平均池化。不再是占位符了，宝贝。
- **自定义网络**：任意层数的隐藏层，每层任意神经元数量。像搭积木一样搭神经网络。
- **激活函数**：`relu`、`leaky_relu`、`sigmoid`、`tanh`。默认 `leaky_relu`（因为神经元死了就不好玩了）。
- **智能权重初始化**：ReLU 系激活用 He 初始化（`sqrt(2/fan_in)`），sigmoid/tanh 用 Xavier 风格，也可以自己传 `init_weights`。
- **数值稳定性**：Sigmoid 输入裁剪到 `[-500, 500]`，softmax 减去最大值，损失加 `1e-12` 保护。告别 `NaN` 惊喜。
- **训练控制**：
  - 全量训练或小批量训练，支持 `epochs` 和 `batch_size`。
  - 学习率衰减（让网络越学越冷静）。
  - Dropout 正则化（随机干掉一些神经元，保持网络清醒）。
  - 可选 Adam 优化器与 BatchNorm，适合更深的 MLP。
- **模型持久化**：保存和加载训练好的模型到 `.npz`。加载时会自动重建层结构。

---

## 📦 安装

克隆仓库，确保安装了 NumPy：

```bash
git clone https://github.com/linuxlemon256/Neural-Network.git
cd Neural-Network
pip install numpy
```

想要 GPU 加速？安装与你的 CUDA 版本匹配的 CuPy：

```bash
pip install cupy-cuda11x   # 或 cupy-cuda12x，详见 https://cupy.dev
```

---

## 🚀 快速开始

```python
import numpy as xp
from FNN_numpy import MLP   # 或：from FlexiNN import MLP

# 生成假数据：100 个样本，5 个特征，3 个类别
X = xp.random.randn(100, 5).astype(xp.float32)
y = xp.eye(3)[xp.random.randint(0, 3, 100)].astype(xp.float32) # 独热编码标签

# 创建网络：输入 5 → 隐藏层 [10] → 输出 3，训练 500 轮
net = MLP(input_size=5, hidden_size=[10], output_size=3,
          learning_time=500, learning_rate=0.1)

# 开始训练
net.train(X, y)

# 预测一下
output = net.predict(X)
pred_class = xp.argmax(output, axis=1)
print("前 5 个预测结果：", pred_class[:5])
```

GPU 版本？改一行 import 就行：

```python
from FNN_cupy import MLP   # 其余代码完全一样
```

---

## ⚙️ 工作原理

### 1. 前向传播

每一层执行 `X @ W + b`，然后过激活函数。最后一层用 Softmax（概率输出才是正经事），中间层用你选的激活函数（默认 LeakyReLU，因为 ReLU 对负数太残忍了）。

前向传播时会保存中间结果，方便后面算梯度。就像给反向传播留了一把钥匙。

### 2. 反向传播（误差反向传播）

用链式法则把损失函数的梯度从输出层传到输入层：

```
dL/dy * dy/dx = dL/dx
```

对于 SoftmaxWithLoss 层，梯度是 `(softmax输出 - 真实标签) / batch_size`。

对于激活层，梯度是：
- Sigmoid: `out * (1 - out)` — S 形曲线的导数
- ReLU: `x > 0 ? 1 : 0` — 非黑即白，简单粗暴
- LeakyReLU: `x > 0 ? 1 : alpha` — 像 ReLU 但给负数留了条活路
- Tanh: `1 - out^2` — 把东西压到 -1 到 1 之间

对于 Affine 层，梯度是：
- `dW = x.T @ dout` — 转置一下就行
- `db = sum(dout, axis=0)` — 直接求和
- `dx = dout @ W.T` — 反向乘回去

### 3. 卷积与池化

卷积用的是经典套路：`im2col` 把每个卷积窗口展开成一列，卷积就变成了一次大矩阵乘法。`col2im` 把梯度放回去（放心，是 scatter-add，不偷懒）。

池化默认最大池化（也可选平均池化），并记住是谁赢的（`mask`），反向传播时梯度只回传给赢家。

### 4. 参数更新

经典的梯度下降：

```
W = W - learning_rate * dW
b = b - learning_rate * db
```

简单、优雅、大部分时候管用（玄学时刻除外）。

---
## 📚 API 文档

### `MLP` 类（全连接网络）

#### 初始化参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `input_size` | `int` | - | 输入特征数量 |
| `hidden_size` | `int` or `list` | - | 隐藏层神经元数量（`[32, 16]` 表示两层隐藏层） |
| `output_size` | `int` | - | 输出类别数量 |
| `learning_rate` | `float` | `0.1` | 学习率 |
| `learning_time` | `int` | `1000` | 训练迭代次数（全量模式） |
| `init_weights` | `float` or `None` | `None` | 权重初始化缩放因子（`None` = ReLU 系用 He / sigmoid 用 Xavier） |
| `print_every` | `int` | `100` | 每 N 轮打印一次损失 |
| `print_output` | `bool` | `True` | 是否打印训练过程 |
| `activation` | `str` | `"leaky_relu"` | 激活函数 |
| `backpropagation` | `str` | `"error-back"` | 反向传播方式 |
| `decay_rate` | `float` | `None` | 学习率衰减率（0.0 到 1.0） |
| `dropout_rate` | `float` | `0.0` | Dropout 概率（作用于隐藏层） |
| `optimizer` | `str` | `"sgd"` | `"sgd"` 或带偏差修正的 `"adam"` |
| `batch_norm` | `bool` | `False` | 训练时对每个隐藏层的仿射输出做归一化 |
| `seed` | `int` 或 `None` | `None` | 初始化、打乱和 Dropout 使用的随机种子 |

#### 核心方法

| 方法 | 返回值 | 描述 |
|------|--------|------|
| `train(x, t, ...)` | `None` | 训练网络 |
| `predict(x)` | `xp.ndarray` | 前向传播，返回 softmax 概率 |
| `accuracy(x, t)` | `float` | 计算分类准确率（传入预测结果和标签） |
| `save(name=None)` | `None` | 保存模型到 `.npz` 文件 |
| `load(name=None)` | `None` | 从 `.npz` 文件加载模型（自动重建层数） |

#### train() 方法参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `x` | `xp.ndarray` | - | 训练数据 |
| `t` | `xp.ndarray` | - | 训练标签（独热编码） |
| `learning_time` | `int` | `None` | 覆盖训练次数 |
| `learning_rate` | `float` | `None` | 覆盖学习率 |
| `batch_size` | `int` | `None` | 小批量大小（`None` = 全量） |
| `epochs` | `int` | `100` | 训练轮数（小批量模式） |
| `decay_rate` | `float` | `None` | 覆盖衰减率 |
| `dropout_rate` | `float` | `None` | 覆盖 Dropout 率 |
| `print_every` | `int` | `None` | 覆盖打印间隔 |
| `print_output` | `bool` | `None` | 覆盖打印开关 |

### `CNN` 类（卷积网络）

#### 初始化参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `input_shape` | `tuple` | - | 输入图像形状 `(C, H, W)` |
| `output_size` | `int` | - | 输出类别数量 |
| `conv_w` | `xp.ndarray` | - | 卷积核，形状 `(FN, C, FH, FW)` |
| `conv_b` | `xp.ndarray` | - | 卷积偏置，形状 `(FN,)` |
| `pool_h` / `pool_w` | `int` | - | 池化窗口大小 |
| `activation` | `str` | `"relu"` | 卷积后的激活函数 |
| `learning_rate` | `float` | `0.1` | 学习率 |
| `learning_time` | `int` | `1000` | 训练迭代次数 |
| `print_every` | `int` | `100` | 每 N 轮打印一次损失 |
| `print_output` | `bool` | `True` | 是否打印训练过程 |

#### 核心方法

| 方法 | 返回值 | 描述 |
|------|--------|------|
| `train(x, t, ...)` | `None` | 训练网络 |
| `predict(x)` | `xp.ndarray` | 前向传播，**直接返回类别索引**（`argmax`） |
| `accuracy(x, t)` | `float` | 计算分类准确率 |

**注意**：CNN 目前是 `stride=1, pad=0` 的卷积 + 一层池化 + 全连接输出层。最后一层全连接的维度会自动计算为 `卷积核数 * pool_H * pool_W`。

---

## 🧪 示例

### 1. 异或问题

异或问题是经典的"一根直线搞不定"的问题，至少需要一个隐藏层才能解决。

```python
import numpy as xp
from FNN_numpy import MLP

X = xp.array([[0,0], [0,1], [1,0], [1,1]]).astype(xp.float32)
y = xp.array([[1,0], [0,1], [0,1], [1,0]]).astype(xp.float32) # 两类独热编码

# 创建网络：输入 2 → 隐藏层 [4] → 输出 2
net = MLP(input_size=2, hidden_size=[4], output_size=2,
          learning_time=2000, learning_rate=0.1)

# 训练
net.train(X, y)

# 预测
out = net.predict(X)
predicted_classes = xp.argmax(out, axis=1)
print("预测类别：", predicted_classes)
```

---

### 2. 鸢尾花分类

机器学习界的"Hello World"数据集。

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from FNN_numpy import MLP
import numpy as xp

# 加载数据
iris = load_iris()
X = xp.array(iris.data, dtype=xp.float32)
y = xp.array(OneHotEncoder(sparse=False).fit_transform(iris.target.reshape(-1,1)), dtype=xp.float32)

# 创建网络：输入 4 → 隐藏层 [8] → 输出 3
net = MLP(input_size=4, hidden_size=[8], output_size=3,
          learning_time=800, learning_rate=0.1)

# 训练
net.train(X, y)

# 预测并计算准确率
pred = net.predict(X)
accuracy = xp.mean(xp.argmax(pred, axis=1) == xp.array(iris.target))
print(f"鸢尾花准确率: {accuracy:.2%}")

# 保存模型
net.save("iris_model")

# 加载模型
net.load("iris_model")
```

---

### 3. 小批量训练与学习率衰减

当你的数据集比面包箱还大时。

```python
import numpy as xp
from FNN_numpy import MLP

# 生成假数据
X = xp.random.randn(500, 10).astype(xp.float32)
y = xp.eye(5)[xp.random.randint(0, 5, 500)].astype(xp.float32)

# 创建网络：输入 10 → 隐藏层 [32, 16] → 输出 5
net = MLP(input_size=10, hidden_size=[32, 16], output_size=5,
          learning_rate=0.1, dropout_rate=0.3)

# 小批量训练，每轮学习率衰减 0.99
net.train(X, y, epochs=200, batch_size=32, decay_rate=0.99)

# 预测
output = net.predict(X)
accuracy = net.accuracy(output, y)
print(f"准确率: {accuracy:.4f}")
```

---

### 4. CNN 图像分类

简化版 MNIST：随机 8x8 灰度图。卷积层 + ReLU + 2x2 最大池化 + 全连接层。

```python
import numpy as xp
from FNN_numpy import CNN

# 假图像：50 个样本，1 通道，8x8
X = xp.random.randn(50, 1, 8, 8).astype(xp.float32)
y = xp.eye(3)[xp.random.randint(0, 3, 50)].astype(xp.float32)

# 4 个卷积核，1 个输入通道，3x3 大小
conv_w = xp.random.randn(4, 1, 3, 3).astype(xp.float32) * 0.1
conv_b = xp.zeros(4).astype(xp.float32)

# 8x8 → 卷积 6x6 → 池化 3x3 → 展平 4*3*3=36 → 输出 3
net = CNN(input_shape=(1, 8, 8), output_size=3,
          conv_w=conv_w, conv_b=conv_b, pool_h=2, pool_w=2,
          learning_time=200, learning_rate=0.1)

net.train(X, y)

# predict() 直接返回类别索引
pred = net.predict(X)
print("前 10 个预测结果：", pred[:10])
```

---
### 5. MNIST 基准：500 个 epoch 内达到 98.5% 的目标

该基准仍只使用 NumPy：`784 → 1024 → 512 → 256 → 10` 的 MLP，配合
BatchNorm、Adam、Dropout 及训练阶段的小幅整数平移；推理时会平均五个平移视图的
概率。原始训练集固定划分为 55,000 个训练样本和 5,000 个验证样本；测试集只有在
显式要求时才会读取并评估。

```bash
python examples/train_mnist.py --epochs 500 --evaluate-test
```

脚本会在验证准确率到达 98.9% 时（或最多 500 个 epoch）停止，恢复最佳验证检查点，
然后只评估一次官方测试集。最终测试准确率低于 98.5% 时会以失败状态退出。

## 🐢 性能说明

因为用了解析梯度（反向传播），比数值微分快多了。每次迭代就一次前向传播 + 一次反向传播。

- **微型网络 (2-4-2)**
  - 约 22 个参数。比如异或问题。
  - 毫秒级完成。眨个眼就没了。

- **小型网络 (4-8-3)**
  - 约 67 个参数。比如鸢尾花分类。
  - 秒级完成。想喝杯咖啡？算了，已经好了。

- **中等网络 (784-128-10)**
  - 约 10 万个参数。比如 MNIST 手写数字。
  - CPU 分钟级完成。终于可以喝杯正经咖啡了。换成 `FNN_cupy.py` 上 GPU？那就只够上个厕所了。

**重要提示**：这是个教学工具。实际项目请用 PyTorch 或 TensorFlow 这种带 GPU 加速的框架。你的未来会感谢你。

**已知限制**：CNN 是刻意保持极简的——一层卷积、一层池化，卷积没有 padding/stride 选项。用来理解原理足够，拿来赢 ImageNet 就差点意思。

---
## 📄 许可证

本项目遵循 **MIT 许可证**。随便用、随便改、随便分发。开心就好。

---

*Made with ❤️ by [linuxlemon256](https://github.com/linuxlemon256)*
