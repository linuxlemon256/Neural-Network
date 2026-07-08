# 🧠 FlexiNN：从零搭建神经网络（但要有趣）

**[🇨🇳 中文版本](README_cn.md)** | **[🇬🇧 English](README.md)**

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243)
![License](https://img.shields.io/badge/License-MIT-green)
![状态](https://img.shields.io/badge/状态-教学向-orange)

这个神经网络完全用 **纯 NumPy** 实现 — 没有花里胡哨的深度学习框架，只有朴素的数学和满满的诚意。

> 友情提示：这只是个教学项目。如果你想用它训练 GPT-4 或者预测股市，建议出门左转 PyTorch。你的时间很宝贵，别浪费在这。

---
## 📖 目录

- [功能](#-功能)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [工作原理](#-工作原理)
- [API 文档](#-api-documentation)
- [示例](#-示例)
- [异或问题](#1-异或问题)
- [鸢尾花分类](#2-鸢尾花分类)
- [性能说明](#-性能说明)
- [许可证](#-许可证)

---

## ✨ 功能

- **零依赖黑魔法** — 只靠 `numpy`，所有梯度手动计算。调包侠退散！
- **极度灵活** — 大部分参数都能随时覆盖。训练到一半想改学习率？没问题，随便改。
- **模块化设计**：内置 `Affine`、`Sigmoid`、`ReLU`、`LeakyReLU`、`Tanh`、`SoftmaxWithLoss`、`Dropout` 等常用层。
- **自定义网络**：任意层数的隐藏层，每层任意神经元数量。像搭积木一样搭神经网络。
- **激活函数**：`relu`、`leaky_relu`、`sigmoid`、`tanh`。默认 `leaky_relu`（因为神经元死了就不好玩了）。
- **训练控制**：
  - 全量训练或小批量训练，支持 `epochs` 和 `batch_size`。
  - 学习率衰减（让网络越学越冷静）。
  - Dropout 正则化（随机干掉一些神经元，保持网络清醒）。
- **模型持久化**：保存和加载训练好的模型。毕竟训练要花时间，谁也不想从头再来。

---

## 📦 安装

克隆仓库，确保安装了 NumPy：

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

# 生成假数据：100 个样本，5 个特征，3 个类别
X = xp.random.randn(100, 5).astype(xp.float32)
y = xp.eye(3)[xp.random.randint(0, 3, 100)].astype(xp.float32) # 独热编码标签

# 创建网络：输入 5 → 隐藏层 [10] → 输出 3，训练 500 轮
net = FlexiNN(input_size=5, hidden_size=[10], output_size=3,
              learning_time=500, learning_rate=0.1)

# 开始训练
net.train(X, y)

# 预测一下
output = net.predict(X)
pred_class = xp.argmax(output, axis=1)
print("前 5 个预测结果：", pred_class[:5])
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

### 3. 参数更新

经典的梯度下降：

```
W = W - learning_rate * dW
b = b - learning_rate * db
```

简单、优雅、大部分时候管用（玄学时刻除外）。

---
## 📚 API 文档

### `FlexiNN` 类

#### 初始化参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `input_size` | `int` | - | 输入特征数量 |
| `hidden_size` | `int` or `list` | - | 隐藏层神经元数量 |
| `output_size` | `int` | - | 输出类别数量 |
| `learning_time` | `int` | `1000` | 训练迭代次数（批量模式） |
| `learning_rate` | `float` | `0.1` | 学习率 |
| `print_every` | `int` | `100` | 每 N 轮打印一次损失 |
| `activation` | `str` | `"leaky_relu"` | 激活函数 |
| `init_weights` | `float` or `None` | `None` | 权重初始化缩放因子 |
| `print_output` | `bool` | `True` | 是否打印训练过程 |
| `backpropagation` | `str` | `"error-back"` | 反向传播方式 |
| `decay_rate` | `float` | `None` | 学习率衰减率（0.0 到 1.0） |
| `dropout_rate` | `float` | `0.0` | Dropout 概率 |

#### 核心方法

| 方法 | 返回值 | 描述 |
|------|--------|------|
| `train(x, t, ...)` | `None` | 训练网络 |
| `predict(x)` | `xp.ndarray` | 前向传播，返回 softmax 概率 |
| `accuracy(x, t)` | `float` | 计算分类准确率 |
| `save(name=None)` | `None` | 保存模型到 `.npz` 文件 |
| `load(name=None)` | `None` | 从 `.npz` 文件加载模型 |

#### train() 方法参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `x` | `xp.ndarray` | - | 训练数据 |
| `t` | `xp.ndarray` | - | 训练标签（独热编码） |
| `learning_time` | `int` | `None` | 覆盖训练次数 |
| `learning_rate` | `float` | `None` | 覆盖学习率 |
| `batch_size` | `int` | `None` | 小批量大小 |
| `epochs` | `int` | `100` | 训练轮数（小批量模式） |
| `decay_rate` | `float` | `None` | 覆盖衰减率 |
| `dropout_rate` | `float` | `None` | 覆盖 Dropout 率 |
| `print_every` | `int` | `None` | 覆盖打印间隔 |
| `print_output` | `bool` | `None` | 覆盖打印开关 |

---

## 🧪 示例

### 1. 异或问题

异或问题是经典的"一根直线搞不定"的问题，至少需要一个隐藏层才能解决。

```python
import numpy as xp
from FlexiNN import FlexiNN

X = xp.array([[0,0], [0,1], [1,0], [1,1]]).astype(xp.float32)
y = xp.array([[1,0], [0,1], [0,1], [1,0]]).astype(xp.float32) # 两类独热编码

# 创建网络：输入 2 → 隐藏层 [4] → 输出 2
net = FlexiNN(input_size=2, hidden_size=[4], output_size=2,
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
from FlexiNN import FlexiNN
import numpy as xp

# 加载数据
iris = load_iris()
X = xp.array(iris.data, dtype=xp.float32)
y = xp.array(OneHotEncoder(sparse=False).fit_transform(iris.target.reshape(-1,1)), dtype=xp.float32)

# 创建网络：输入 4 → 隐藏层 [8] → 输出 3
net = FlexiNN(input_size=4, hidden_size=[8], output_size=3,
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
from FlexiNN import FlexiNN

# 生成假数据
X = xp.random.randn(500, 10).astype(xp.float32)
y = xp.eye(5)[xp.random.randint(0, 5, 500)].astype(xp.float32)

# 创建网络：输入 10 → 隐藏层 [32, 16] → 输出 5
net = FlexiNN(input_size=10, hidden_size=[32, 16], output_size=5,
              learning_rate=0.1, dropout_rate=0.3)

# 小批量训练，每轮学习率衰减 0.99
net.train(X, y, epochs=200, batch_size=32, decay_rate=0.99)

# 预测
output = net.predict(X)
accuracy = net.accuracy(output, y)
print(f"准确率: {accuracy:.4f}")
```

---
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
  - 分钟级完成。终于可以喝杯正经咖啡了。

**重要提示**：这是个教学工具。实际项目请用 PyTorch 或 TensorFlow 这种带 GPU 加速的框架。你的未来会感谢你。

**已知限制**：`Convolution` 和 `Pooling` 类目前只是占位符（`pass`）。CNN 功能正在路上... 大概。

---
## 📄 许可证

本项目遵循 **MIT 许可证**。随便用、随便改、随便分发。开心就好。

---

*Made with ❤️ by [linuxlemon256](https://github.com/linuxlemon256)*
