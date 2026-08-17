# 🔥 PyTorch 从零实现 Transformer

> 基于论文 **《Attention Is All You Need》** (Vaswani et al., 2017) 的**纯手写、逐层拆解、逐行注释**的 Transformer 实现。
> 适合所有想从底层**彻底搞懂 Transformer 原理**的初学者。

> 本仓库是基于本人学习与工程实践整理的可复现实践项目，不包含任何公司数据、模型权重、内部系统或生产配置；性能参数均为教学配置或待实测目标，实际结果取决于硬件、数据与版本组合。

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 项目特色

| 特色 | 说明 |
|------|------|
| 🧱 **逐层拆解** | 8 个核心模块从最小单元到完整模型，依赖清晰，学习曲线平滑 |
| 📝 **中文注释详尽** | 每个类、每个函数、每个形状变换都有中文说明 |
| 🔬 **可单独测试** | 每个模块文件都内置 `__main__` 独立测试，验证每一步输出形状 |
| 🚀 **端到端可运行** | `python demo_full_transformer.py` 一键跑完整链路 |
| 📐 **论文对齐** | 默认参数完全对齐论文：N=6, d_model=512, d_ff=2048, h=8 |
| 📦 **标准结构** | 标准 Python 包结构，可直接 `from transformer import ...` 作为库使用 |

---

## 🗂️ 文件结构与学习路线

**建议按下方顺序阅读，每个文件都对应论文的一节：**

```
transformer-from-scratch/
├── transformer/                       # 📦 transformer 核心包
│   ├── __init__.py                    # 统一导出所有关键类
│   ├── embedding_position.py          # 📖 3.4 & 3.5  词嵌入层 + 位置编码 (sin/cos)
│   ├── attention_modules.py           # 📖 3.2 & 3.3  缩放点积注意力 / 多头注意力 / FFN / LayerNorm
│   ├── sublayer_connection.py         # 📖 Figure 1   Add & Norm (残差连接 + 层归一化)
│   ├── encoder_layer.py               # 📖 3.1        单个编码器层 (自注意力 + FFN)
│   ├── encoder_stack.py               # 📖 3.1        编码器栈 (N=6 层堆叠)
│   ├── decoder_layer.py               # 📖 3.1        单个解码器层 (自注意力 + 编解码注意力 + FFN)
│   ├── decoder_stack.py               # 📖 3.1        解码器栈 (N=6 层堆叠)
│   └── output_generator.py            # 📖 3.4        输出层 (Linear + Softmax -> 词汇表概率)
│
├── demo_full_transformer.py           # 🚀 完整模型组装 + 端到端 Demo（入口文件）
├── requirements.txt                   # 依赖清单
├── .gitignore                         # Git 忽略规则
├── LICENSE                            # MIT 协议
└── README.md                          # 本文件
```

### 🧩 组件依赖关系图

```
embedding_position  (词嵌入 + 位置编码)
        │
        ▼
attention_modules   (Attention / MultiHead / FFN / LayerNorm)
        │
        ▼
sublayer_connection (Add + Norm)
        │
        ├───► encoder_layer ──► encoder_stack(N=6) ──┐
        │                                              ├─► demo_full_transformer
        └───► decoder_layer ──► decoder_stack(N=6) ──┘
                                                           │
                                                           ▼
                                                 output_generator
```

---

## ⚙️ 环境要求

- Python **3.8+**
- PyTorch **1.9+**（仅依赖 PyTorch，无需 numpy/matplotlib 即可跑核心模型）

```bash
# 推荐方式（使用 pip）
pip install -r requirements.txt

# 或者直接装 PyTorch
pip install torch>=1.9.0
```

---

## 🚀 快速开始

### 运行测试

```bash
pip install -e ".[dev]"
pytest -q
```

### 1️⃣ 一键跑完整 Transformer（推荐）

```bash
cd transformer-from-scratch
python demo_full_transformer.py
```

你将看到类似输出：

```
======================================================================
  Transformer 完整架构 - 端到端测试
  (论文配置: N=6, d_model=512, d_ff=2048, h=8)
======================================================================

[模型参数] 总参数量: 55,013,224

EncoderDecoder(
  (source_embed): Sequential(
    (0): Embedding( ... )
    (1): PositionalEncoding( ... )
  )
  (encoder): Encoder( ... 6 layers ... )
  (decoder): Decoder( ... 6 layers ... )
  (generator): Generator(
    (linear): Linear(in_features=512, out_features=1000, bias=True)
  )
)

[输入]  source_x 形状: (2, 4)   源序列 (2 句, 每句 4 词)
[输入]  target_y 形状: (2, 4)   目标序列 (2 句, 每句 4 词)

[输出]  result 形状:   (2, 4, 1000)    (batch=2, seq_len=4, vocab_size=1000)
[输出]  数值范围:      [0.0000, 0.0029]
[验证]  第 1 句第 1 词概率和 = 1.000000  (应为 1.000000)
[验证]  第 2 句第 3 词概率和 = 1.000000  (应为 1.000000)

✅ Transformer 端到端测试通过！
```

### 2️⃣ 单独测试每个组件

每个模块都可以单独运行，方便你**断点调试每一步**：

```bash
# 只测词嵌入 + 位置编码
python transformer/embedding_position.py

# 只测多头注意力 / FFN / LayerNorm
python transformer/attention_modules.py

# 只测编码器（6 层堆叠）
python transformer/encoder_stack.py

# 只测解码器（6 层堆叠）
python transformer/decoder_stack.py
```

### 3️⃣ 作为库使用

也可以直接 import，在你自己的项目中使用：

```python
import torch
from transformer import Encoder, EncoderLayer, MultiHeadAttention, FeedForward

# 组装一个自定义编码器
multi_head = MultiHeadAttention(d_model=512, head=8)
ff = FeedForward(d_model=512, d_ff=2048)
encoder_layer = EncoderLayer(d_model=512, self_attn=multi_head, feed_forward=ff)
encoder = Encoder(layer=encoder_layer, N=6)

x = torch.randn(2, 10, 512)     # [batch=2, seq_len=10, d_model=512]
mask = torch.ones(1, 10, 10)
output = encoder(x, mask)       # [2, 10, 512]
print(output.shape)
```

---

## 📐 论文默认参数速查

| 参数 | 含义 | 默认值 |
|------|------|--------|
| N | 编码器/解码器堆叠层数 | **6** |
| d_model | 词向量维度 | **512** |
| d_ff | FFN 中间隐藏层维度 | **2048** |
| h | 多头注意力头数 | **8** |
| d_k | 每个头的维度 (d_model / h) | **64** |
| P_drop | Dropout 概率 | **0.1** |
| ε | LayerNorm 分母小常数 | **1e-6** |

使用自定义参数构建模型：

```python
from demo_full_transformer import make_model

# 小模型示例（适合资源有限的环境）
small_model = make_model(
    src_vocab=5000,
    tgt_vocab=5000,
    N=2,          # 2 层即可快速验证
    d_model=128,
    d_ff=512,
    h=4,
    dropout=0.1,
)
```

---

## 🧠 关键知识点标记

代码中每个模块都用注释标明了对应论文的章节，建议搭配阅读：

- 🎯 **缩放点积注意力为什么除以 sqrt(d_k)？** → 见 `transformer/attention_modules.py` `attention()` 函数
- 🎯 **位置编码为什么用 sin/cos 而不用可学习？** → 见 `transformer/embedding_position.py` `PositionalEncoding`
- 🎯 **多头注意力怎么分头/拼接？** → 见 `transformer/attention_modules.py` `MultiHeadAttention.forward()`
- 🎯 **残差连接 + LayerNorm 的顺序？** → 见 `transformer/sublayer_connection.py`（Post-Norm 实现）
- 🎯 **解码器有哪两种注意力？区别？** → 见 `transformer/decoder_layer.py`（自注意力 vs 编解码注意力）
- 🎯 **target_mask 下三角矩阵的作用？** → 防止看到未来 token，代码中以 mask 参数体现

---

## 🔧 Bug 修复 & 代码改进说明

整理过程中对原始教学代码做了以下工程化改进：

| 问题 | 修复内容 |
|------|----------|
| 模块名以数字开头（Python 语法错误） | 重构为标准 `transformer/` 子包结构，字母开头模块名 |
| `SublayerConnection` 中 `LayerNorm(512)` 写死 | 改为使用传入的 `d_model` 参数，支持任意维度 |
| 原 `dm09_transformer.py` 无用 import | 移除 `unittest.result`、`pyparsing.PositionToken` 等未使用引用 |
| 模块间 `from dmXX import *` 依赖混乱 | 改为相对导入，加上 `__init__.py` 统一导出 |
| 夹杂 matplotlib / numpy 可视化代码 | 移除核心路径上的可视化依赖，只需 torch 即可运行 |
| 缺少参数初始化 | 增加 Xavier 权重初始化，加速模型收敛 |
| 缺少 `requirements.txt` / `.gitignore` / `LICENSE` | 补齐标准 GitHub 项目文件 |
| 缺乏复用性 | 整理为标准 Python 包，可直接 `from transformer import ...` 使用 |

---

## 📚 延伸阅读

1. 📄 **原文论文**：[Attention Is All You Need (NeurIPS 2017)](https://arxiv.org/abs/1706.03762)
2. 🎯 **经典讲解**：[The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
3. 🔬 **Google 官方实现**：[tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)

---

## 📄 License

本项目仅供学习研究使用，基于 **MIT License** 开源。
