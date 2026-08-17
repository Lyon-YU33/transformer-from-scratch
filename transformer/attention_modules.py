"""
Transformer 注意力模块与核心组件：
    - attention(): 缩放点积注意力
    - clones(): 克隆 N 个相同模块
    - MultiHeadAttention: 多头注意力机制
    - FeedForward: 前馈全连接网络 (FFN)
    - LayerNorm: 层归一化 (Layer Normalization)

对应论文《Attention Is All You Need》3.2 / 3.3 / 5.4 节
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding_position import (
    Embedding,
    PositionalEncoding,
    _get_position_demo_input as _use_position,
)


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
):
    """
    缩放点积注意力 (Scaled Dot-Product Attention)。
    公式：Attention(Q, K, V) = Softmax(Q * K^T / sqrt(d_k)) * V

    Args:
        query:  [batch_size, (head), seq_len, d_k]
        key:    [batch_size, (head), seq_len, d_k]
        value:  [batch_size, (head), seq_len, d_k]
        mask:   掩码张量，0 表示遮挡位置；形状通常与 scores 匹配
        dropout: 可选的随机失活层
    Returns:
        (注意力输出, 注意力权重分布)
    """
    d_k = query.size(-1)

    # 1. 计算原始注意力分数：Q * K^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 掩码处理：将遮挡位置填充为极小值，softmax 后权重趋近于 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. 在最后一维做 softmax，得到注意力权重
    p_attn = F.softmax(scores, dim=-1)

    # 4. 可选 dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 5. 权重加权求和得到最终输出
    return torch.matmul(p_attn, value), p_attn


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """
    克隆函数：深拷贝 N 个相同的模块，返回 ModuleList。
    用于堆叠 N 个编码器层/解码器层、创建 4 个线性投影层等。

    Args:
        module: 待克隆的模块
        N: 克隆次数
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)。
    将 d_model 维分成 h 个头，分别在小维度上做注意力，再拼接，增强表达能力。
    """

    def __init__(self, embed_dim: int, head: int, dropout_p: float = 0.1):
        """
        Args:
            embed_dim: 词向量维度 d_model，必须能被 head 整除
            head: 头数 h，论文默认 8
            dropout_p: 随机失活概率
        """
        super().__init__()
        assert embed_dim % head == 0, "embed_dim 必须能被 head 整除"
        self.d_k = embed_dim // head
        self.head = head

        # 4 个线性层：前 3 个分别投影 Q/K/V，最后 1 个做输出融合
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)
        self.dropout = nn.Dropout(p=dropout_p)
        self.attn = None  # 保存注意力权重，可用于可视化

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 增加 head 维度，便于广播
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        # 1. 线性投影 + 维度变形：[batch, seq_len, d_model] -> [batch, head, seq_len, d_k]
        query, key, value = [
            model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears, (query, key, value))
        ]

        # 2. 并行计算多头注意力
        x, self.attn = attention(query, key, value, mask, self.dropout)

        # 3. 拼接多头结果：[batch, head, seq_len, d_k] -> [batch, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 4. 最后一个线性层输出融合结果
        return self.linears[-1](x)


class FeedForward(nn.Module):
    """
    位置-wise 前馈全连接网络 (Position-wise Feed-Forward Networks)。
    公式：FFN(x) = max(0, x*W1 + b1)*W2 + b2
    两层 MLP，中间升维到 d_ff（论文默认 2048），再降回 d_model。
    """

    def __init__(self, d_model: int, d_ff: int, dropout_p: float = 0.1):
        """
        Args:
            d_model: 输入/输出维度
            d_ff: 中间隐藏层维度，论文默认 2048
            dropout_p: 随机失活概率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.dropout(F.relu(x))
        x = self.linear2(x)
        return x


class LayerNorm(nn.Module):
    """
    层归一化 (Layer Normalization)。
    对最后一维（词向量维度）做标准化，并加上可学习的缩放/平移参数。
    公式：LN(x) = a * (x - mean) / (std + eps) + b
    """

    def __init__(self, features: int, eps: float = 1e-6):
        """
        Args:
            features: 词向量维度 d_model
            eps: 小常数，防止分母为 0
        """
        super().__init__()
        self.a = nn.Parameter(torch.ones(features))   # 缩放参数 gamma
        self.b = nn.Parameter(torch.zeros(features))  # 平移参数 beta
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(-1, keepdim=True)
        x_std = x.std(-1, keepdim=True)
        return self.a * (x - x_mean) / (x_std + self.eps) + self.b


# =============== 独立测试代码 ===============
def _test_attention():
    position_x = _use_position()  # [2, 4, 512]
    q = k = v = position_x
    result, p_attn = attention(q, k, v)
    print(f"Attention 输出形状: {result.shape}")     # [2, 4, 512]
    print(f"Attention 权重形状: {p_attn.shape}")     # [2, 4, 4]


def _test_multi_head_attention():
    mha = MultiHeadAttention(512, 8)
    x = _use_position()
    mask = torch.zeros(1, 4, 4)
    result = mha(x, x, x, mask)
    print(f"Multi-Head Attention 输出形状: {result.shape}")  # [2, 4, 512]


def _test_feed_forward():
    ff = FeedForward(512, 2048)
    x = torch.randn(2, 4, 512)
    result = ff(x)
    print(f"FeedForward 输出形状: {result.shape}")  # [2, 4, 512]


def _test_layer_norm():
    ln = LayerNorm(512)
    x = torch.randn(2, 4, 512)
    result = ln(x)
    print(f"LayerNorm 输出形状: {result.shape}")   # [2, 4, 512]


if __name__ == '__main__':
    _test_attention()
    _test_multi_head_attention()
    _test_feed_forward()
    _test_layer_norm()
