"""
Transformer 输入模块：词嵌入层 (Word Embedding) + 位置编码 (Positional Encoding)

对应论文《Attention Is All You Need》：
    - 3.4 Embeddings and Softmax
    - 3.5 Positional Encoding
"""

import math
import torch
import torch.nn as nn


class Embedding(nn.Module):
    """词嵌入层：将单词索引映射为 d_model 维的词向量，并乘以 sqrt(d_model) 缩放。"""

    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: 词汇表大小（去重后的单词总数）
            d_model: 词嵌入维度，论文默认 512
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入索引张量，形状 [batch_size, seq_len]
        Returns:
            缩放后的词向量，形状 [batch_size, seq_len, d_model]
        """
        # 乘以 sqrt(d_model) 的目的：平衡梯度，防止梯度消失/爆炸
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    位置编码层：使用 sin/cos 函数为词向量注入位置信息。
    公式（论文 3.5 节）：
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        """
        Args:
            d_model: 词向量维度
            dropout: 随机失活概率
            max_len: 支持的最大句子长度，默认 5000
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码矩阵，形状 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]

        # 计算 1 / 10000^(2i/d_model)，使用 exp & log 等价变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # [d_model / 2]

        position_value = position * div_term  # 广播后形状 [max_len, d_model/2]
        pe[:, 0::2] = torch.sin(position_value)  # 偶数位置使用 sin
        pe[:, 1::2] = torch.cos(position_value)  # 奇数位置使用 cos

        pe = pe.unsqueeze(0)  # 升维为 [1, max_len, d_model] 便于 batch 广播
        self.register_buffer('pe', pe)  # 注册到缓冲区，不参与梯度更新

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 词向量，形状 [batch_size, seq_len, d_model]
        Returns:
            词向量 + 对应位置编码，然后 dropout，形状不变
        """
        # 截取前 seq_len 个位置的编码并与词向量相加
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =============== 独立测试函数（供本模块单独运行 & 其它模块调用） ===============
def _test_embedding():
    vocab_size, d_model = 1000, 512
    my_embed = Embedding(vocab_size, d_model)
    x = torch.tensor([[100, 2, 421, 300], [500, 888, 306, 509]])
    result = my_embed(x)
    print(f"Embedding 输出形状: {result.shape}")  # [2, 4, 512]


def _get_position_demo_input():
    """供其他模块测试使用的标准化输入。"""
    vocab_size, d_model = 1000, 512
    my_embed = Embedding(vocab_size, d_model)
    x = torch.tensor([[100, 2, 421, 300], [500, 888, 306, 509]])
    embed_x = my_embed(x)
    my_position = PositionalEncoding(d_model=d_model, dropout=0.1)
    return my_position(embed_x)  # [2, 4, 512]


def _test_position():
    result = _get_position_demo_input()
    print(f"Embedding + PositionalEncoding 输出形状: {result.shape}")  # [2, 4, 512]


if __name__ == '__main__':
    _test_embedding()
    _test_position()
