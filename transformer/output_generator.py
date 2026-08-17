"""
输出层 Generator：将解码器的 d_model 维特征映射为词汇表概率分布。

Linear(d_model -> vocab_size) + Softmax(dim=-1)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding_position import Embedding, PositionalEncoding
from .attention_modules import MultiHeadAttention, FeedForward
from .encoder_layer import EncoderLayer
from .encoder_stack import Encoder
from .decoder_layer import DecoderLayer
from .decoder_stack import Decoder, _test_decoder as _use_decoder


class Generator(nn.Module):
    """
    将解码器的特征输出投影到词汇表维度，并做 Softmax 得到概率分布。
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Args:
            d_model:    词向量维度
            vocab_size: 目标词汇表大小
        """
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, vocab_size]，最后一维和为 1
        """
        x = self.linear(x)
        x = F.softmax(x, dim=-1)
        return x


# =============== 独立测试代码 ===============
def _test_generator():
    decoder_output = _use_decoder()  # [2, 4, 512]

    generator = Generator(512, 1000)
    output = generator(decoder_output)  # [2, 4, 1000]

    print(f"Generator 输出形状: {output.shape}")

    # 验证每个位置概率和为 1
    probs = output[0, 0]
    print(f"第 1 个样本第 1 个词的概率和: {probs.sum().item():.6f}")  # 应为 1.0


if __name__ == '__main__':
    _test_generator()
