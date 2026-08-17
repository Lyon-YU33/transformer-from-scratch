"""
编码器栈 (Encoder Stack)。

由 N 个 EncoderLayer 堆叠而成（论文默认 N=6），
最后再加一层 LayerNorm 输出。
"""

import torch
import torch.nn as nn

from .embedding_position import _get_position_demo_input as _use_position
from .attention_modules import MultiHeadAttention, FeedForward, LayerNorm, clones
from .encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """
    完整编码器：N 层 EncoderLayer 堆叠 + 最终 LayerNorm。
    """

    def __init__(self, layer: EncoderLayer, N: int):
        """
        Args:
            layer: 单个 EncoderLayer 模板（会被深拷贝 N 次）
            N:     堆叠层数，论文默认 6
        """
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)  # 最终全局归一化

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [batch_size, seq_len, d_model] 词嵌入+位置编码后的输入
            mask: [batch_size, seq_len, seq_len] 源序列掩码
        Returns:
            [batch_size, seq_len, d_model] 编码器最终表征
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# =============== 独立测试代码 ===============
def _test_encoder():
    x = _use_position()  # [2, 4, 512]

    multi_head = MultiHeadAttention(512, 8)
    ff = FeedForward(512, 2048)
    encoder_layer = EncoderLayer(512, multi_head, ff)
    encoder = Encoder(encoder_layer, 6)  # 论文默认 6 层

    mask = torch.zeros(1, 4, 4)
    encoder_output = encoder(x, mask)

    print(f"Encoder 输出形状: {encoder_output.shape}")  # [2, 4, 512]
    print(f"输入示例(前5维): {x[0, 0, :5]}")
    print(f"输出示例(前5维): {encoder_output[0, 0, :5]}")
    return encoder_output


if __name__ == '__main__':
    _test_encoder()
