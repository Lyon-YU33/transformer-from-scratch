"""
编码器层 (Encoder Layer)。

每个编码器层由 2 个子层组成：
    1) 多头自注意力 + Add & Norm
    2) 前馈全连接 FFN + Add & Norm
"""

import torch
import torch.nn as nn

from .embedding_position import _get_position_demo_input as _use_position
from .attention_modules import MultiHeadAttention, FeedForward, clones
from .sublayer_connection import SublayerConnection


class EncoderLayer(nn.Module):
    """
    单个编码器层：多头自注意力子层 + FFN 子层。
    """

    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout_p: float = 0.1,
    ):
        """
        Args:
            d_model: 词向量维度
            self_attn: 多头自注意力对象
            feed_forward: FFN 对象
            dropout_p: 随机失活概率
        """
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 两个子层连接结构（自注意力 + FFN）
        self.sublayer = clones(SublayerConnection(d_model, dropout_p), 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] 源序列掩码
        Returns:
            同形状张量
        """
        # 子层 1：自注意力（Q=K=V=x）
        x = self.sublayer[0](x, lambda t: self.self_attn(t, t, t, mask))
        # 子层 2：FFN
        x = self.sublayer[1](x, lambda t: self.feed_forward(t))
        return x


# =============== 独立测试代码 ===============
def _test_encoder_layer():
    x = _use_position()  # [2, 4, 512]

    multi_head = MultiHeadAttention(512, 8)
    ff = FeedForward(512, 2048)
    encoder_layer = EncoderLayer(512, multi_head, ff)

    mask = torch.zeros(1, 4, 4)
    output = encoder_layer(x, mask)

    print(f"EncoderLayer 输出形状: {output.shape}")  # [2, 4, 512]
    print(f"EncoderLayer 结构:\n{encoder_layer}")


if __name__ == '__main__':
    _test_encoder_layer()
