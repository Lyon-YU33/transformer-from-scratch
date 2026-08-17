"""
解码器层 (Decoder Layer)。

每个解码器层由 3 个子层组成：
    1) Masked 多头自注意力 + Add & Norm
    2) 编码器-解码器 多头注意力 + Add & Norm
    3) FFN + Add & Norm
"""

import copy
import torch
import torch.nn as nn

from .embedding_position import Embedding, PositionalEncoding
from .attention_modules import MultiHeadAttention, FeedForward, clones
from .sublayer_connection import SublayerConnection
from .encoder_layer import EncoderLayer
from .encoder_stack import Encoder, _test_encoder as _use_encoder


class DecoderLayer(nn.Module):
    """
    单个解码器层：自注意力子层 + 编解码注意力子层 + FFN 子层。
    """

    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        src_attn: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout_p: float = 0.1,
    ):
        """
        Args:
            d_model:      词向量维度
            self_attn:    目标序列自注意力（第1个子层，带mask防止看未来）
            src_attn:     编码器-解码器注意力（第2个子层，K/V来自编码器，Q来自解码器）
            feed_forward: FFN 对象
            dropout_p:    随机失活概率
        """
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        # 3 个子层连接结构
        self.layers = clones(SublayerConnection(d_model, dropout_p), 3)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:              解码器输入 [batch_size, tgt_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            source_mask:    源序列掩码 [batch_size, 1, src_len]，对所有目标位置广播
            target_mask:    目标序列掩码 [batch_size, tgt_len, tgt_len]
                            （下三角形式，防止看到未来 token）
        Returns:
            [batch_size, tgt_len, d_model]
        """
        # 子层 1：掩码自注意力（Q=K=V=x）
        x = self.layers[0](x, lambda t: self.self_attn(t, t, t, target_mask))
        # 子层 2：编解码注意力（Q=x，K=V=encoder_output）
        x = self.layers[1](x, lambda t: self.src_attn(t, encoder_output, encoder_output, source_mask))
        # 子层 3：FFN
        x = self.layers[2](x, lambda t: self.feed_forward(t))
        return x


# =============== 独立测试代码 ===============
def _test_decoder_layer():
    # 构造解码器输入
    y = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    my_embed = Embedding(1000, 512)
    embed_y = my_embed(y)
    my_position = PositionalEncoding(512, 0.1)
    position_y = my_position(embed_y)  # [2, 4, 512]

    # 3 个注意力/FFN 模块
    multi_attn = MultiHeadAttention(512, 8)
    self_attn = copy.deepcopy(multi_attn)
    src_attn = copy.deepcopy(multi_attn)
    ff = FeedForward(512, 2048)

    encoder_output = _use_encoder()  # 获取编码器输出
    source_mask = torch.zeros(1, 4, 4)
    target_mask = torch.zeros(1, 4, 4)

    decoder_layer = DecoderLayer(512, self_attn, src_attn, ff)
    result = decoder_layer(position_y, encoder_output, source_mask, target_mask)

    print(f"DecoderLayer 输出形状: {result.shape}")  # [2, 4, 512]
    print(f"DecoderLayer 结构:\n{decoder_layer}")


if __name__ == '__main__':
    _test_decoder_layer()
