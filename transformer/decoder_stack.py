"""
解码器栈 (Decoder Stack)。

由 N 个 DecoderLayer 堆叠而成（论文默认 N=6），
最后再加一层 LayerNorm 输出。
"""

import copy
import torch
import torch.nn as nn

from .embedding_position import Embedding, PositionalEncoding
from .attention_modules import MultiHeadAttention, FeedForward, LayerNorm, clones
from .encoder_layer import EncoderLayer
from .encoder_stack import Encoder, _test_encoder as _use_encoder
from .decoder_layer import DecoderLayer


class Decoder(nn.Module):
    """
    完整解码器：N 层 DecoderLayer 堆叠 + 最终 LayerNorm。
    """

    def __init__(self, layer: DecoderLayer, N: int):
        """
        Args:
            layer: 单个 DecoderLayer 模板（会被深拷贝 N 次）
            N:     堆叠层数，论文默认 6
        """
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

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
            source_mask:    源序列掩码
            target_mask:    目标序列掩码（下三角）
        Returns:
            [batch_size, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)


# =============== 独立测试代码 ===============
def _test_decoder():
    # 解码器输入
    y = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    my_embed = Embedding(1000, 512)
    embed_y = my_embed(y)
    my_position = PositionalEncoding(512, 0.1)
    position_y = my_position(embed_y)  # [2, 4, 512]

    # 各模块
    multi_attn = MultiHeadAttention(512, 8)
    self_attn = copy.deepcopy(multi_attn)
    src_attn = copy.deepcopy(multi_attn)
    ff = FeedForward(512, 2048)

    encoder_output = _use_encoder()
    source_mask = torch.zeros(1, 4, 4)
    target_mask = torch.zeros(1, 4, 4)

    decoder_layer = DecoderLayer(512, self_attn, src_attn, ff)
    decoder = Decoder(decoder_layer, 6)  # 论文默认 6 层

    result = decoder(position_y, encoder_output, source_mask, target_mask)

    print(f"Decoder 输出形状: {result.shape}")  # [2, 4, 512]
    print(f"Decoder 结构:\n{decoder}")


if __name__ == '__main__':
    _test_decoder()
