"""
子层连接结构：残差连接 (Add) + 层归一化 (Norm)。

每个子层（多头注意力 / FFN）都被包裹在这个结构中：
    output = LayerNorm(x + Dropout(Sublayer(x)))
即论文中的 Post-Norm 实现。
"""

import torch
import torch.nn as nn

from .embedding_position import (
    Embedding,
    PositionalEncoding,
    _get_position_demo_input as _use_position,
)
from .attention_modules import (
    MultiHeadAttention,
    FeedForward,
    LayerNorm,
)


class SublayerConnection(nn.Module):
    """
    子层连接结构：Sublayer + Residual Add + LayerNorm + Dropout。
    对应论文 "Add & Norm" 组件。
    """

    def __init__(self, d_model: int, dropout_p: float = 0.1):
        """
        Args:
            d_model: 输入/输出维度
            dropout_p: 随机失活概率
        """
        super().__init__()
        self.norm = LayerNorm(d_model)       # 使用传入的 d_model（修复硬编码 512）
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            sublayer: 可调用的子层，如多头注意力 / FFN
        Returns:
            经过 子层 -> 残差 -> 归一化 -> dropout 后的结果
        """
        # Post-Norm: 子层输出先 dropout，再与残差相加，最后层归一化。
        return self.norm(x + self.dropout(sublayer(x)))


# =============== 独立测试代码 ===============
def _test_sublayer_connection():
    x = _use_position()  # [2, 4, 512]

    sublayer_conn = SublayerConnection(512)

    # 1) 多头注意力子层测试
    result = sublayer_conn(x, lambda t: MultiHeadAttention(512, 8)(t, t, t))
    print(f"多头注意力子层输出形状: {result.shape}")  # [2, 4, 512]

    # 2) FFN 子层测试
    result = sublayer_conn(result, lambda t: FeedForward(512, 2048)(t))
    print(f"FFN 子层输出形状: {result.shape}")         # [2, 4, 512]


if __name__ == '__main__':
    _test_sublayer_connection()
