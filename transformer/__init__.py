"""
transformer - PyTorch 从零实现 Transformer 的核心包。

模块依赖顺序（按阅读顺序）：
    embedding_position  -> attention_modules -> sublayer_connection
    -> encoder_layer -> encoder_stack -> decoder_layer -> decoder_stack
    -> output_generator

最后通过 demo_full_transformer.py 组装并运行完整模型。
"""

# 统一导出关键类，便于外部使用
from .embedding_position import Embedding, PositionalEncoding
from .attention_modules import (
    attention,
    clones,
    MultiHeadAttention,
    FeedForward,
    LayerNorm,
)
from .sublayer_connection import SublayerConnection
from .encoder_layer import EncoderLayer
from .encoder_stack import Encoder
from .decoder_layer import DecoderLayer
from .decoder_stack import Decoder
from .output_generator import Generator

__all__ = [
    # 输入层
    "Embedding",
    "PositionalEncoding",
    # 核心组件
    "attention",
    "clones",
    "MultiHeadAttention",
    "FeedForward",
    "LayerNorm",
    # 子层连接
    "SublayerConnection",
    # 编码器
    "EncoderLayer",
    "Encoder",
    # 解码器
    "DecoderLayer",
    "Decoder",
    # 输出层
    "Generator",
]
