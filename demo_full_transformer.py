"""
完整 Transformer 模型组装 + 端到端测试 Demo。

对应论文《Attention Is All You Need》Figure 1 完整架构：
    Input -> Embedding + PositionalEncoding
        -> Encoder Stack (N=6 EncoderLayer)
            -> Decoder Stack (N=6 DecoderLayer)
                -> Generator (Linear + Softmax)
                    -> Output Probabilities

使用方法：
    python demo_full_transformer.py
"""

import copy
import torch
import torch.nn as nn

from transformer import (
    Embedding,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    EncoderLayer,
    Encoder,
    DecoderLayer,
    Decoder,
    Generator,
)


def subsequent_mask(size: int, device=None) -> torch.Tensor:
    """创建解码器因果掩码，当前位置只能关注自身及之前的 token。"""
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).unsqueeze(0)


class EncoderDecoder(nn.Module):
    """
    标准的编码器-解码器架构，即完整 Transformer。
    """

    def __init__(
        self,
        source_embed: nn.Sequential,
        encoder: Encoder,
        target_embed: nn.Sequential,
        decoder: Decoder,
        generator: Generator,
    ):
        """
        Args:
            source_embed: 编码器输入处理链：Embedding + PositionalEncoding
            encoder:      编码器栈
            target_embed: 解码器输入处理链：Embedding + PositionalEncoding
            decoder:      解码器栈
            generator:    输出生成器
        """
        super().__init__()
        self.source_embed = source_embed
        self.encoder = encoder
        self.target_embed = target_embed
        self.decoder = decoder
        self.generator = generator

    def encode(self, source_x: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """编码器前向传播"""
        embed_x = self.source_embed(source_x)
        return self.encoder(embed_x, source_mask)

    def decode(
        self,
        target_y: torch.Tensor,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """解码器前向传播"""
        embed_y = self.target_embed(target_y)
        return self.decoder(embed_y, encoder_output, source_mask, target_mask)

    def forward(
        self,
        source_x: torch.Tensor,
        target_y: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        完整前向传播：编码 -> 解码 -> 生成概率分布

        Args:
            source_x:   源序列索引 [batch_size, src_len]
            target_y:   目标序列索引 [batch_size, tgt_len]
            source_mask: 源序列掩码 [batch_size, 1, src_len]
            target_mask: 目标序列掩码 [batch_size, tgt_len, tgt_len]
        Returns:
            [batch_size, tgt_len, vocab_size] 词汇表概率分布
        """
        encoder_result = self.encode(source_x, source_mask)
        decoder_result = self.decode(target_y, encoder_result, source_mask, target_mask)
        output = self.generator(decoder_result)
        return output


def make_model(
    src_vocab: int = 1000,
    tgt_vocab: int = 1000,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> EncoderDecoder:
    """
    构建一个完整 Transformer 模型的工具函数。

    Args:
        src_vocab: 源词汇表大小，默认 1000
        tgt_vocab: 目标词汇表大小，默认 1000
        N:         编码器/解码器层数，论文默认 6
        d_model:   词向量维度，论文默认 512
        d_ff:      FFN 中间维度，论文默认 2048
        h:         多头注意力头数，论文默认 8
        dropout:   随机失活概率，默认 0.1

    Returns:
        完整的 EncoderDecoder 模型实例
    """
    c = copy.deepcopy

    # ---- 编码器侧 ----
    source_embed = nn.Sequential(
        Embedding(src_vocab, d_model),
        PositionalEncoding(d_model, dropout),
    )
    self_attn = MultiHeadAttention(d_model, h, dropout)
    ff = FeedForward(d_model, d_ff, dropout)
    encoder_layer = EncoderLayer(d_model, c(self_attn), c(ff), dropout)
    encoder = Encoder(encoder_layer, N)

    # ---- 解码器侧 ----
    target_embed = nn.Sequential(
        Embedding(tgt_vocab, d_model),
        PositionalEncoding(d_model, dropout),
    )
    decoder_self_attn = MultiHeadAttention(d_model, h, dropout)
    decoder_src_attn = MultiHeadAttention(d_model, h, dropout)
    decoder_ff = FeedForward(d_model, d_ff, dropout)
    decoder_layer = DecoderLayer(
        d_model, c(decoder_self_attn), c(decoder_src_attn), c(decoder_ff), dropout
    )
    decoder = Decoder(decoder_layer, N)

    # ---- 输出层 ----
    generator = Generator(d_model, tgt_vocab)

    # ---- 组装完整模型 ----
    model = EncoderDecoder(source_embed, encoder, target_embed, decoder, generator)

    # 论文推荐：用 Xavier 初始化参数，加速收敛
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def run_demo():
    """端到端 Demo：构建模型并做一次前向传播，验证整个链路。"""
    print("=" * 70)
    print("  Transformer 完整架构 - 端到端测试")
    print("  (论文配置: N=6, d_model=512, d_ff=2048, h=8)")
    print("=" * 70)

    # 1. 构建模型（论文默认配置）
    model = make_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[模型参数] 总参数量: {total_params:,}\n")
    print(model)

    print("\n" + " -.- " * 10)

    # 2. 构造测试输入：batch=2, 每句 4 个 token
    source_x = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    target_y = torch.LongTensor([[3, 8, 6, 4], [9, 6, 2, 6]])

    # 3. 构造掩码：source_mask 处理 padding，target_mask 阻止看到未来 token。
    source_mask = torch.ones(2, 1, 4, dtype=torch.bool)
    target_mask = subsequent_mask(4)

    # 4. 前向传播
    result = model(source_x, target_y, source_mask, target_mask)

    print(f"\n[输入]  source_x 形状: {tuple(source_x.shape)}   源序列 (2 句, 每句 4 词)")
    print(f"[输入]  target_y 形状: {tuple(target_y.shape)}   目标序列 (2 句, 每句 4 词)")
    print(f"\n[输出]  result 形状:   {tuple(result.shape)}    (batch=2, seq_len=4, vocab_size=1000)")
    print(f"[输出]  数值范围:      [{result.min().item():.4f}, {result.max().item():.4f}]")

    # 5. 验证概率分布
    sample = result[0, 0]
    print(f"[验证]  第 1 句第 1 词概率和 = {sample.sum().item():.6f}  (应为 1.000000)")
    print(f"[验证]  第 2 句第 3 词概率和 = {result[1, 2].sum().item():.6f}  (应为 1.000000)")

    print("\n✅ Transformer 端到端测试通过！")
    return model


if __name__ == '__main__':
    run_demo()
