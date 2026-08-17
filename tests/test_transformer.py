import torch

from demo_full_transformer import make_model, subsequent_mask


def test_transformer_supports_different_source_and_target_lengths():
    model = make_model(src_vocab=32, tgt_vocab=32, N=1, d_model=16, d_ff=32, h=4, dropout=0.0)
    model.eval()

    source = torch.tensor([[1, 2, 3]])
    target = torch.tensor([[1, 2]])
    source_mask = torch.ones(1, 1, 3, dtype=torch.bool)

    output = model(source, target, source_mask, subsequent_mask(2))

    assert output.shape == (1, 2, 32)
    assert torch.allclose(output.sum(dim=-1), torch.ones(1, 2), atol=1e-6)
