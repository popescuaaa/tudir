import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.nn import functional as F
from networks.transformers.vanilla_transformer import SimpleTransformerBlocks, TransformerBlockConfig


def test_transformer_encoder():
    model = SimpleTransformerBlocks().cuda()
    x = torch.randn(1, 1, model.config.embedding_dim).cuda()
    y = model(x)

def test_transformer_with_embeddings():
    vocab_size = 100
    model = SimpleTransformerBlocks(vocab_size=vocab_size).cuda()
    x = torch.arange(vocab_size).cuda()
    x = x.view(1, -1)
    y = model(x)

def test_transformer_with_stochastic_depth():
    vocab_size = 100

    model_config = TransformerBlockConfig(
        embedding_dim=768,
        hidden_dim=2048,
        num_layers=12,
        num_heads=16,
        dropout=0.1,
        activation=F.gelu,
        stochastic_depth=True,
    )

    model = SimpleTransformerBlocks(vocab_size=vocab_size).cuda()
    
    x = torch.arange(vocab_size).cuda()
    x = x.view(1, -1)
    for name, layer in model.named_children():
        if "StochasticDepth" in name:
            assert layer.p == model_config.stochastic_depth_range_prob[idx]
            idx += 1

