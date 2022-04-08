from typing import Callable, Optional, OrderedDict, Tuple
from pyparsing import Opt
import torch
from torchvision.ops.stochastic_depth import StochasticDepth
from torch import nn
from torch import Tensor
from torch.nn import functional as F

def probability_string(probability: float) -> str:
    """
    Converts a probability to a string.
    """
    return f"{probability:.2f}".replace("0.", "0,").replace("0,0", "0,0")

class TransformerBlockConfig:
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        activation: Callable[[Tensor], Tensor],
        stochastic_depth: Optional[bool] = True,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.stochastic_depth = stochastic_depth

        if stochastic_depth:
            # generate stochastic depth ranges
            stochastic_depth_range = range(1, num_layers + 1)
            # stochastic depth goes from [0, dropout]
            self.stochastic_depth_range_prob = [1 * (i + 1) / num_layers * dropout for i in stochastic_depth_range]
            # do not apply dropout to the final layer
            self.stochastic_depth_range_prob[-1] = 0.

DefaultTransformerConfig = TransformerBlockConfig(
    embedding_dim=768,
    hidden_dim=2048,
    num_layers=12,
    num_heads=16,
    dropout=0.1,
    activation=F.gelu,
    stochastic_depth=False,
)

class SimpleTransformerBlocks(nn.Module):
    def __init__(
        self,
        config: TransformerBlockConfig = DefaultTransformerConfig,
        vocab_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        
        # set transformer encoder layers followed by stochastic depth
        modules = []
        for idx in range(config.num_layers):
            block = []
            # add the multi-head self-attention layer
            block.append((
                f'Encoder Layer {idx+1:02}',
                nn.TransformerEncoderLayer(
                    d_model=config.embedding_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.hidden_dim,
                    dropout=config.dropout,
                    activation=config.activation,
                    batch_first=True,
                    norm_first=False,)
            ))
            # add the stochastic depth layer if enabled and if stochastic depth prob is not 0
            if config.stochastic_depth and config.stochastic_depth_range_prob[idx]:
                block.append((
                    f'StochasticDepth {idx+1:02} prob:{probability_string(config.stochastic_depth_range_prob[idx])}',
                    StochasticDepth(p=config.stochastic_depth_range_prob[idx], mode="row")
                ))
            
            modules.append(block)

        # flatten blocks list
        modules = [item for block in modules for item in block]

        # add an embedding layer if vocab_size is provided
        if vocab_size is not None:
            embedding_layer = nn.Embedding(vocab_size, config.embedding_dim)            
            modules.insert(0, (f'Embedding Layer', embedding_layer))

        self.blocks = nn.Sequential(OrderedDict(modules))

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        for name, layer in self.blocks.named_children():
            print(name)
            if name.startswith('Encoder'):
                x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            else:
                x = layer(x)
        return x






        