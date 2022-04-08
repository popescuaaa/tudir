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
    ) -> None:
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

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
        modules = [[
            (
                f'Encoder Layer {i+1:02}',
                nn.TransformerEncoderLayer(
                    d_model=config.embedding_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.hidden_dim,
                    dropout=config.dropout,
                    activation=config.activation,
                    batch_first=True,
                    norm_first=False,)
            ),
            (
                f'StochasticDepth {i+1:02} prob:{probability_string(config.stochastic_depth_range_prob[i])}',
                 StochasticDepth(p=config.stochastic_depth_range_prob[i], mode="row")
            ),
        ] for i in range(config.num_layers)]

        # flatten module list
        modules = [item for sublist in modules for item in sublist]

        if vocab_size is not None:
            embedding_layer = nn.Embedding(vocab_size, config.embedding_dim)            
            modules.insert(0, (f'Embedding Layer', embedding_layer))

        self.blocks = nn.Sequential(OrderedDict(modules))


    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        for name, layer in self.blocks.named_children():
            if name.startswith('Encoder'):
                x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            else:
                x = layer(x)
        return x

if __name__ == '__main__':
    x = torch.randn(32, 256, 768).cuda()
    model = SimpleTransformerBlocks().cuda()
    for _ in range(10):
        print(model(x).shape)



        