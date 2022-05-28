import os
import sys
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn, tensor
from torch.nn import functional as F

from networks.transformers.encoding import PositionalEncoding

sys.path.append(os.getcwd())
from dataset.msmarco_orcas.loader import QueryDocumentOrcasDataset
from networks.transformers.vanilla_transformer import DefaultTransformerConfig, TransformerBlockConfig, SimpleTransformerBlocks


class QueryDocumentTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        transformer_config: TransformerBlockConfig = DefaultTransformerConfig,
        task_names: List[str] = ["mlm", "sop"],
    ) -> None:
        super().__init__()
        
        self.transformer_blocks = SimpleTransformerBlocks(config=transformer_config)
        self.embedding = nn.Embedding(vocab_size, transformer_config.embedding_dim)
        self.projection = nn.ModuleDict({
            t: nn.Linear(transformer_config.embedding_dim, transformer_config.embedding_dim, bias=False) \
                 for t in task_names
        })

        self.positional_encoding = PositionalEncoding(transformer_config.embedding_dim)

    def forward(self, 
                x: Tensor, 
                src_mask: Tensor,
                src_key_padding_mask: Tensor,
                task: Dict[str, Tuple[int, int]]) -> Tuple[Tensor, Tensor]:

        emb = self.embedding(x)

        emb = self.positional_encoding(emb)
        
        emb_m, emb_s = torch.chunk(emb, 2)
        emb_m = self.projection["mlm"](emb_m)
        emb_s = self.projection["sop"](emb_s)
        emb = torch.cat([emb_m, emb_s], dim=0)

        emb = self.transformer_blocks(emb, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return emb 
