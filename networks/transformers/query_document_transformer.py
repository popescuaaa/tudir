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
            t: nn.Linear(transformer_config.embedding_dim, vocab_size, bias=False) \
                 for t in task_names
        })

        self.positional_encoding = PositionalEncoding(transformer_config.hidden_dim)

    def forward(self, 
                x: Tensor, 
                src_mask: Tensor,
                src_key_padding_mask: Tensor,
                task: Dict[str, Tuple[int, int]]) -> Tuple[Tensor, Tensor]:

        emb = self.embedding(x)

        emb = self.positional_encoding(emb) # TODO review by ponku

        for t, (t_start, t_end) in task.items():
            emb[t_start:t_end, ...] = self.projection[t](emb[t_start:t_end, ...])

        emb = self.transformer_blocks(emb, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return emb 
