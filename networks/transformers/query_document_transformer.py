import os
import sys
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn
from torch.nn import functional as F

sys.path.append(os.getcwd())
from dataset.msmarco_orcas.loader import QueryDocumentOrcasDataset


class QueryDocumentTransformer:
    def __init__(
        self,
        vocab_size: int,
        model_init: Tuple,
        heads_init: List[Tuple],
        heads_domain: Dict[str, str],
    ) -> None:
        
        # unpack initializers
        self.vocab_size = vocab_size
        self.transformer_config, self.transformer_constructor = model_init
        
        self.input_domains = heads_domain
        self.heads_init = {}
        for (head_name, (head_config, head_constructor)) in heads_init:
            self.heads_init[head_name] = (head_config, head_constructor)

        # check that all heads name are in the domain
        for head_name in self.heads_init:
            if head_name not in self.input_domains:
                raise ValueError(f"Head name {head_name} is not in the domain map {self.input_domains}")

        # check that domains are either query or document
        for head_name, head_domain in heads_domain.items():
            if head_domain not in ["query", "document"]:
                raise ValueError(f"Head domain {head_domain} is not in the domain map: '{['query', 'document']}'")


        # setup embedding tables
        self.embedding = nn.Embedding(self.vocab_size, self.transformer_config.embedding_dim)
        self.embedding_projections = nn.ModuleDict({
            "query": nn.Linear(self.transformer_config.embedding_dim, self.transformer_config.embedding_dim, bias=False),
            "document": nn.Linear(self.transformer_config.embedding_dim, self.transformer_config.embedding_dim, bias=False),
        })

        # setup transformer blocks and heads
        self.transformer_blocks = self.transformer_constructor(self.transformer_config)
        self.heads = nn.ModuleDict()

        for head_name, (head_config, head_constructor) in self.heads_init.items():
            self.heads[head_name] = head_constructor(head_config)

    def forward(self, x: Tensor, task: str) -> Tuple[Tensor, Tensor]:
        """
        x: (batch_size, seq_len)
        task: str - the task name to be performed
        """
        # [b, b, b, b] -> [4b]
        # {'MLM': querry, 'NSP': document}
        # select task domain
        task_domain = self.input_domains[task]
        x, y = self.heads[task].prepare_input(x)
        
        x = self.embedding(x) 

        # embed input
        x = self.embedding_projections[task_domain](x)

        # apply transformer blocks
        x = self.transformer_blocks(x)

        # apply heads
        task_outs, task_loss = self.heads[task](x, y)

        return task_outs, task_loss
