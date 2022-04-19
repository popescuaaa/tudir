
# NOT TESTED YET


import math
import os
import sys
import torch
from networks.task_heads.mask_utils_lucid import mask_with_tokens
sys.path.append(os.getcwd())
from task_head import TaskHead
from task_head import TaskConfig
from typing import List, Optional
from functools import reduce

from torch.nn import functional as F

from networks.transformers.vanilla_transformer \
    import SimpleTransformerBlocks, TransformerBlockConfig


class MLM_Config(TaskConfig):
    """
    Config for the MLM head.
    """
    def __init__(
            self,
            name: str,
            input_dim: int,
            output_dim: int,
            mask_prob: float,
            replace_prob: float,
            random_token_prob: float,
            num_tokens: Optional[int],
            mask_token_id: int,
            pad_token_id: int,
            mask_ignore_token_ids: List[int]) -> None:
        
        super().__init__(name, input_dim, output_dim)
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_token_prob = random_token_prob
        
        self.num_tokens = num_tokens
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.mask_ignore_token_ids = mask_ignore_token_ids


class MLM_head(TaskHead):
    """
    Masked Language Modeling head.    
    """

    def __init__(
            self,
            config: MLM_Config,
            transformer: SimpleTransformerBlocks) -> None:
        super().__init__(config)
        self.config = config
        self.transformer = transformer


    def prob_mask_like(input, prob):
        """
        Generate a mask with the same shape as input, where the mask is True
        with probability prob."""
        return torch.zeros_like(input).float().uniform_(0, 1) < prob


    def mask_with_tokens(self, token_seq, token_ids):
        """
        True for tokens that should not be masked.
        False for tokens that can be masked.
        """
        init_no_mask = torch.full_like(token_seq, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (token_seq == el), token_ids, init_no_mask)
        return mask


    def get_mask_subset_with_prob(self, mask, prob):
        batch, seq_len, device = *mask.shape, mask.device
        max_masked = math.ceil(prob * seq_len)

        num_tokens = mask.sum(dim=-1, keepdim=True)
        mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
        mask_excess = mask_excess[:, :max_masked]

        rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
        _, sampled_indices = rand.topk(max_masked, dim=-1)
        sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

        new_mask = torch.zeros((batch, seq_len + 1), device=device)
        new_mask.scatter_(-1, sampled_indices, 1)
        return new_mask[:, 1:].bool()


    def prepare_inputs(self, inputs, **kwargs):
        
        # Ignore tokens such as [CLS], [SEP], etc.
        # no_mask will have True for these tokens.
        no_mask = self.mask_with_tokens(inputs, self.config.mask_ignore_token_ids)

        mask = self.get_mask_subset_with_prob(~no_mask, self.config.mask_prob)

        masked_seq = inputs.clone().detach()

        if self.config.random_token_prob > 0:
            raise "Not implemented yet (random token prob)"

        # generate a replace probability tensor
        replace_prob = self.prob_mask_like(inputs, self.replace_prob)

        # mask the tokens with probability replace_prob
        masked_seq = masked_seq.masked_fill(mask * replace_prob, self.mask_token_id)

        return masked_seq, inputs.clone()


    def forward(self, input_batch, targets=None):
        """
        input_batch: [batch_size, seq_len, input_dim]
        """

        mlm_inputs, mlm_targets = self.prepare_inputs(input_batch)

        mlm_outputs = self.transformer(mlm_inputs)

        mlm_loss = F.cross_entropy(
            mlm_outputs.transpose(1, 2),
            mlm_targets,
            ignore_index = self.pad_token_id
        )

        return mlm_loss




mlm_transformer_config = TransformerBlockConfig(
    embedding_dim=768,
    hidden_dim=2048,
    num_layers=12,
    num_heads=16,
    dropout=0.1,
    activation=F.gelu,
    stochastic_depth=False,
)
mlm_transformer = SimpleTransformerBlocks(mlm_transformer_config, vocab_size=30_000)

mlm_head_config = MLM_Config(
    name="mlm_head",
    input_dim=768,
    output_dim=768,
    mask_prob=0.15,
    replace_prob=0.9,
    random_token_prob=0,
    num_tokens=None,
    mask_token_id=2,
    pad_token_id=0,
    mask_ignore_token_ids=[]
)

mlm_head = MLM_head(mlm_head_config, mlm_transformer)

'''
for batch_queries in enumerate(queries_dataloader):
    # TODO: batch_encode_plus - get the attention mask from dict output
    batch_queries = Tokenizer.batch_encode(batch_queries, max_length=max_query_length).ids

    batch_queries = torch.tensor(batch_queries, dtype=torch.long)
    batch_queries = batch_queries.to(device)
    mlm_inputs, mlm_targets = mlm_head.prepare_inputs(batch_queries)
    
    mlm_hidden_states = Transformer(mlm_inputs, attention_mask=None, padding ...)

    mlm_outputs, mlm_loss = mlm_head(mlm_hidden_states, mlm_targets)
'''

