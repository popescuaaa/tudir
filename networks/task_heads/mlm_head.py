import os
import sys
import torch
from torch import Tensor
from task_head import TaskHead
from task_head import TaskConfig
from typing import List, Optional, Tuple
from functools import reduce
from torch.nn import functional as F

sys.path.append(os.getcwd())
from vanilla_transformer_mod import SimpleTransformerBlocks, TransformerBlockConfig


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

    def mask_with_tokens(self, t, token_ids):
        init_no_mask = torch.full_like(t, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
        return mask

    def prepare_inputs(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Prepare inputs for the MLM head.
        """
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.config.mask_prob)
        special_tokens_mask = self.mask_with_tokens(labels, self.config.mask_ignore_token_ids)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1

        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.config.replace_prob)).bool() & masked_indices
        inputs[indices_replaced] = self.config.mask_token_id

        indices_random = torch.bernoulli(
            torch.full(labels.shape, self.config.random_token_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.config.num_tokens, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def forward(self, inputs: Tensor, targets: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the MLM head.
        """
        mlm_inputs, mlm_targets = self.prepare_inputs(inputs)

        print("MASKED SEQ:\n", mlm_inputs.shape)
        print("TARGET SEQ:\n", mlm_targets.shape)

        mlm_output = self.transformer(mlm_inputs)

        print(mlm_output.shape)

        mlm_loss = F.cross_entropy(mlm_output.transpose(1, 2), mlm_targets, ignore_index=-1)

        return mlm_output, mlm_loss


vocab_size = 10

# Transformer config
transf_embedding_dim = 10
transf_hidden_dim = 10
transf_num_layers = 1
transf_num_heads = 1
transf_dropout = 0
transf_act = F.gelu

mlm_transformer_config = TransformerBlockConfig(
    embedding_dim=transf_embedding_dim,
    hidden_dim=transf_hidden_dim,
    num_layers=transf_num_layers,
    num_heads=transf_num_heads,
    dropout=transf_dropout,
    activation=F.gelu,
    stochastic_depth=False,
)
mlm_transformer = SimpleTransformerBlocks(mlm_transformer_config, vocab_size=vocab_size)

UNK_TOK = 0
SEP_TOK = 1
MASK_TOK = 2
CLS_TOK = 3
PAD_TOK = 4

mlm_head_config = MLM_Config(
    name="mlm_head",

    input_dim=768,
    output_dim=768,

    mask_prob=0.99,
    replace_prob=0.99,
    random_token_prob=0.5,  # percentage of the remaining 1-replace_prob tokens

    num_tokens=vocab_size,
    mask_token_id=MASK_TOK,
    pad_token_id=PAD_TOK,
    mask_ignore_token_ids=[SEP_TOK, CLS_TOK, PAD_TOK]
)

mlm_head = MLM_head(mlm_head_config, mlm_transformer)

# batch_size, seq_len, input_dim
trans_input_shape = (2, 9)
tok_input = torch.randint(5, vocab_size, trans_input_shape)
tok_input[0][0] = 3
tok_input[0][7] = 1
tok_input[0][8] = 4
tok_input[1][0] = 3
tok_input[1][7] = 1
tok_input[1][8] = 4

print(tok_input)

mlm_head(tok_input)
