import os
import sys
import torch
import random
from torch import Tensor
from networks.task_heads.task_head import TaskHead, TaskConfig
from typing import Callable, List, Optional, Tuple
from functools import reduce
from torch.nn import functional as F
import torch.nn as nn


class MLM_Config(TaskConfig):
    """
    Config for the MLM head.
    """

    def __init__(
            self,
            name: str,
            input_dim: int,
            output_dim: int,
            dropout: float,
            mask_prob: float,
            num_tokens: Optional[int],
            mask_token_id: int,
            pad_token_id: int,
            cls_token_id: int,
            sep_token_id: int
        ):
        super().__init__(name, input_dim, output_dim)

        self.mask_prob = mask_prob
        self.dropout = dropout
        self.num_tokens = num_tokens
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id


class MLM_head(TaskHead):
    """
    Masked Language Modeling head.    
    """

    def __init__(
            self,
            config: MLM_Config,
        ):
        super().__init__(config)
        self.config = config

        self.head = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.input_dim, self.config.output_dim, bias=False)
        )

    def replace_with_mask(self, x):
        return 2

    def replace_with_random(self, x):
        # ? exclude the special tokens from the random sampling
        return random.randint(0, self.config.num_tokens - 1)

    def replace_with_identity(self, x):
        return x

    def pad_inputs(self, inputs: Tensor) -> Tensor:
        """
        Pad inputs to the correct length.
        """
        return inputs

    def prepare_inputs(self, inputs: Tensor, max_len: int) -> Tuple[Tensor, Tensor]:
        """
        Prepare inputs for the MLM head.
        """

        functions = [self.replace_with_mask, self.replace_with_random, self.replace_with_identity]

        function_weights = [0.8, 0.1, 0.1]

        seqs_masked = []
        for seq in inputs:

            seq_masked = seq.clone()

            replace_indices = random.choices(list(range(len(seq))), k=min(1, int(self.config.num_tokens * len(seq))))

            replace_functions = random.choices(functions, weights=function_weights, k=len(replace_indices))

            replaced_values = [f(seq[i]) for i, f in zip(replace_indices, replace_functions)]

            for i, v in zip(replace_indices, replaced_values):
                seq_masked[i] = v

            seqs_masked.append(seq_masked)

        seqs_target = [seq.clone() for seq in inputs]

        # pad the beginning of the sequences with cls token
        seqs_target = [F.pad(seq, (1, 0), value=self.config.cls_token_id, mode='constant') for seq in seqs_target]
        seqs_masked = [F.pad(seq, (1, 0), value=self.config.cls_token_id, mode='constant') for seq in seqs_masked]

        # insert sep token
        seqs_target = [F.pad(seq, (0, 1), value=self.config.sep_token_id, mode='constant') for seq in seqs_target]
        seqs_masked = [F.pad(seq, (0, 1), value=self.config.sep_token_id, mode='constant') for seq in seqs_masked]

        # pad the sequences
        seqs_target = [F.pad(seq, (0, max_len - seq.shape[0]), value=self.config.pad_token_id, mode='constant') for seq in seqs_target]
        seqs_masked = [F.pad(seq, (0, max_len - seq.shape[0]), value=self.config.pad_token_id, mode='constant') for seq in seqs_masked]
        
        # for seq, seq_masked in zip(seqs_target, seqs_masked):
        #     print("OG", seq)
        #     print("MK:", seq_masked)
        #     print()

        seqs_target = torch.stack(seqs_target, dim=0)
        seqs_masked = torch.stack(seqs_masked, dim=0)

        return seqs_masked, seqs_target

        

    def forward(self, inputs: Tensor, targets: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the MLM head.
        """
        output = self.head(inputs)
        output = output.view(-1, self.config.output_dim)
        loss = F.cross_entropy(output, targets, reduction='batchmean')
        return output, loss


# if __name__ == "__main__":
#     vocab_size = 100

#     # Transformer config
#     transf_embedding_dim = 100
#     transf_hidden_dim = 100
#     transf_num_layers = 1
#     transf_num_heads = 1
#     transf_dropout = 0
#     transf_act = F.gelu

#     mlm_transformer_config = TransformerBlockConfig(
#         embedding_dim=transf_embedding_dim,
#         hidden_dim=transf_hidden_dim,
#         num_layers=transf_num_layers,
#         num_heads=transf_num_heads,
#         dropout=transf_dropout,
#         activation=F.gelu,
#         stochastic_depth=False,
#     )
#     mlm_transformer = SimpleTransformerBlocks(mlm_transformer_config, vocab_size=vocab_size)

#     UNK_TOK = 0
#     SEP_TOK = 1
#     MASK_TOK = 2
#     CLS_TOK = 3
#     PAD_TOK = 4

#     mlm_head_config = MLM_Config(
#         name="mlm_head",

#         input_dim=768,
#         output_dim=768,

#         mask_prob=0.15,

#         num_tokens=vocab_size,
#         mask_token_id=MASK_TOK,
#         pad_token_id=PAD_TOK,
#         cls_token_id=CLS_TOK,
#         sep_token_id=SEP_TOK,
#         mask_ignore_token_ids=[SEP_TOK, CLS_TOK, PAD_TOK]
#     )

#     mlm_head = MLM_head(mlm_head_config, mlm_transformer)

#     # generate random sequences
#     seqs = []
#     for i in range(1024):
#         seq_len = random.randint(3, 10)
#         seq = torch.randint(5, vocab_size - 1, (seq_len,))
#         seqs.append(seq)
    
#     # print the sequences
#     # print("Input of the MLM head:")
#     # for s in seqs:
#     #     print(s)

#     mlm_out, mlm_loss = mlm_head(seqs)

#     print("MLM loss:", mlm_loss.item())
#     print("MLM out shape:", mlm_out.shape)
