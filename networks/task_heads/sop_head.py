import sys
import os
from copy import deepcopy
from typing import Tuple, List
import numpy as np
from tokenizers import BertWordPieceTokenizer, Tokenizer
from networks.task_heads.task_head import TaskHead, TaskConfig
import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

class SOPConfig(TaskConfig):
    """
    Config for the SOP head.
    """

    def __init__(self, name: str, input_dim: int, output_dim: int) -> None:
        super().__init__(name, input_dim, output_dim)
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim


class SentenceOrderPrediction(TaskHead):
    """
    Sentence order prediction head.
    """

    def __init__(self, config: TaskConfig, dropout: float = 0.1) -> None:
        super().__init__(config=config)
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.linear = nn.Linear(config.input_dim, config.output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.config = config

        self.sop_classifier = nn.Sequential(
            self.dropout,
            self.linear,
        )

        self.sep_token_id = 3
        self.cls_token_id = 2
        self.pad_token_id = 0

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.sop_classifier(inputs[:, 0, :])
        loss = self.loss(output, targets)
        return output, loss

    def split_inputs(self, batch_inputs: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        res = []
        for inputs in batch_inputs:
            # Split tensor by sep + cls positions and perform random shuffle
            indexes = torch.arange(0, inputs.shape[0], dtype=torch.long)

            pad_mask = (inputs == self.pad_token_id).long().nonzero().view(-1)
            if pad_mask.sum() < 1:
                first_pad = len(inputs)
            else:
                first_pad = indexes.gather(0, pad_mask)[0]
            
            non_pad_inputs = inputs[1:first_pad]
            non_pad_indexes = indexes[1:first_pad]

            print(non_pad_inputs.shape)
            mask = (non_pad_inputs == self.sep_token_id).long().nonzero().view(-1)
            print(mask.shape)
            sep_indexes = non_pad_indexes.gather(0, mask)
            print(sep_indexes.shape)
            sep_indexes[0] += 1
            sep_indexes = sep_indexes - torch.cat([torch.tensor([0]), sep_indexes[:-1] - 1])
            if first_pad == len(inputs):
                sep_indexes[-1] = len(inputs) - torch.sum(sep_indexes[:-1])
            sep_indexes[-1] -= 1

            if torch.sum(sep_indexes) < first_pad:
                sep_indexes[-1] += first_pad - sep_indexes[-1].item() - 1
            
            print(sum([t.item() for t in sep_indexes]))
            try:
                non_pad_inputs = list(non_pad_inputs.split(list(map(lambda t: t.item(), list(sep_indexes)))))
            except RuntimeError:
                print(torch.sum(sep_indexes), "pad:", first_pad, "len:", len(inputs))

            if len(inputs[first_pad:]) != 0:
                non_pad_inputs.append(inputs[first_pad:])

            res.append(non_pad_inputs)
        
        return res

    def prepare_inputs(self, inputs: torch.Tensor, **kwargs):
        """
        Function that prepares inputs token ids for the task head objective

        Args:
            inputs: (batch_size, seq_len)
            **kwargs:

        Returns:
            (batch_size, seq_len)
        """
        original_inputs = deepcopy(inputs)
        batch_split_inputs = self.split_inputs(inputs)
        res_inputs = []
        res_targets = []

        for split_inputs in batch_split_inputs:
            permute = False

            # Shuffle the inputs
            if np.random.rand() < 0.5:
                indexes = torch.randperm(len(split_inputs))
                permute = True
            else:
                indexes = torch.arange(len(split_inputs))
            
            split_inputs = [split_inputs[index] for index in indexes]
            split_inputs = torch.cat(split_inputs)
            inputs = torch.cat([original_inputs[0][0].unsqueeze(0), split_inputs])

            res_inputs.append(inputs)
            if permute:
                res_targets.append(torch.tensor([1]))
            else:
                res_targets.append(torch.tensor([0]))

        return torch.stack(res_inputs), torch.cat(res_targets)

class SpanOrderPrediction(TaskHead):
    """
    Span Order prediction task: given text tokens: [t1, t2, .... t20] predict if
    text[j:k] is before or after text[i:j]    
    """

    def __init__(self, config):
        super().__init__(config)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def prepare_inputs(self, inputs: torch.Tensor, **kwargs):
        """
        Function that prepares inputs token ids for the task head objective

        Args:
            inputs: (batch_size, seq_len)
            **kwargs:

        Returns:
            (batch_size, seq_len)
        """
        targets = None
        return inputs, targets


class SpanContextPrediction(TaskHead):
    """
    Span Context prediction task: given text tokens: [t1, t2, ..... t20] predict if
    text[j:k] is contained in text[i:p]
    """

    def __init__(self, config):
        super().__init__(config)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def split_inputs(self, inputs: torch.Tensor, **kwargs):
        """
        Split the inputs by CLS, SEP, and PAD tokens
        """
        pass

    def prepare_inputs(self, inputs: torch.Tensor, **kwargs):
        """
        Function that prepares inputs token ids for the task head objective

        Args:
            inputs: (batch_size, seq_len)
            **kwargs:

        Returns:
            (batch_size, seq_len)
        """
        # Prepare inputs for albert sentence order prediction
        targets = 0
        print(inputs)
        return inputs, targets


# if __name__ == '__main__':
#     UNK_TOK = 0
#     SEP_TOK = 1
#     MASK_TOK = 2
#     CLS_TOK = 3
#     PAD_TOK = 4

#     vocab_size = 20

#     # Transformer config
#     transf_embedding_dim = 10
#     transf_hidden_dim = 10
#     transf_num_layers = 1
#     transf_num_heads = 1
#     transf_dropout = 0
#     transf_act = F.gelu

#     sop_transformer_config = TransformerBlockConfig(
#         embedding_dim=transf_embedding_dim,
#         hidden_dim=transf_hidden_dim,
#         num_layers=transf_num_layers,
#         num_heads=transf_num_heads,
#         dropout=transf_dropout,
#         activation=F.gelu,
#         stochastic_depth=False,
#     )

#     sop_transformer = SimpleTransformerBlocks(config=sop_transformer_config, vocab_size=vocab_size)
#     sop_head_config = TaskConfig("sop", input_dim=transf_hidden_dim, output_dim=transf_hidden_dim)
#     sop_head = SentenceOrderPrediction(config=sop_head_config)
#     inputs = torch.tensor(np.array([3, 2, 0, 0, 5, 6, 1, 8, 9, 10, 11, 12, 13, 14, 1, 16, 17, 18, 7, 7])).unsqueeze(0)
#     inputs = inputs.repeat(repeats=(10, 1))
#     inputs = inputs.long()
#     inputs, targets = sop_head.prepare_inputs(inputs=inputs)
#     inputs = sop_transformer(inputs)
#     print(inputs, targets)
#     task_output, task_loss = sop_head(inputs=inputs, targets=targets)