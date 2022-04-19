
import sys
import os
from typing import Tuple
from task_head import TaskHead
from task_head import TaskConfig
import torch
from torch import nn
from networks.transformers.vanilla_transformer import SimpleTransformerBlocks, TransformerBlockConfig, DefaultTransformerConfig

sys.path.append(os.getcwd())


class SenteceOrderPrediction(TaskHead):
    """
    Masked Language Modeling head.    
    """
    def __init__(self, config):
        super().__init__(config)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
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

class SpanOrderPrediction(TaskHead):
    """
    Span Order prediction task: given text tokens: [t1, t2, .... t20] predict if
    text[j:k] is before or after text[i:j]    
    """
    def __init__(self, config):
        super().__init__(config)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
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



sop_config = TaskConfig("mlm", input_dim=768, output_dim=768)
sop_head = SenteceOrderPrediction(sop_config)

if __name__ == '__main__':
    # Load tokenizer from BERT_tok-trained.json
    # tokenizer = Tokenizer.from_file("BERT_tok-trained.json")
    # use dummy text as input
    dummy_text = [
        ['Ma cheama George si sunt un Babalau. Teo e cel mai prost om. Ceachi e frumos'],
        ['aplicatia asta nu inteleg cum functioneaza cum naiba vin requesturile. adica nu ai metode separate care sa trateze anumite requesturi']
    ]
    # tokenize the text
    # dummy_text = tokenizer.batch_encode(dummy_text, max_length=512)
    # input_ids, target_ids = task_head.prepare_inputs(dummy_text)
    # task_output, task_loss = task_head(input_ids, target_ids)



