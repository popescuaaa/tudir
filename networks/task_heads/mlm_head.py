import os
import sys
import torch
sys.path.append(os.getcwd())
from task_head import TaskHead
from task_head import TaskConfig



class MLM_Config(TaskConfig):
    """
    treat special tokens
    prepare inputs for mlm taks
    """
    def __init__(self, name: str, input_dim: int, output_dim: int) -> None:
        super().__init__(name, input_dim, output_dim)
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim



class MLM_head(TaskHead):
    """
    Masked Language Modeling head.    
    """

    def __init__(self, config: MLM_Config) -> None:
        super().__init__(config)
        self.config = config

    def prepare_inputs(self, inputs, **kwargs):
        """
        inputs: [batch_size, seq_len] dtype: Torch.long (token ids din tokenizer)
        """
        pass



    def forward(self, inputs, targets=None):
        """
        """

        mlm_outputs = 0

        if targets is not None:
            """
            compute the mlm task loss
            """
            mlm_outputs, mlm_loss = 0, 0

        return mlm_outputs, mlm_loss


# mlm_config = MLM_Config("mlm", input_dim=768, output_dim=768)
# mlm_head = MLM_head(mlm_config)

