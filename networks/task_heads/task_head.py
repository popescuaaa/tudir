import os
import sys
from typing import Optional, Tuple, Union
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class TaskConfig:
    def __init__(
        self,
        name: str,
        input_dim: int,
        output_dim: int,
    ) -> None:
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim

class TaskHead(nn.Module): 
    def __init__(
        self,
        config: TaskConfig,
    ) -> None:
        super().__init__()
        self.config = config
        
    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Function that runs inputs through the task heads and computes the output."""
        raise NotImplementedError()

    @classmethod
    def prepare_inputs(cls, inputs: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        """Function that prepares inputs for the task heads."""
        raise NotImplementedError()