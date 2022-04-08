
from task_head import TaskHead
from task_head import TaskConfig


class MLM_head(TaskHead):
    """
    Masked Language Modeling head.    
    """
    def __init__(self, config):
        super().__init__(config)

    def forward(self, inputs, targets=None):
        pass

    def prepare_inputs(self, inputs, **kwargs):
        pass

mlm_config = TaskConfig("mlm", input_dim=768, output_dim=768)
mlm_head = MLM_head(mlm_config)
