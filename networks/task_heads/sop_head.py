
from task_head import TaskHead
from task_head import TaskConfig


class SOP_head(TaskHead):
    """
    Masked Language Modeling head.    
    """
    def __init__(self, config):
        super().__init__(config)

    def forward(self, inputs, targets=None):
        pass

    def prepare_inputs(self, inputs, **kwargs):
        pass

sop_config = TaskConfig("sop", input_dim=768, output_dim=768)
sop_head = SOP_head(sop_config)
