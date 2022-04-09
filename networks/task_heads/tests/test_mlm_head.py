import os
import sys
import torch
import transformers
# sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join('..')))

print(os.getcwd())

from tokenizers import Tokenizer
from task_heads.mlm_head import MLM_Config, MLM_head

""" Notes:
    text = "Beyonce, was she born in 1981?"
    use Tokenizer
    tokens = ["Beyonce", ",", "was", "she", "born", "in", "1981", "?"]
    check fw pass of the mlm head
"""

def test_mlm_head_forward(queries_dataloader, tokenizer):
    """
    Test forward pass for the MLM head.
    """

    # mlm_config = MLM_Config("mlm", input_dim=768, output_dim=768)
    # mlm_head = MLM_head(mlm_config)

    # for batch_queries in enumerate(queries_dataloader):
    #     batch_queries = tokenizer.batch_encode_plus(batch_queries)

    #     batch_inputs = batch_inputs.to(device)
    #     batch_targets = batch_targets.to(device)

    #     batch_inputs, batch_targets = mlm_head.prepare_inputs(batch_inputs, **kwargs)

    #     mlm_outputs, mlm_loss = mlm_head(batch_inputs, batch_targets)

    #     assert mlm_outputs.size() == batch_targets.size()
    #     assert mlm_loss.size() == torch.Size([])

    
    return None

tokenizer = Tokenizer.from_file("./BERT_tok-trained.json")

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
