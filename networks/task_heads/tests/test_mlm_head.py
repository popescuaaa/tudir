import enum
import os
import sys
import token
from matplotlib.transforms import Transform
import torch
import transformers
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join('..')))

from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.nn import functional as F

from task_heads.mlm_head import MLM_Config, MLM_head
from networks.transformers.query_document_transformer import QueryDocumentTransformer
from networks.transformers.vanilla_transformer import SimpleTransformerBlocks, TransformerBlockConfig


def test_mlm_head_forward(queries_dataloader, tokenizer):
    """
    Test forward pass for the MLM head.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer_config = TransformerBlockConfig(embedding_dim=768,
                                                hidden_dim=768,
                                                num_layers=4,
                                                num_heads=4,
                                                dropout=0.1,
                                                attention_dropout=0.1,
                                                activation=F.gelu,
                                                stochastic_depth=False
                                                )
    transformer = SimpleTransformerBlocks(transformer_config, vocab_size=tokenizer.get_vocab_size())

    mlm_config = MLM_Config("mlm", input_dim=768, output_dim=768)
    mlm_head = MLM_head(mlm_config)

    # data size = [num_batches, batch_size, seq_len, embedding_dim]
    queries_dataloader = torch.randint(0, 20000, (2, 8, 1024)).cuda()

    for batch_queries in enumerate(queries_dataloader):

        '''
        batch_encode_plus generate a dict of:
        {
            'input_ids': [batch_size, seq_len],
            'attention_mask': [batch_size, seq_len],
            'token_type_ids': [batch_size, seq_len],
        }
        '''
        batch_queries = tokenizer.batch_encode_plus(batch_queries)

        batch_inputs = batch_queries.to(device)

        mlm_hidden_size = transformer(batch_inputs)

        mlm_outputs, mlm_loss = mlm_head(mlm_hidden_size, targets=batch_targets)

        assert mlm_outputs.size() == batch_targets.size()
        assert mlm_loss.size() == torch.Size([])

    
    return None


################ This should be the test code ################
