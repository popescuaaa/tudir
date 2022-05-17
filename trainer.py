import random
from re import I
from dataset.msmarco_orcas.loader import QueryDocumentOrcasDataset
from tokenization.vocab_tokenizers import train_BertWordPieceTokenizer
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Union
from tokenizers import Tokenizer
import sys
import os
import torch
from networks.task_heads.mlm_head import MLM_head, MLM_Config
from networks.task_heads.sop_head import SentenceOrderPrediction, SOPConfig
from networks.transformers.query_document_transformer import QueryDocumentTransformer
import torch.nn.functional as F
from torch.cuda import amp

def build_dataloader() -> DataLoader:
    ds = QueryDocumentOrcasDataset(split="tiny")
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=16)
    return dl

def build_tokenizer(tokenizer_path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

def create_transformer(vocab: Dict) -> QueryDocumentTransformer:
    return QueryDocumentTransformer(
        vocab_size=len(vocab.keys()),
    )

def create_mlm_head(vocab: Dict) -> MLM_head:
    mlm_config = MLM_Config(
        name="MLM_Pervers",
        input_dim=len(vocab.keys()),
        output_dim=2,
        dropout=0.1,
        mask_prob=0.15,
        num_tokens=len(vocab.keys()),
        mask_token_id=4,
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3
    )

    return MLM_head(config=mlm_config)

def create_sop_head() -> SentenceOrderPrediction:
    sop_config = SOPConfig(
        name="SOP_Pervers",
        input_dim=768,
        output_dim=2,
    )

    return SentenceOrderPrediction(config=sop_config)

def pretraining_step(
                    dl: DataLoader, 
                    tokenizer: Tokenizer, 
                    mlm_head: MLM_head, 
                    sop_head: SentenceOrderPrediction,
                    transformer: QueryDocumentTransformer,
                    optimizer: torch.optim.Optimizer,
                ):

    max_len = 512
    for idx, batch in enumerate(dl):
        q_batch, d_batch = batch
        
        # Tokenize the batch
        tok_q_batch = tokenizer.encode_batch(q_batch, is_pretokenized=False, add_special_tokens=True)
        tok_d_batch = tokenizer.encode_batch(d_batch, is_pretokenized=False, add_special_tokens=True)

        # Prepare qs
        q_inputs, q_targets = mlm_head.prepare_inputs(inputs=[torch.tensor(t.ids) for t in tok_q_batch], max_len=256)

        # Prepare ds
        seqs = [t.ids for t in tok_d_batch]
        for idx, s in enumerate(seqs):
            start = random.randint(1, max(len(s) - max_len - 1, 1))
            seqs[idx] = s[start:start + max_len - 1]
            if seqs[idx][-1] == 3:
                seqs[idx][-1] = 0 # foce final token to be a pad token
            
            # Pad the sequence
            seqs[idx] = [2] + seqs[idx] + [0] * (max_len - len(seqs[idx]) - 1)
            
        sep_list = set([18, 35, 5])
        seqs = [torch.tensor([tok if tok not in sep_list else 3  for tok in s]) for s in seqs]

        inputs = torch.stack(seqs, dim=0)
        print(inputs.shape)
        d_mask = (inputs == 0).long()
        d_inputs, d_targets = sop_head.prepare_inputs(inputs=inputs)

        print(d_inputs.shape)



def contrastive_step(dl: DataLoader, tok: Tokenizer):
    pass

if __name__ == "__main__":
    dl = build_dataloader()
    tokenizer = build_tokenizer(tokenizer_path="./BERT_tok-trained.json")
    mlm_head = create_mlm_head(vocab=tokenizer.get_vocab())
    sop_head = create_sop_head()
    transformer = create_transformer(vocab=tokenizer.get_vocab())
    optimizer = torch.optim.RAdam(
        list(transformer.parameters()) + \
        list(mlm_head.parameters()) + \
        list(sop_head.parameters()),
        lr=1e-4
    )

    pretraining_step(
        dl=dl, 
        tokenizer=tokenizer, 
        mlm_head=mlm_head, 
        sop_head=sop_head, 
        transformer=transformer, 
        optimizer=optimizer
    )


