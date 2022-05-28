from collections import defaultdict
from email.policy import default
import random
from re import I

from sklearn.preprocessing import scale
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
import wandb
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
scaler = amp.GradScaler()

def build_dataloader() -> DataLoader:
    ds = QueryDocumentOrcasDataset(split="all")
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=16)
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
        name="MLM_Head",
        input_dim=768,
        output_dim=len(vocab.keys()),
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
    with amp.autocast(enabled=True):
        for epoch in range(100):
            loss_dict = defaultdict(float)
            for idx, batch in tqdm(enumerate(dl), total=len(dl), desc="Epoch {}".format(epoch)):
                q_batch, d_batch = batch

                optimizer.zero_grad()
                
                # Tokenize the batch
                tok_q_batch = tokenizer.encode_batch(q_batch, is_pretokenized=False, add_special_tokens=True)
                tok_d_batch = tokenizer.encode_batch(d_batch, is_pretokenized=False, add_special_tokens=True)

                # Prepare qs
                q_inputs, q_targets = mlm_head.prepare_inputs(inputs=[torch.tensor(t.ids) for t in tok_q_batch], max_len=max_len)

                # Prepare ds
                seqs = [t.ids for t in tok_d_batch]
                d_inputs, d_targets = sop_head.prepare_inputs(inputs=seqs, max_len=max_len)

                # Move to GPU
                q_inputs = q_inputs.cuda()
                q_targets = q_targets.cuda()
                d_inputs = d_inputs.cuda()
                d_targets = d_targets.cuda()

                # Task indexes
                task_indexes = {
                    "mlm": [0, len(q_batch)],
                    "sop": [len(q_batch), 2*len(q_batch)]
                }

                # Prepare the transformer
                emb_ids = torch.cat([q_inputs, d_inputs], dim=0)
                with torch.no_grad():
                    mask = (emb_ids == 0).float() # pad ids

                embs = transformer(emb_ids, None, mask, task_indexes)

                # Compute losses
                losses = {
                    "mlm": mlm_head(embs[:len(q_batch)], q_targets)[1],
                    "sop": sop_head(embs[len(q_batch):], d_targets)[1]
                }

                report_loss = {
                    k: v.detach().cpu().item() for k, v in losses.items()
                }

                for k, v in losses.items():
                    loss_dict[k] += v.detach().cpu().item()

                loss = sum(losses.values())
                unscaled_loss = loss
                loss = scaler.scale(loss)
                loss.backward()
                scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(mlm_head.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(sop_head.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

                # if idx % 2 == 1:
                #     # print("Loss: ", unscaled_loss.item())
                #     print(report_loss)

                if idx % 100 == 0:
                    for k, v in report_loss.items():
                        report_loss[k] /= 100
                        print(f"{k}: {v}")
                        wandb.log({"{}_loss".format(k): v})
                    report_loss = defaultdict(float)

def contrastive_step(dl: DataLoader, tok: Tokenizer):
    # diagrama CLIP
    pass

if __name__ == "__main__":
    run_name = "SSL_Query_doc"
    wandb.init(config={}, project='_ssl_', name=run_name)

    dl = build_dataloader()
    tokenizer = build_tokenizer(tokenizer_path="./BERT_tok-trained.json")
    
    mlm_head = create_mlm_head(vocab=tokenizer.get_vocab()).cuda()
    sop_head = create_sop_head().cuda()
    transformer = create_transformer(vocab=tokenizer.get_vocab()).cuda()

    # Lr scheduler cosine annealing with cold restarts
    optimizer = torch.optim.RAdam(
        list(transformer.parameters()) + \
        list(mlm_head.parameters()) + \
        list(sop_head.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )

    pretraining_step(
        dl=dl, 
        tokenizer=tokenizer, 
        mlm_head=mlm_head, 
        sop_head=sop_head, 
        transformer=transformer, 
        optimizer=optimizer
    )

   


