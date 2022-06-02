from collections import defaultdict
from email.policy import default
import random
from re import I
import numpy as np
from numpy import arange
from sklearn import datasets
from sklearn.metrics import consensus_score
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
import timeit
import faiss

os.environ["TOKENIZERS_PARALLELISM"] = "false"
scaler = amp.GradScaler()

# Set cuda visible devices
torch.cuda.empty_cache()

device = torch.device("cuda:0")

# Set max_split_size 
def build_dataloader() -> DataLoader:
    ds = QueryDocumentOrcasDataset(split="small")
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=16)
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
        name="SOP_Head",
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

    # Create a dictionary for models
    if not os.path.exists("./models"):
        os.mkdir("./models")

    max_len = 256
    with amp.autocast(enabled=True):
        for epoch in range(3):
            loss_dict = defaultdict(float)
            for idx, batch in tqdm(enumerate(dl), total=len(dl), desc="Epoch {}".format(epoch)):
                q_batch, d_batch = batch

                optimizer.zero_grad()
                
                # Tokenize the batch
                tok_q_batch = tokenizer.encode_batch(q_batch, is_pretokenized=False, add_special_tokens=True)
                tok_d_batch = tokenizer.encode_batch(d_batch, is_pretokenized=False, add_special_tokens=True)

                # start = timeit.default_timer()
                # Prepare qs
                q_inputs, q_targets = mlm_head.prepare_inputs(inputs=[torch.tensor(t.ids) for t in tok_q_batch], max_len=max_len)
                # stop = timeit.default_timer()
                # print('MLM Head prepare: ', stop - start)  
              
                
                # start = timeit.default_timer()
                # Prepare ds
                seqs = [t.ids for t in tok_d_batch]
                d_inputs, d_targets = sop_head.prepare_inputs(inputs=seqs, max_len=max_len)
                # stop = timeit.default_timer()
                # print('SOP Head prepare: ', stop - start)  

                # Move to GPU
                q_inputs = q_inputs.long().to(device)
                q_targets = q_targets.long().to(device)
                d_inputs = d_inputs.long().to(device)
                d_targets = d_targets.long().to(device)

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

                if idx == len(q_batch) - 1:
                    for k, v in report_loss.items():
                        report_loss[k] /= 100
                        print(f"{k}: {v}")
                        # wandb.log({"{}_loss".format(k): v})
                    report_loss = defaultdict(float)
    
                    # Save the model
                    transformer = transformer.cpu()
                    torch.save(transformer.state_dict(), "./models/transformer_pretrain_{}.pth".format(epoch))
                    transformer = transformer.to(device)

                    mlm_head = mlm_head.cpu()
                    torch.save(mlm_head.state_dict(), "./models/mlm_head_pretrain_{}.pth".format(epoch))
                    mlm_head = mlm_head.to(device)

                    sop_head = sop_head.cpu()
                    torch.save(sop_head.state_dict(), "./models/sop_head_pretrain_{}.pth".format(epoch))
                    sop_head = sop_head.to(device)

def contrastive_step(
                    dl: DataLoader, 
                    tokenizer: Tokenizer,
                    mlm_head: MLM_head,
                    sop_head: SentenceOrderPrediction,
                    transformer: QueryDocumentTransformer,
                    optimizer: torch.optim.Optimizer,
                ):
    
    # Create a directory for pretrained models
    if not os.path.exists("./models"):
        os.mkdir("./models")

    loss_fn = torch.nn.CrossEntropyLoss()
    max_len = 512
    with amp.autocast(enabled=True):
        for epoch in range(3):
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
                q_inputs = q_inputs.to(device)
                q_targets = q_targets.to(device)
                d_inputs = d_inputs.to(device)
                d_targets = d_targets.to(device)

                # Task indexes
                task_indexes = {
                    "mlm": [0, len(q_batch)],
                    "sop": [len(q_batch), 2*len(q_batch)]
                }

                emb_ids = torch.cat([q_inputs, d_inputs], dim=0)
                with torch.no_grad():
                    mask = (emb_ids != 0).float() # pad ids

                embs = transformer(emb_ids, None, mask, task_indexes)

                qs = embs[:len(q_batch)]
                ds = embs[len(q_batch):]

                # Split mask
                q_mask = mask[:len(q_batch)]
                d_mask = mask[len(q_batch):]

                with torch.no_grad():
                    qs_n_tokens = q_mask.sum(dim=1)
                
                with torch.no_grad():
                    ds_n_tokens = d_mask.sum(dim=1)

                with torch.no_grad():
                    d_cls_mask = (emb_ids[len(q_batch):] == 2).float()
                    d_cls_n_token = d_cls_mask.sum(dim=1)

                # # Compute average embedding
                qs = (qs * q_mask.unsqueeze(2)).sum(dim=1) / qs_n_tokens.unsqueeze(1)
                ds = (ds * d_cls_mask.unsqueeze(2)).sum(dim=1) / d_cls_n_token.unsqueeze(1)               
                
                # Compute loss
                qs_ds = torch.bmm(qs.unsqueeze(0), ds.unsqueeze(0).permute(0, 2, 1))
                qs_ds = qs_ds.squeeze(0)
                targets = (torch.arange(len(q_batch)) + 1) * torch.eye(len(q_batch))

                targets = targets.to(device)
                
                contrastive_loss = loss_fn(qs_ds, targets)
                contrastive_loss.backward()
                optimizer.step()

                if (idx + 1) % 20 == 1:
                    print("Contrastive loss: ", contrastive_loss.item())
                    print("Contrastive loss: {}".format(contrastive_loss.item()))
                    wandb.log({"Contrastive_loss": contrastive_loss.item()})

                if idx == len(q_batch) - 1:
                    
                    # Save the model
                    transformer = transformer.cpu()
                    torch.save(transformer.state_dict(), "./models/transformer_fine_{}.pth".format(epoch))
                    transformer = transformer.to(device)

                    mlm_head = mlm_head.cpu()
                    torch.save(mlm_head.state_dict(), "./models/mlm_head_fine_{}.pth".format(epoch))
                    mlm_head = mlm_head.to(device)

                    sop_head = sop_head.cpu()
                    torch.save(sop_head.state_dict(), "./models/sop_head_fine_{}.pth".format(epoch))
                    sop_head = sop_head.to(device)

def evaluate_model(
    tokenizer: Tokenizer,
    transformer: QueryDocumentTransformer,
    mlm_head: MLM_head,
    sop_head: SentenceOrderPrediction
):


    ds_test = QueryDocumentOrcasDataset(split="small")
    dl_test = DataLoader(ds_test, batch_size=4, shuffle=False, num_workers=0)
    
    max_len = 512
    
    mlm_head.eval()
    sop_head.eval()
    transformer.eval()

    
    d_embs = []
    q_d_dict = {}
    query_doc_index = 0
    for idx, batch in tqdm(enumerate(dl_test), total=len(dl_test), desc="Evaluating"):
        q_batch, d_batch = batch

        # Tokenize the batch
        tok_q_batch = tokenizer.encode_batch(q_batch, is_pretokenized=False, add_special_tokens=True)
        tok_d_batch = tokenizer.encode_batch(d_batch, is_pretokenized=False, add_special_tokens=True)

        # Prepare qs
        _, q_input = mlm_head.prepare_inputs(inputs=[torch.tensor(t.ids) for t in tok_q_batch], max_len=max_len)

        # Prepare ds
        seqs = [t.ids for t in tok_d_batch]
        d_input, _ = sop_head.prepare_inputs(inputs=seqs, max_len=max_len)

        # Move to GPU
        q_input = q_input.to(device)
        d_input = d_input.to(device)

        # Task indexes
        task_indexes = {
            "mlm": [0, len(q_batch)],
            "sop": [len(q_batch), 2*len(q_batch)]
        }

        # print("q_inp: ", q_input.shape)
        # print("d_inp: ", d_input.shape)

        emb_ids = torch.cat([q_input, d_input], dim=0)
        with torch.no_grad():
            mask = (emb_ids != 0).float() # pad ids

        embs = transformer(emb_ids, None, mask, task_indexes)

        qs = embs[:len(q_batch)]
        ds = embs[len(q_batch):]

        # Split mask
        q_mask = mask[:len(q_batch)]
        d_mask = mask[len(q_batch):]

        with torch.no_grad():
            qs_n_tokens = q_mask.sum(dim=1)
        

        with torch.no_grad():
            d_cls_mask = (emb_ids[len(q_batch):] == 2).float()
            d_cls_n_token = d_cls_mask.sum(dim=1)

        # # Compute average embedding
        qs = (qs * q_mask.unsqueeze(2)).sum(dim=1) / qs_n_tokens.unsqueeze(1)
        ds = (ds * d_cls_mask.unsqueeze(2)).sum(dim=1) / d_cls_n_token.unsqueeze(1)

        # print("qs_shape", qs.shape)
        # print("ds_shape", ds.shape)

        q_emb_faiss = qs.detach().cpu().numpy()
        d_emb_faiss = ds.detach().cpu().numpy()

        for i in range(len(q_batch)):
            q_d_dict[query_doc_index] = q_emb_faiss[i]
            d_embs.append(d_emb_faiss[i])
            query_doc_index += 1


        # dict: idx: (q_emb) // idx should be the same for the document in faiss

        

    print("Final index:", query_doc_index)
    print("Len of d_embs:", len(d_embs))

    index = faiss.IndexFlatL2(768)
    embs = np.vstack(d_embs)
    index.train(embs)
    index.add(embs)
    # faiss.write_index(index, "./data/index_small.faiss")

    
    for k in [5, 10, 20, 50, 100]:
        true_cnt = 0
        for qd_idx, q_emb in q_d_dict.items():
            # print("Qemb:", q_emb.shape)
            D, I = index.search(np.expand_dims(q_emb, axis=0), k)
            # print("D", D)
            # print("qd_idx", qd_idx)
            # print(qd_idx in I[0])
            # print("I", I)
            if i in I[0]:
                # print(np.where(I[0] == i)[0])
                true_cnt += 1
        print("Top {}: {}".format(k, true_cnt/len(q_d_dict)))
        
     


        
if __name__ == "__main__":
    # run_name = "contrast_transformer"
    # wandb.init(config={}, project='_ssl_', name=run_name)

    # dl = build_dataloader()
    # tokenizer = build_tokenizer(tokenizer_path="./BERT_tok-trained.json")
    
    # mlm_head = create_mlm_head(vocab=tokenizer.get_vocab()).to(device=device)
    # sop_head = create_sop_head().to(device=device)
    # transformer = create_transformer(vocab=tokenizer.get_vocab()).to(device=device)

    # # Lr scheduler cosine annealing with cold restarts
    # optimizer = torch.optim.RAdam(
    #     list(transformer.parameters()) + \
    #     list(mlm_head.parameters()) + \
    #     list(sop_head.parameters()),
    #     lr=1e-4,
    #     weight_decay=1e-5
    # )

    # pretraining_step(
    #     dl=dl, 
    #     tokenizer=tokenizer, 
    #     mlm_head=mlm_head, 
    #     sop_head=sop_head, 
    #     transformer=transformer, 
    #     optimizer=optimizer
    # )
        
    # Set cuda visible devices
    # torch.cuda.empty_cache()

    # dl = build_dataloader()
    # tokenizer = build_tokenizer(tokenizer_path="./BERT_tok-trained.json")

    # pretrained_mlm = create_mlm_head(vocab=tokenizer.get_vocab())
    # pretrained_mlm.load_state_dict(torch.load("./models/mlm_head_pretrain_2.pth"))
    # pretrained_mlm = pretrained_mlm.to(device=device)

    # pretrained_sop = create_sop_head()
    # pretrained_sop.load_state_dict(torch.load("./models/sop_head_pretrain_2.pth"))
    # pretrained_sop = pretrained_sop.to(device=device)

    # pretrained_transformer = create_transformer(vocab=tokenizer.get_vocab())
    # pretrained_transformer.load_state_dict(torch.load("./models/transformer_pretrain_3.pth"))
    # pretrained_transformer = pretrained_transformer.to(device=device)

    # optimizer = torch.optim.RAdam(
    #     list(pretrained_transformer.parameters()) + \
    #     list(pretrained_mlm.parameters()) + \
    #     list(pretrained_sop.parameters()),
    #     lr=1e-4,
    #     weight_decay=1e-5
    # )

    # contrastive_step(
    #     dl= dl, 
    #     tokenizer=tokenizer, 
    #     mlm_head=pretrained_mlm, 
    #     sop_head=pretrained_sop, 
    #     transformer=pretrained_transformer, 
    #     optimizer=optimizer
    # )

    tokenizer = build_tokenizer(tokenizer_path="./BERT_tok-trained.json")

    fine_mlm = create_mlm_head(vocab=tokenizer.get_vocab())
    fine_mlm.load_state_dict(torch.load("./models/mlm_head_fine_2.pth"))
    fine_mlm = fine_mlm.to(device=device)

    fine_sop = create_sop_head()
    fine_sop.load_state_dict(torch.load("./models/sop_head_fine_2.pth"))
    fine_sop = fine_sop.to(device=device)

    fine_transformer = create_transformer(vocab=tokenizer.get_vocab())
    fine_transformer.load_state_dict(torch.load("./models/transformer_fine_2.pth"))
    fine_transformer = fine_transformer.to(device=device)

    evaluate_model(
        tokenizer=tokenizer,
        transformer=fine_transformer,
        mlm_head=fine_mlm,
        sop_head=fine_sop
    )


   


