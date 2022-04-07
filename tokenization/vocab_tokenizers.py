import sys
import os
sys.path.append(os.getcwd())

import torch
import tokenizers
import sklearn
from tokenizers import SentencePieceBPETokenizer
from tokenizers import SentencePieceUnigramTokenizer
from tokenizers import BertWordPieceTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer, BpeTrainer, UnigramTrainer
# whitespace pretokenizer ?
from tokenizers.pre_tokenizers import Whitespace
# use bert pretokenizer
from typing import List

from dataset.squad.iterators import create_corpus_from_document_query_lists as squad_corpus
from dataset.squad.iterators import create_query_document_lists_squad as squad_query_document_list


unk_token = "<UNK>"
spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]

def is_filepath_list(filelist: List[str]) -> bool:
    """
    Check if a list of filepaths is a list of files.
    """
    for file in filelist:
        if not os.path.isfile(file):
            return False
    return True

def train_iterator_mul_files(files):
    for path in files:
        with open(path, "r") as f:
            for line in f:
                yield line

def train_WordPieceTokenizer(file_list: List[str], vocab_size=30_000, min_frequency=5, limit_alphabet=500, save: bool=True):
    """
    Train WP tokenizer from a list of files.
    """
    tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=spl_tokens,
        show_progress=True,
        limit_alphabet=limit_alphabet
    )
    tokenizer.pre_tokenizer = Whitespace()
    
    if is_filepath_list(file_list):
        tokenizer.train(file_list, trainer=trainer)
    else:
        trainer.train_from_iterator(file_list, trainer=trainer)
    
    if save:
        tokenizer.save("./WP_tok-trained.json")
        tokenizer = Tokenizer.from_file("./WP_tok-trained.json")
    return tokenizer

def train_SentencePieceBPETokenizer(files: List[str], vocab_size=30_000, min_frequency=5, limit_alphabet=500, save: bool=True):
    """
    trin SP_BPE tokenizer from a list of files.
    """
    if is_filepath_list(files):
        train_it = train_iterator_mul_files(files)
    else:
        train_it = files

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train_from_iterator(
        train_it,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        limit_alphabet=limit_alphabet,
    )

    if save:
        tokenizer.save("./SP_BPE_tok-trained.json")
        tokenizer = Tokenizer.from_file("./SP_BPE_tok-trained.json")
    return tokenizer  

def train_SentencePieceUGTokenizer(filelist: List[str], vocab_size=30_000, save: bool=True):
    """
    trin SP_UG tokenizer from a list of files.
    """
    if is_filepath_list(filelist):
        train_it = train_iterator_mul_files(filelist)
    else:
        train_it = filelist

    tokenizer = SentencePieceUnigramTokenizer()
    tokenizer.train_from_iterator(
        train_it,
        vocab_size=vocab_size,
        show_progress=True
    )
    if save:
        tokenizer.save("./SP_UG_tok-trained.json")
        tokenizer = Tokenizer.from_file("./SP_UG_tok-trained.json")
    return tokenizer  

def train_BertWordPieceTokenizer(filelist: List[str], vocab_size=30_000, min_frequency=5, limit_alphabet=500, save: bool=True):
    """
    trin BERT tokenizer from a list of files.
    """
    if is_filepath_list(filelist):
        train_it = train_iterator_mul_files(filelist)
    else:
        train_it = filelist

    tokenizer = BertWordPieceTokenizer()
    tokenizer.normalizer = tokenizers.normalizers.BertNormalizer(strip_accents=True, lowercase=True)
    tokenizer.train_from_iterator(
        train_it,
        vocab_size=vocab_size,
        show_progress=True,
        min_frequency=min_frequency,
        limit_alphabet=limit_alphabet,
    )
    
    if save:
        tokenizer.save("./BERT_tok-trained.json")
        tokenizer = Tokenizer.from_file("./BERT_tok-trained.json")
    return tokenizer

def get_vocab_from_tokenizer(tokenizer: Tokenizer):
    """
    Get vocab from tokenizer.
    """
    vocab = tokenizer.get_vocab()
    return vocab

if __name__ == '__main__':
    # create corpus
    queries, documents = squad_query_document_list()
    corpus = squad_corpus(documents, queries)
    tokenizer = train_BertWordPieceTokenizer(corpus, vocab_size=30_000)
