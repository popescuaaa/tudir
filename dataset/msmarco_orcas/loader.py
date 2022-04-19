from collections import defaultdict
import gzip
import random
import os
from typing import Any, Dict, List, Tuple, Optional
from pip import main
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
import ir_datasets
from tqdm import tqdm
import pickle
import random
 

class OrcasSplit:
    size_splits = {
        "tiny": 100,
        "small": 100_000,
        "medium": 1_000_000,
        "large": 2_500_000,
        "all": 100_000_000,
    }

    def __init__(
        self,
        split: str,
        overwrite_cache: bool=False,
        overwrite_corpus: bool=False,
        dataset_dir: str="./.dataset_caches/orcas_compressed_cache",
        corpus_dir: str="./.corpus_caches/orcas"
    ) -> None:

        self.split = split
        self.data = ir_datasets.load("msmarco-document/orcas")

        self.corpus_dir = corpus_dir
        self.dataset_dir = dataset_dir
        
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        
        if not os.path.exists(self.corpus_dir):
            os.makedirs(self.corpus_dir)

        if not os.path.exists(self.dataset_cache_path) or overwrite_cache:
            self.build_split_cache(split)

        self.document_query_dict, self.query_text = self.load_split_cache()

        if not os.path.exists(self.corpus_files_path) or overwrite_corpus:
            self.build_corpus()

    @property
    def dataset_cache_path(self) -> str:
        return os.path.join(self.dataset_dir, f"d_cache_{self.split}.pickle.gz")

    @property
    def corpus_files_path(self) -> str:
        return os.path.join(self.corpus_dir, f"{self.split}")

    def build_split_cache(self, split: str) -> None:
        document_query_dict = defaultdict(list)
        query_ids = set()

        qrels_iter = self.data.qrels_iter()
        for i in tqdm(range(self.size_splits[split]), desc="doc query dict"):
            qrel = next(qrels_iter)

            document_query_dict[qrel.doc_id].append(qrel.query_id)
            query_ids.add(qrel.query_id)

        query_text = {}
        for query in tqdm(self.data.queries_iter(), total=self.data.queries_count(), desc="associated query text"):
            if query.query_id in query_ids:
                query_text[query.query_id] = query.text

        with gzip.open(self.dataset_cache_path, 'wb') as f:
            pickle.dump(document_query_dict, f)
            pickle.dump(query_text, f)

    def build_corpus(self) -> str:
        if not os.path.exists(self.corpus_files_path):
            os.makedirs(self.corpus_files_path)

        # iterate over queries
        queries_path = os.path.join(self.corpus_files_path, f"queries.txt")
        with open(queries_path, 'w') as f:
            for query_id, query_text in tqdm(self.query_text.items(), total=len(self.query_text), desc="queries"):
                f.write(query_text + '\n')

        # load docstore
        doc_store = self.data.docs_store()

        # iterate over documents
        for doc_id in tqdm(self.document_query_dict, total=len(self.document_query_dict), desc="documents"):
            doc_path = os.path.join(self.corpus_files_path, f"document_{doc_id}.txt")
            # lod document item from docstore
            doc_item = doc_store.get(doc_id)
            with open(doc_path, 'w') as f:
                f.write(doc_item.body)

        return self.corpus_files_path

    def load_split_cache(self) -> Tuple[Dict[int, int], Dict[int, str]]:
        with gzip.open(self.dataset_cache_path, 'rb') as f:
           document_query_dict = pickle.load(f)
           query_text = pickle.load(f)
        return document_query_dict, query_text

 

class QueryDocumentOrcasDataset(Dataset): 
    def __init__(self, split: str) -> None:
        self.orcas_dataset = OrcasSplit(split)
        self.doc_keys = list(self.orcas_dataset.document_query_dict.keys())
        docs_store = self.orcas_dataset.data.docs_store()

        self.doc_body = {}

        for doc_id in tqdm(self.orcas_dataset.document_query_dict.keys(), total=len(self.orcas_dataset.document_query_dict), desc="Addings queries to documents index"):
            doc_tuple = docs_store.get(doc_id)
            self.doc_body[doc_id] = doc_tuple.body

    def __getitem__(self, index: int) -> Tuple[str, str]:
        # sample a document id
        doc_id = self.doc_keys[index]
        # select the queries for the document
        doc_queries = self.orcas_dataset.document_query_dict[doc_id]
        # sample a query id
        query_id = random.choice(doc_queries)
        # return the query text and document text
        return self.orcas_dataset.query_text[query_id], self.doc_body[doc_id]

    def __len__(self) -> int:
        return len(self.doc_keys)
