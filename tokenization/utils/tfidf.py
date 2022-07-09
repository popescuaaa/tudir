import math
import sys
import os

import numpy as np
sys.path.append(os.getcwd())
from typing import Dict, List, Set, Union
from tqdm import tqdm
from tokenization.corpus_tokenizers import HuggingFaceCorpusTokenizer
from notebooks.vocab_tokenizers import train_BertWordPieceTokenizer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from dataset.squad.iterators import create_corpus_from_document_query_lists as squad_corpus
from dataset.squad.iterators import create_query_document_lists_squad as squad_query_document_list

class SparseTFidfVectorizer:
    def __init__(self, corpus: List[str], vocabulary: Dict = None, chunk_size: int = 10000) -> None:
        self.vectorizer = SparseTFidfVectorizer.get_tfidf_vectorizer(corpus, vocabulary)
        feature_names = self.vectorizer.get_feature_names_out()
        self.feature_indices = {i: feature_names[i] for i in range(len(feature_names))}
        self.chunk_size = chunk_size

    def get_scores(self, text: Union[str, List[str]]) -> Union[Dict[str, float], List[Dict[str, float]]]:
        is_array = False
        if isinstance(text, str):
            text = [text]
            is_array = True
        
        scores = []
        print('Computing TF-IDF scores for input text...')
        try:
            for i in tqdm(range(0, len(text), self.chunk_size), total=math.ceil(len(text) // self.chunk_size)):
                chunk_scores = self.vectorizer.transform(text[i:i+self.chunk_size]).toarray()
                scores.append(chunk_scores)
        except np.core._exceptions._ArrayMemoryError:
            print('Error: Memory error. Please try modifying the chunk size.')
            sys.exit(1)

        try: 
            indexed_scores = []
            print('Compressing sparse matrices...')
            for score_ndarray in tqdm(scores, total=len(scores)):
                score_sparse_array   = sparse.coo_matrix(score_ndarray)
                score_sparse_n_rows  = score_sparse_array.shape[0]
                for idx in range(score_sparse_n_rows):
                    sparse_scores = {
                        self.feature_indices[i]: v for i, v in 
                        zip(score_sparse_array.getrow(idx).indices, score_sparse_array.getrow(idx).data)
                    }
                    indexed_scores.append(sparse_scores)
        except np.core._exceptions._ArrayMemoryError:
            print('Error: Memory error. Score array too large. Please try using a paged approach for processing your text.')
            sys.exit(1)

        if is_array:
            indexed_scores = indexed_scores[0]
        return indexed_scores

    @classmethod
    def get_tfidf_vectorizer(cls, corpus: List[str], vocabulary: Dict) -> TfidfVectorizer:
        """Function that computes a tfidf vectorizer for a corpus"""
        if os.path.isfile(corpus[0]):
            input_params = {'input': 'filename'}
        else:
            input_params = {'input': 'content'}

        tfidf_vectorizer = TfidfVectorizer(input=input_params['input'], vocabulary=vocabulary, ngram_range=(1, 2))    
        tfidf_vectorizer.fit(corpus)
        return tfidf_vectorizer

def top_k_tfidf(text: Union[str, List[str]], vectorizer: SparseTFidfVectorizer, k: int) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """Function that computes the top k tfidf scores for a list of scores"""
    is_array = True
    if isinstance(text, str):
        text = [text]
        is_array = False
    
    # get tfidf feature names
    tfidf_scores = vectorizer.get_scores(text)

    top_scores = []
    # for each text input
    print('Computing top {} TF-IDF scores for input text...'.format(k))
    for i in tqdm(range(len(text)), total=len(text)):
        # get word name and score
        token_names, token_tfidf = list(tfidf_scores[i].keys()), list(tfidf_scores[i].values())
        token_scores = list(zip(token_names, token_tfidf))
        # select top k scores
        token_scores = {k: v for k, v in sorted(token_scores, key=lambda x: x[1], reverse=True)[:k]}
        top_scores.append(token_scores)
    
    if not is_array:
        top_scores = top_scores[0]
    return top_scores

def rank_words_tfidf(text: Union[str, List[str]], vectorizer: SparseTFidfVectorizer) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """Function that ranks words in a text based on their tfidf scores"""
    is_array = True
    if isinstance(text, str):
        text = [text]
        is_array = False
    

    for i in tqdm(range(len(text)), total=len(text)):
        # get word name and score
        tfidf_scores = top_k_tfidf_summary(text[i], vectorizer, k=len(text[i]))
        token_names, token_tfidf = list(tfidf_scores.keys()), list(tfidf_scores.values())
        token_scores = list(zip(token_names, token_tfidf))
        # rank words
        token_scores = {k: v for k, v in sorted(token_scores, key=lambda x: x[1], reverse=True)}
        ranked_words.append(token_scores)
    
    if not is_array:
        ranked_words = ranked_words[0]
    return ranked_words

def top_k_tfidf_summary(text: Union[str, List[str]], vectorizer: SparseTFidfVectorizer, k: int) -> List[List[str]]:
    """Function that computes the top k tfidf scores for a list of scores"""
    is_array = True
    if isinstance(text, str):
        text = [text]
        is_array = False
    
    # get tfidf feature names
    top_tfidf_scores = top_k_tfidf(text, vectorizer, k)

    text_summaries = []
    for i, subtext in enumerate(text):
        summary = {}
        for word in subtext.split(' '):
            # if word is in top k scores
            if any([k in word.lower() for k in top_tfidf_scores[i].keys()]):
                if word not in summary:
                    summary[word] = len(summary)
        # sort the summary
        summary = {k: v for k, v in sorted(summary.items(), key=lambda x: x[1])}
        text_summaries.append(list(summary.keys()))

    if not is_array:
        text_summaries = text_summaries[0]
    return text_summaries

def mutual_set_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Function that computes the similarity between two sets"""
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))
    
    

if __name__ == '__main__':
    # create corpus
    queries, documents = squad_query_document_list()
    corpus = squad_corpus(documents, queries)
    tokenizer = train_BertWordPieceTokenizer(corpus, vocab_size=60_000)
    # flatten list of queries
    queries = [item for sublist in queries for item in sublist]
    corpus_tokenizer = HuggingFaceCorpusTokenizer(tokenizer)
    query_tokens = corpus_tokenizer.tokenize_corpus(queries, join_delimiter=' ')
    query_vectorizer = SparseTFidfVectorizer(query_tokens, vocabulary=tokenizer.get_vocab(), chunk_size=100)
    query_summaries = top_k_tfidf_summary(query_tokens[:200], query_vectorizer, 8)
    

    similarities = {}
    for i in range(100):
        for j in range(i, 100):
            if i != j:
                similarities[(i, j)] = mutual_set_similarity(set(query_summaries[i]), set(query_summaries[j]))
    
    # sort similarities
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    for (i, j), similarity in sorted_similarities[:50]:
        print(f'Mutual set similarity: {similarity} for ({i} and query {j})')
        print(f'TF-IDF summary: {query_summaries[i]} QUERY: {queries[i]}')
        print(f'TF-IDF summary: {query_summaries[j]} QUERY: {queries[j]}')
        print()
    

    
