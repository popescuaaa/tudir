import itertools
import sys
import os
sys.path.append(os.getcwd())

from notebooks.vocab_tokenizers import train_BertWordPieceTokenizer
from dataset.squad.iterators import create_corpus_from_document_query_lists as squad_corpus
from dataset.squad.iterators import create_query_document_lists_squad as squad_query_document_list

from collections import defaultdict
from typing import Dict, List, Tuple
from plot import plot_distribution, plot_cummulative_dsitribution
import numpy as np
from tokenization.corpus_tokenizers import CorpusTokenizer, HuggingFaceCorpusTokenizer
from tqdm import tqdm
import math


def compute_corpus_pmis(corpus: List[str], corpus_tokenizer: CorpusTokenizer) -> Dict[Tuple[str, str], float]:
    """
    Computes PMI scores for corpus.
    :param corpus: corpus
    :param corpus_tokenizer: corpus tokenizer
    :return: PMI scores
    """

    # compute PMI scores
    new_corpus = corpus_tokenizer.tokenize_corpus(corpus)

    bigram_count: Dict[Tuple[str, str], int] = defaultdict(int)
    token_count:  Dict[str, int]  = defaultdict(int)

    # if using zip, need to do go line by line
    for line_token_list in tqdm(new_corpus, total=len(new_corpus)):
        for token_a, token_b in zip(line_token_list, line_token_list[1:]):
            # increment bigram count and token count
            bigram_count[(token_a, token_b)] += 1
            token_count[token_a] += 1
            token_count[token_b] += 1

    # sum up the bigram counts and token counts
    total_token_counts = sum(token_count.values())
    total_bigram_counts = total_token_counts - 1
    squared_token_counts = total_token_counts ** 2

    pmi_scores = {}

    # compute pmi scores
    for (token_a, token_b), count in bigram_count.items():
        # normalize counts into probabilities
        bigram_prob = count / total_bigram_counts
        token_a_prob = token_count[token_a] 
        token_b_prob = token_count[token_b]
        # set PMI for bigram
        pmi_scores[(token_a, token_b)] = math.log(bigram_prob / (token_a_prob * token_b_prob / squared_token_counts))

    return pmi_scores

def compute_corpus_set_pmis(corpus: List[str], corpus_tokenizer: CorpusTokenizer) -> Dict[Tuple[str, str], float]:
    """
    Compute PMI scores for corpus with a twist. Instead of computing PMI using a bigram, it computes PMI using any 2-token set from the sequence.
    """
    new_corpus = corpus_tokenizer.tokenize_corpus(corpus)

    bigram_count: Dict[Tuple[str, str], int] = defaultdict(int)
    token_count:  Dict[str, int]  = defaultdict(int)

    # if using zip, need to do go line by line
    for line_token_list in tqdm(new_corpus, total=len(new_corpus)):
        combination_tokkens = set()
        for token_a, token_b in zip(line_token_list, line_token_list[1:]):
            token_count[token_a] += 1
            token_count[token_b] += 1
            # add permutable tokens to set
            combination_tokkens.add(token_a)
            combination_tokkens.add(token_b)
        # generate all 2-token sets
        combinations = itertools.combinations(combination_tokkens, 2)
        for combination in combinations:
            bigram_count[combination] += 1

    # sum up the bigram counts and token counts
    total_token_counts = sum(token_count.values())
    total_bigram_counts = sum(bigram_count.values())
    squared_token_counts = total_token_counts ** 2

    pmi_scores = {}

    # compute pmi scores
    for (token_a, token_b), count in bigram_count.items():
        # normalize counts into probabilities
        bigram_prob = count / total_bigram_counts
        token_a_prob = token_count[token_a] 
        token_b_prob = token_count[token_b]
        # set PMI for bigram
        pmi_scores[(token_a, token_b)] = math.log(bigram_prob / (token_a_prob * token_b_prob / squared_token_counts))

    return pmi_scores


def compute_pmi_distribution(pmi_scores: Dict[Tuple[str, str], float]) -> Dict[float, float]:
    """Function that computer the probability distribution for each PMI score"""

    # sort scores by value
    sorted_pmi = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)
    # take only floats
    sorted_scores = [x[1] for x in sorted_pmi]
    cummulative_divisor = 0.
    # compute score counts
    score_probabilities: Dict[float, float] = defaultdict(float)
    for score in sorted_scores:
        cummulative_divisor += 1
        score_probabilities[score] += 1

    # compute probability for each score
    for score in score_probabilities:
        score_probabilities[score] /= cummulative_divisor
    
    # sort probabilities by pmi_score
    sorted_probabilities = {k: v for (k, v) in sorted(score_probabilities.items(), key=lambda x: x[0])}
    return sorted_probabilities

def linspace_distribution_bin(distribution: Dict[float, float], bin_range: Tuple[float, float] = None, step_size: float = 0.1) -> Dict[Tuple[float, float], float]:
    """Function that bins the distribution into bins of linear space size"""
    def _bin_hash(x: float) -> int:
        # small function to get the bin index
        return int((x - bin_range[0]) / step_size)

    # if range is not specified, use the distribution range
    if bin_range is None:
        bin_range = (min(distribution.keys()), max(distribution.keys()))
    
    # compute bin ranges
    bin_ranges = list(zip(np.arange(bin_range[0], bin_range[1], step_size), np.arange(bin_range[0] + step_size, bin_range[1] + step_size, step_size)))
    bins = [0. for _ in bin_ranges]

    # sum up the distribution in each bin
    for score, prob in distribution.items():
        bins[_bin_hash(score)] += prob

    # convert to dict
    bins = {(bin_ranges[i][0], bin_ranges[i][1]): bins[i] for i in range(len(bins))}

    # sanity check that the sum of the bins is 1
    assert abs(sum(bins.values()) - 1) < 0.00001, "Bin sum is not 1, sum: {}".format(sum(bins.values()))

    return bins

if __name__ == '__main__':
    # create corpus
    queries, documents = squad_query_document_list()
    corpus = squad_corpus(documents, queries)
    # create tokenizer
    tokenizer = train_BertWordPieceTokenizer(corpus, vocab_size=30_000)
    corpus_tokenizer = HuggingFaceCorpusTokenizer(tokenizer)
    # compute pmi scores
    pmi_scores = compute_corpus_set_pmis(corpus, corpus_tokenizer)
    # sort scores by value
    sorted_pmi = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)
    # return top 10 scores
    print(sorted_pmi[:50])
    print(pmi_scores)
    exit(0)
    # compute probability distribution
    probability_distribution = compute_pmi_distribution(pmi_scores)
    # bin distribution
    binned_distribution = linspace_distribution_bin(probability_distribution, step_size=0.2)
    # print distribution
    for bin_range, cummulative_prob in binned_distribution.items():
        print(f'{bin_range}: {cummulative_prob}')
    
    # plot distribution
    plot_distribution(values=[k[1] for k in binned_distribution.keys()], probs=[v for v in binned_distribution.values()])
    # plot cummulative distribution
    plot_cummulative_dsitribution(values=[k[1] for k in binned_distribution.keys()], probs=[v for v in binned_distribution.values()])

