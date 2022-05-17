from typing import List, Union
from tokenizers import Tokenizer

SPL_TOKENS = ["<UNK>", "<SEP>", "<MASK>", "<CLS>", "<PAD>", "<EOS>", "<SOS>"]


class CorpusTokenizer:
    def __init__(
            self
    ) -> None:
        super().__init__()

    def tokenize_corpus(self, corpus: List[str], join_delimiter: str = '') -> List[List[str]]:
        """
        Split a corpus into lists of tokens
        """
        raise NotImplementedError()


class WhiteSpaceCorpusTokenizer(CorpusTokenizer):
    """Class for tokenizing a corpus by simply splitting on whitespace all possible symbols"""

    def __init__(self) -> None:
        super().__init__()

    def __process_text(self, text: str) -> List[str]:
        """
        Process a text by splitting it into tokens
        """

        # convert to lower case to avoid case-sensitive issues
        text = text.lower()

        # add spaces around punctuation and separators
        text = text.replace(",", " , ") \
            .replace(".", " . ") \
            .replace("?", " ? ") \
            .replace("!", " ! ") \
            .replace("'", " ' ") \
            .replace('"', ' " ') \
            .replace("(", " ( ") \
            .replace(")", " ) ") \
            .replace("[", " [ ") \
            .replace("]", " ] ") \
            .replace("-", " - ") \
            .replace("/", " / ") \
            .replace("\\", " \\ ") \
            .replace(";", " ; ") \
            .replace(":", " : ") \
            .replace("\n", " \n ") \
            .replace("\t", " \t ") \
 \
        # remember to filter out empty strings
        return list(filter(lambda x: x != '', text.split()))

    def tokenize_corpus(self, corpus: List[str], join_delimiter: str = '') -> Union[List[str], List[List[str]]]:
        """
        Split a corpus into lists of tokens
        """

        # Beyonce, was she born in 1981?
        # ["Beyonce", ",", "was"..., "?"]
        token_corpus = []
        for text in corpus:
            token_corpus.append(self.__process_text(text))

        if join_delimiter:
            token_corpus = [join_delimiter.join(tokens) for tokens in token_corpus]
        return token_corpus


class HuggingFaceCorpusTokenizer(CorpusTokenizer):
    """Class for tokenizing a corpus using the HuggingFace tokenizer. Asserts that the tokenizer is an already loaded / trained Tokenizer."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def tokenize_corpus(self, corpus: List[str], join_delimiter: str = '') -> Union[List[str], List[List[str]]]:
        """
        Split a corpus into lists of tokens
        """
        token_corpus = self.tokenizer.encode_batch(corpus)
        token_corpus = [c.tokens for c in token_corpus]
        # clean up special tokens
        for i, tokens in enumerate(token_corpus):
            token_corpus[i] = [t for t in tokens if t not in SPL_TOKENS]

        if join_delimiter:
            token_corpus = [join_delimiter.join(tokens) for tokens in token_corpus]

        return token_corpus
