import sys
import os
from typing import Tuple, List
import numpy as np
from tokenizers import BertWordPieceTokenizer, Tokenizer
from networks.task_heads.task_head import TaskHead, TaskConfig
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from concurrent import futures
from copy import deepcopy
from functools import partial
import random
from typing import List, Set, Tuple, Union
import numpy as np

sys.path.append(os.getcwd())

def truncate_pad_sentence(
    sentence: List[int],
    max_length: int,
    pad_id: int = 0,
    cls_id: int = 2,
) -> List[int]:
    if len(sentence) > max_length:
        start = random.randint(0, len(sentence) - max_length - 1)
        sentence = sentence[start:start + max_length]
        sentence[0] = cls_id
    else:
        sentence = sentence + [pad_id] * (max_length - len(sentence))
    return sentence

def process_sentence2(
    sentence: List[int],
    pad_id: int = 0,
    cls_id: int = 2,
    sep_id: int = 3
) -> Tuple[torch.Tensor, bool]:
    if len(sentence) == 0:
        sentence[0] = cls_id
        return sentence, 0

    # Compute pad length
    intial_len = len(sentence)

    # Remove padding
    sentence = [word for word in sentence if word != pad_id]

    t_sentence = torch.tensor(sentence[:-1], dtype=torch.long)
    # Find first pad_id in sentence
    pad_idx = t_sentence.eq(pad_id).nonzero()
    if pad_idx.size(0) > 0:
        pad_idx = pad_idx[0][0]
    else:
        pad_idx = len(sentence) - 1

    pad_len = intial_len - pad_idx - 1
    split_idx = random.randint(1, pad_idx)
    
    splitted_sentences = [t_sentence[:split_idx], t_sentence[split_idx:]]

    # Perform random shuffle
    shuffle = random.random() > 0.5
    if shuffle:
        splitted_sentences[0], splitted_sentences[1] = splitted_sentences[1], splitted_sentences[0]

    # Combine sentences
    t_sentence = torch.cat([splitted_sentences[0],  torch.tensor([sep_id]), splitted_sentences[1], torch.tensor([pad_id] * pad_len)], dim=0)
    
    return t_sentence, int(shuffle)


def process_sentence(
    sentence: List[int],
    pad_id: int = 0,
    cls_id: int = 2,
    sep_id: int = 3,
    delim: Set[int] = set([5]),
) -> Tuple[List[int], bool]:

    res = []
    acc = []

    for idx, word in enumerate(sentence[1:]):
        # break when we reach padding
        if word == pad_id:
            break
        # when we reach a delimiter, create a new sentence, append accumulator
        # if idx == split_idx:
        #     # replace delimiter with SEP token
        #     res.append(acc)
        #     acc = []
        # # otherwise increase current accumulator
        # else:
        acc.append(word)
    # if there are still words in the accumulator, append it
    if acc:
        res.append(acc)
    # shuffle the sentences
    
    if len(res) == 0 or len(res) == 1:
        sentence[0] = cls_id
        return sentence, 0
    elif len(res) > 1:
        split_idx = random.randint(1, len(res[0]) - 1)
        res = [res[0][:split_idx], res[0][split_idx:]]
        shuffle = random.random() > 0.5
        if shuffle:
            is_correct = False
            while not is_correct:
                perm = np.random.permutation(len(res))
                for i in range(len(res)):
                    if i != perm[i]: 
                        break
                if i != len(res) - 1:
                    is_correct = True
            
            res = [res[i] for i in perm]
    else:
        shuffle = False

    # add CLS token to first sentence
    res[0] = [cls_id] + res[0]
    # for i in range(len(res) - 1):
    #     res[i] += [sep_id]
    
    res[0][-1] = sep_id

    # flatten the sentences
    res = [word for sent in res for word in sent]
    
    # extend with padding
    if idx != len(sentence) - 1:
        # print("Pads", len(sentence[idx + 1:]))
        # res.extend(sentence[idx + 1: len(sentence) - len(res) - 1])
        # print("Len res 2", len(res))
        # Add padding
        res.extend([pad_id] * (len(sentence) - len(res)))

    return res, int(shuffle)

class Parser:
    def __init__(
        self,
        max_length: int,
        pad_id: int = 0,
        cls_id: int = 2,
        sep_id: int = 3,
        delim: List[int] = [5],
        num_workers: int = 1,
    ) -> None:
        self.max_length = max_length
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.delim = set(delim)
        self.num_workers = num_workers

        self.__pad_func = partial(truncate_pad_sentence, max_length=max_length, pad_id=pad_id, cls_id=cls_id)
        self.__split_func = partial(process_sentence, pad_id=pad_id, cls_id=cls_id, sep_id=sep_id, delim=self.delim)

    def parse(self, sentence: Union[List[int], List[List]]) -> Tuple[List[int], List[int]]:
        if not isinstance(sentence[0], list):
            sentence = [sentence]

        with futures.ProcessPoolExecutor(max_workers=1) as executor:
            pad_res = list(executor.map(self.__pad_func, sentence))
            split_res = list(executor.map(self.__split_func, pad_res))
        
        sentences, shuffles = [], []
        for res, shuffle in split_res:
            sentences.append(res)
            shuffles.append(shuffle)
        if len(sentences) == 1:
            return sentences[0], shuffles[0]
        else:
            return sentences, shuffles

class SOPConfig(TaskConfig):
    """
    Config for the SOP head.
    """

    def __init__(self, name: str, input_dim: int, output_dim: int) -> None:
        super().__init__(name, input_dim, output_dim)
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim


class SentenceOrderPrediction(TaskHead):
    """
    Sentence order prediction head.
    """

    def __init__(self, config: TaskConfig, dropout: float = 0.1) -> None:
        super().__init__(config=config)
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.linear = nn.Linear(config.input_dim, config.output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.config = config

        self.sop_classifier = nn.Sequential(
            self.dropout,
            self.linear,
        )

        self.sep_token_id = 3
        self.cls_token_id = 2
        self.pad_token_id = 0

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = inputs[:, 0, :]
        output = self.sop_classifier(inputs)
        loss = self.loss(output, targets)
        return output, loss
    
    @torch.no_grad()
    def prepare_inputs(self, inputs: torch.Tensor, max_len: int) -> Tuple[Tensor, Tensor]:
        parser = Parser(max_length=max_len, pad_id=self.pad_token_id, cls_id=self.cls_token_id, sep_id=self.cls_token_id, delim=[18, 35, 5], num_workers=16)
        response = parser.parse(inputs)

        sentences, shuffles = response
        sentences = torch.tensor(sentences, dtype=torch.long)
        shuffles = torch.tensor(shuffles, dtype=torch.long)
        # sentences = torch.stack(sentences, dim=0)
        return sentences, shuffles

class SpanOrderPrediction(TaskHead):
    """
    Span Order prediction task: given text tokens: [t1, t2, .... t20] predict if
    text[j:k] is before or after text[i:j]    
    """

    def __init__(self, config):
        super().__init__(config)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def prepare_inputs(self, inputs: torch.Tensor, **kwargs):
        """
        Function that prepares inputs token ids for the task head objective

        Args:
            inputs: (batch_size, seq_len)
            **kwargs:

        Returns:
            (batch_size, seq_len)
        """
        targets = None
        return inputs, targets


class SpanContextPrediction(TaskHead):
    """
    Span Context prediction task: given text tokens: [t1, t2, ..... t20] predict if
    text[j:k] is contained in text[i:p]
    """

    def __init__(self, config):
        super().__init__(config)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def split_inputs(self, inputs: torch.Tensor, **kwargs):
        """
        Split the inputs by CLS, SEP, and PAD tokens
        """
        pass

    def prepare_inputs(self, inputs: torch.Tensor, **kwargs):
        """
        Function that prepares inputs token ids for the task head objective

        Args:
            inputs: (batch_size, seq_len)
            **kwargs:

        Returns:
            (batch_size, seq_len)
        """
        # Prepare inputs for albert sentence order prediction
        targets = 0
        print(inputs)
        return inputs, targets
