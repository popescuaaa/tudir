from concurrent import futures
from copy import deepcopy
from functools import partial
import random
from typing import List, Set, Tuple, Union
import numpy as np

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
        if word in delim:
            # replace delimiter with SEP token
            res.append(acc)
            acc = []
        # otherwise increase current accumulator
        else:
            acc.append(word)
    # if there are still words in the accumulator, append it
    if acc:
        res.append(acc)
    # shuffle the sentences
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

    # add CLS token to first sentence
    res[0] = [cls_id] + res[0]
    for i in range(len(res) - 1):
        res[i] += [sep_id]

    # flatten the sentences
    res = [word for sent in res for word in sent]
    # extend with padding
    res.extend(sentence[idx + 1:])

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
        with futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
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

if __name__ == "__main__":
    parser = Parser(max_length=512, num_workers=16)
    res = parser.parse(1000 * [[2] + [7] * 130 + [5] + [4] * 80])
