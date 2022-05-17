from dataclasses import replace
from typing import List, Tuple
import numpy as np
import random

def replace_with_mask(x):
    return 2

def replace_with_random(x):
    return random.randint(0, 32_000)


def replace_with_identity(x):
    return x

functions = [replace_with_mask, replace_with_random, replace_with_identity]

function_weights = [0.8, 0.1, 0.1]

def generate_random_sequence():
    seq_len = random.randint(3, 50)
    
    indices = [random.randint(0, 32_000) for i in range(seq_len)]

    return indices, seq_len

def pad_sequences(input: List[Tuple[List[int], int]]):
    sequences = [i[0] for i in input]
    lengths = [i[1] for i in input]
    padded_sequences = []
    padding_starts = []

    max_len = max(lengths)

    for sequence, length in zip(sequences, lengths):
        padded_sequence = sequence + [0] * (max_len - length)
        pad_start = length

        padded_sequences.append(padded_sequence)
        padding_starts.append(pad_start)

    return padded_sequences, padding_starts




if __name__ == "__main__":

    indices = [random.randint(0, 32_000) for i in range(20)]

    # replace_indices = random.choices(list(range(len(indices))), k=int(0.15 * len(indices)))

    replace_indices = random.choices(list(range(len(indices))), k=min(1, int(0.15 * len(indices))))

    replace_functions = random.choices(functions, weights=function_weights, k=len(replace_indices))

    replaced_values = [f(indices[i]) for i, f in zip(replace_indices, replace_functions)]

    print(indices)
    print(replace_indices)

    print(replaced_values)

    print(np.array(indices)[replace_indices])

    batch_indices = [generate_random_sequence() for _ in range(5)]
    padded_indices, padding_starts = pad_sequences(batch_indices)

    for indices, pad_start in zip(padded_indices, padding_starts):

        replace_indices = random.choices(list(range(pad_start)), k=min(1, int(0.15 * pad_start)))

        replace_functions = random.choices(functions, weights=function_weights, k=len(replace_indices))

        replaced_values = [f(indices[i]) for i, f in zip(replace_indices, replace_functions)]
