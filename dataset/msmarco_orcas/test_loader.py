import os
import sys
sys.path.append(os.getcwd())
from loader import *


def test_constructor():
    print("Test cache...")
    split = OrcasSplit("small")
    split2 = OrcasSplit("small")
    print("Test passed!")

def test_iterate_through_dataset():
    print("Testing iterate_through_dataset...")
    test_data = QueryDocumentOrcasDataset('medium')
    for i, (query, document) in enumerate(test_data):
        print(query)
        print(document)
        if i == 10:
            break
    print("Test passed!")
