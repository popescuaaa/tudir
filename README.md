## SELF-SUPERVISED DOCUMENT RETRIEVER WITH CONTRASTIVE LEARNING

---

### How to run the project:

1. Environment setup
> pip install -r requirements.txt

2. Project structure
```
.
├── BERT_tok-trained.json
├── dataset
│   ├── __init__.py
│   ├── msmarco_document
│   ├── msmarco_orcas
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── test_loader.py
│   └── squad
│       ├── __init__.py
│       ├── iterators.py
│       └── utils.py
├── LICENSE
├── models # a few pretrained models
├── networks
│   ├── __init__.py
│   ├── task_heads
│   │   ├── __init__.py
│   │   ├── mlm_head.py
│   │   ├── sop_head.py
│   │   ├── sop_play.ipynb
│   │   ├── task_head.py
│   │   ├── tests
│   │   │   ├── test_mlm_head.py
│   │   │   └── test_sop_head.py
│   │   └── vanilla_transformer_mod.py
│   └── transformers
│       ├── encoding.py
│       ├── __init__.py
│       ├── query_document_transformer.py
│       ├── tests
│       │   └── test_vanilla_transformer.py
│       └── vanilla_transformer.py
├── parser.py
├── pmi_cummulative_distribution.png
├── pmi_distribution.png
├── README.md
├── requirements.txt
├── setup.py
├── squad-data
│   └── SQuAD2
│       └── train-v2.0.json
├── tokenization
│   ├── basic_setup.ipynb
│   ├── corpus_tokenizers.py
│   ├── data
│   │   └── queries
│   │       ├── query1.txt
│   │       ├── query2.txt
│   │       ├── query3.txt
│   │       └── query4.txt
│   ├── tokenizers.ipynb
│   ├── token_statistics.ipynb
│   ├── utils
│   │   ├── __init__.py
│   │   ├── plot.py
│   │   ├── pmi.py
│   │   └── tfidf.py
│   └── vocab_tokenizers.py
├── trainer_and_evaluator.py # the main entry point
```

3. Running training and evaluation
> python traininer_and_evaluator.py

All the models will be saved the the models folder. There are no additional steps to be performed in order to run the system.

### Data acquisition
The data should be downloaded independently from any source: the orcas dataset for example could be found on the MS-Marco competition page.

