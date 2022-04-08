import ir_datasets

dataset = ir_datasets.load('msmarco-document/orcas')
for query, doc in zip(dataset.queries_iter(), dataset.docs_iter()):
    print(query)
    break