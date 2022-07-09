from dataset.msmarco_orcas.loader import QueryDocumentOrcasDataset

if __name__ == '__main__':
    # split = OrcasSplit(split="all")
    # path = split.build_corpus()
    # text_paths = os.listdir(path)
    #
    # text_paths = [os.path.join(path, text_path) for text_path in text_paths]
    # tokenizer = train_BertWordPieceTokenizer(text_paths)
    ds = QueryDocumentOrcasDataset(split="tiny")
    print(ds[0])
    print(ds[1])

    """
    Pretraining:

    for q, d in ds:
        # Tokenizam pisatu (seq len fix)
        # Qs
        q_input, q_target = mlm_head_prepare(q)

        # Docs
        d_input, d_target = sop_head_prepare(d)

        task_indexes = {
            "mlm_head": (0, b // 2),
            "sop_head": (b // 2, b),
        }
    
        embs_ids = torch.cat([q_input, d_input], dim=0)
        embs = tranformer.embs(embs_ids) # Emb table
        for k, v in task_indexes.items():
            embs[v[0]:v[1]] = transformer.emb_proj[k](embs[v[0]:v[1]])

        embs = transofmer.layers(embs)
        losses = {
            "mlm_head": mlm_head_loss(embs[:b // 2], q_target),
            "sop_head": sop_head_loss(embs[b // 2:], d_target),
        } 

        loss = sum(losses.values()) // 2
        loss.backward()
        optimizer.step()


    2. Contrastive:
       for q, d in ds:
        # Tokenizam pisatu (seq len fix)
        # Qs
        q_input, q_target = mlm_head_prepare(q)

        # Docs
        d_input, d_target = sop_head_prepare(d)

        task_indexes = {
            "mlm_head": (0, b // 2),
            "sop_head": (b // 2, b),
        }
    
        embs_ids = torch.cat([q_input, d_input], dim=0)
        embs = tranformer.embs(embs_ids) # Emb table
        for k, v in task_indexes.items():
            embs[v[0]:v[1]] = transformer.emb_proj[k](embs[v[0]:v[1]])

        embs = transofmer.layers(embs)

        qs = embs[:b // 2]
        ds = embs[b // 2:]

        qs = query_reduction(qs)
        ds = doc_reduction(ds)

        # Dot prod
        qs_ds = torch.bmm(qs, ds.transpose(1, 2)) # B x B
        targets = torch.arange(b)

        contrastive_loss = F.BCE(qs_ds, targets)
        contrastive_loss.backward()
        optimizer.step()
    """

