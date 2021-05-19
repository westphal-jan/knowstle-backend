"""
Training with hard negatives from cross encoder
"""

from torch.utils.data import DataLoader
from sentence_transformers import losses, util, models
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import os
from shutil import copyfile
import sys
from MultipleNegativesRankingWithMarginLoss import MultipleNegativesRankingWithMarginLoss
#from MultipleNegativesMarginLoss import MultipleNegativesMarginLoss
from NTXentLossTriplet import NTXentLossTriplet
import math
from FirstPooling import FirstPooling
import gzip
import tqdm
from QueriesDataset import QueriesDataset
import random
import json
import torch

if __name__ == "__main__":
    num_epoch = 10
    train_batch_size = 140 #75
    base_model = 'cnt_training_microsoft_MiniLM-L6_v3' #'cnt_training_microsoft_MiniLM-L12-H384-L6' #'google/electra-small-discriminator'
    use_identifier = False

    model_save_path = '../output/{}-mined_hard_neg-mean-pooling-{}-epoch{}-batchsize{}-{}'.format(base_model.replace('/', '_'), "" if use_identifier else "no_identifier", num_epoch, train_batch_size, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Write self to path
    os.makedirs(model_save_path, exist_ok=True)

    train_script_path = os.path.join(model_save_path, 'train_script.py')
    copyfile(__file__, train_script_path)
    with open(train_script_path, 'a') as fOut:
        fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))
        
    # Fill GPU
    #fill_gpu = torch.eye(85000, dtype=torch.float, device='cuda')
    #del fill_gpu

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    

    qry_idt = ""
    doc_idt = ""

    
    logging.info("Use identifier: {}".format(use_identifier))

    if use_identifier:
        qry_idt = "[QRY] "
        doc_idt = "[DOC] "

    
    if base_model.startswith('cnt_training'):
        model = SentenceTransformer('../output/cnt_training_microsoft_MiniLM-L12-H384-L6-mined_hard_neg-mean-pooling-no_identifier-epoch10-batchsize100-2021-04-09_22-25-20')
        model.max_seq_length = 300
        
        ############# Remove layers
        if False:
            auto_model = model._first_module().auto_model
            layers_to_keep = [0, 2, 4, 6, 8, 10]
            print("Reduce model to {} layers".format(len(layers_to_keep)))
            new_layers = torch.nn.ModuleList([layer_module for i, layer_module in enumerate(auto_model.encoder.layer) if i in layers_to_keep])
            auto_model.encoder.layer = new_layers
            auto_model.config.num_hidden_layers = len(layers_to_keep)
            
       ###################
    else:
        word_embedding_model = models.Transformer(base_model, max_seq_length=350)

        if use_identifier:
            tokens = [qry_idt.strip(), doc_idt.strip()]
            word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
            word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 
                                        pooling_mode_mean_tokens=True, 
                                        pooling_mode_cls_token=False, 
                                        pooling_mode_max_tokens=False)
                                        
        #pooling_model = FirstPooling(word_embedding_model.get_word_embedding_dimension())
        #norm = models.Normalize()
        #dense = models.Dense(pooling_model.get_sentence_embedding_dimension(), 768, bias=False, activation_function=torch.nn.Identity())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    
    model.to('cuda')




    corpus = {}
    train_queries = {}

    #### Read train file
    with gzip.open('../data/collection.tsv.gz', 'rt') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            passage = doc_idt + passage
            corpus[pid] = passage

    with open('../data/queries.train.tsv', 'r') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            query = qry_idt+query
            train_queries[qid] = {'query': query,
                                  'pos': set(),
                                  'soft-pos': set(),
                                  'neg': set()}



    #Read qrels file for relevant positives per query
    with open('../data/qrels.train.tsv') as fIn:
        for line in fIn:
            qid, _, pid, _ = line.strip().split()
            train_queries[qid]['pos'].add(pid)




    logging.info("Clean train queries")
    deleted_queries = 0
    for qid in list(train_queries.keys()):
        if len(train_queries[qid]['pos']) == 0:
            deleted_queries += 1
            del train_queries[qid]
            continue

    logging.info("Deleted queries pos-empty: {}".format(deleted_queries))
    hard_neg_files = [
    '../data/hard-negatives-ann-cnt_training_microsoft_MiniLM-L12-H384-L6-mined_hard_neg-mean-pooling-no_identifier-epoch10-batchsize100-2021-04-09_22-25-20.jsonl.gz',
    #'../data/hard-negatives-ann-microsoft_MiniLM-L12-H384-uncased-mined_hard_neg-mean-pooling-no_identifier-epoch10-batchsize70-2021-04-05_07-37-55.jsonl.gz',
    '../data/hard-negatives-ann-distilroberta-base-mined_hard_neg-mean-pooling-dot_prod-no_identifier-epoch10-batchsize75-2021-03-25_12-47-04.jsonl.gz',        #'../data/hard-negatives-ann-distilbert-base-uncased-mined_hard_neg-mean-pooling-dot_prod-no_identifier-epoch10-batchsize75-2021-03-21_13-53-07.jsonl.gz',
        '../data/hard-negatives-ann-msmarco-distilbert-base-v2.jsonl.gz',
        #'../data/hard-negatives-ann-roberta.jsonl.gz',
        #'../data/hard-negatives-ann.jsonl.gz',
        #'../data/hard-negatives-ann-no_idnt.jsonl.gz', 
        #'../data/hard-negatives-bm25.jsonl.gz'
    ]
    for hard_neg_file in hard_neg_files: 
        logging.info("Read hard negatives: "+hard_neg_file)
        with gzip.open(hard_neg_file, 'rt') as fIn:
            try:
                for line in fIn:
                    try:
                        data = json.loads(line)
                    except:
                        continue
                    qid = data['qid']

                    if qid in train_queries:
                        neg_added = 0
                        max_neg_added = 20

                        hits = sorted(data['hits'], key=lambda x: x['score'] if 'score' in x else x['bm25-score'], reverse=True)
                        for hit in hits:
                            pid = hit['corpus_id'] if 'corpus_id' in hit else hit['pid']

                            if pid in train_queries[qid]['pos']:    #Skip entries we have as positives
                                continue

                            if hit['bert-score'] < 0.1 and neg_added < max_neg_added:
                                train_queries[qid]['neg'].add(pid)
                                neg_added += 1
                            elif hit['bert-score'] > 0.9:
                                train_queries[qid]['soft-pos'].add(pid)
            except:
                pass

    #Use soft-pos as positives
    #for qid in list(train_queries.keys()):
    #    if len(train_queries[qid]['soft-pos']) == 0:
    #        del train_queries[qid]
    #    else:
    #        train_queries[qid]['pos'] = train_queries[qid]['soft-pos']


    logging.info("Clean train queries with empty neg set")
    deleted_queries = 0
    for qid in list(train_queries.keys()):
        if len(train_queries[qid]['neg']) == 0:
            deleted_queries += 1
            del train_queries[qid]
            continue

    logging.info("Deleted queries neg empty: {}".format(deleted_queries))

 
    #### Read dev file
    logging.info("Create dev dataset")
    dev_corpus_max_size = 100*1000

    dev_queries_file = '../data/queries.dev.small.tsv'
    needed_pids = set()
    needed_qids = set()
    dev_qids = set()

    dev_queries = {}
    dev_corpus = {}
    dev_rel_docs = {}

    with open(dev_queries_file) as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            dev_qids.add(qid)

    with open('../data/qrels.dev.tsv') as fIn:
        for line in fIn:
            qid, _, pid, _ = line.strip().split('\t')

            if qid not in dev_qids:
                continue

            if qid not in dev_rel_docs:
                dev_rel_docs[qid] = set()
            dev_rel_docs[qid].add(pid)

            needed_pids.add(pid)
            needed_qids.add(qid)

    with open(dev_queries_file) as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            if qid in needed_qids:
                dev_queries[qid] = qry_idt+query

    with gzip.open('../data/collection-rnd.tsv.gz', 'rt') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            passage = doc_idt+passage

            if pid in needed_pids or dev_corpus_max_size <= 0 or len(dev_corpus) <= dev_corpus_max_size:
                dev_corpus[pid] = passage



    logging.info("Train size: {}".format(len(train_queries)))
    logging.info("Dev queries: {}".format(len(dev_queries)))
    logging.info("Dev Corpus: {}".format(len(dev_corpus)))

    ir_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, dev_corpus, dev_rel_docs)

    train_dataset = QueriesDataset(train_queries, corpus, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = NTXentLossTriplet(model, scale=20)

    



    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=ir_evaluator,
              epochs=num_epoch,
              warmup_steps=1000,
              output_path=model_save_path,
              evaluation_steps=math.ceil(len(train_dataloader)/3)+1,
              use_amp=True
              )

    latest_path = model_save_path.rstrip('/')+"_latest"
    os.makedirs(latest_path, exist_ok=True)
    model.save(latest_path)


# Script was called via:
#python train_msmarco_triplets_hard-neg.py