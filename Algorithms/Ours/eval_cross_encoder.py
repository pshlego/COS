"""
This examples show how to train a Cross-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The query and the passage are passed simoultanously to a Transformer network. The network then returns
a score between 0 and 1 how relevant the passage is for a given query.

The resulting Cross-Encoder can then be used for passage re-ranking: You retrieve for example 100 passages
for a given query, for example with Elasticsearch, and pass the query+retrieved_passage to the CrossEncoder
for scoring. You sort the results then according to the output of the CrossEncoder.

This gives a significant boost compared to out-of-the-box Elasticsearch / BM25 ranking.

Running this script:
python train_cross-encoder.py
"""

import gzip
import logging
import os
import tarfile
from datetime import datetime

import tqdm
from torch.utils.data import DataLoader

from sentence_transformers import InputExample, LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from FlagEmbedding import LayerWiseFlagLLMReranker, FlagReranker


# First, we define the transformer model we want to fine-tune
# We set num_labels=1, which predicts a continuous score between 0 and 1
# model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# model = CrossEncoder(model_name, num_labels=1, max_length=512)
model_name = "/mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/Merged_BAAI_RERANKER_15_96_ckpt_400"
model = LayerWiseFlagLLMReranker(model_name, use_fp16=True)
# model_name = "/mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/slm_reranker_baai/checkpoint-6000"
# model = FlagReranker(model_name, use_fp16=True)
dev_corpus = {}
dev_collection_filepath = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Development_Dataset/edge/collection.tsv"
with open(dev_collection_filepath, "r", encoding="utf8") as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        dev_corpus[pid] = passage


### Read the train queries, store in queries dict
dev_queries = {}
dev_queries_filepath = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Development_Dataset/edge/queries.tsv"
with open(dev_queries_filepath, "r", encoding="utf8") as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        dev_queries[qid] = query

### Now we create our training & dev data
train_samples = []
dev_samples = {}

# We use 200 random queries from the train set for evaluation during training
# Each query has at least one relevant and up to 200 irrelevant (negative) passages
num_dev_queries = 2214
num_max_dev_negatives = 200

# msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz and msmarco-qidpidtriples.rnd-shuf.train.tsv.gz is a randomly
# shuffled version of qidpidtriples.train.full.2.tsv.gz from the MS Marco website
# We extracted in the train-eval split 500 random queries that can be used for evaluation during training
train_eval_filepath = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Development_Dataset/edge/triples.tsv"

with open(train_eval_filepath, "r", encoding="utf8") as fIn:
    for line in fIn:
        qid, pos_id, neg_id = line.strip().split("\t")

        if qid not in dev_samples and len(dev_samples) < num_dev_queries:
            dev_samples[qid] = {"query": dev_queries[qid], "positive": set(), "negative": set()}

        if qid in dev_samples:
            dev_samples[qid]["positive"].add(dev_corpus[pos_id])

            if len(dev_samples[qid]["negative"]) < num_max_dev_negatives:
                dev_samples[qid]["negative"].add(dev_corpus[neg_id])

evaluator = CERerankingEvaluator(dev_samples, name="train-eval")

mean_mrr = evaluator(model)
print("Mean MRR:", mean_mrr)