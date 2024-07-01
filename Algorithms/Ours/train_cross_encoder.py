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



# First, we define the transformer model we want to fine-tune
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
train_batch_size = 32
num_epochs = 1
model_save_path = (
    "/mnt/sdd/shpark/cross_encoder/training_ott_qa_cross-encoder-"
    + model_name.replace("/", "-")
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)


# We train the network with as a binary label task
# Given [query, passage] is the label 0 = irrelevant or 1 = relevant?
# We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
# in our training setup. For the negative samples, we use the triplets provided by MS Marco that
# specify (query, positive sample, negative sample).
pos_neg_ration = 4

# Maximal number of training samples we want to use
max_train_samples = 1e6

# We set num_labels=1, which predicts a continuous score between 0 and 1
model = CrossEncoder(model_name, num_labels=1, max_length=512)


#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}
collection_filepath = "/mnt/sdc/shpark/training_data/collection.tsv"
with open(collection_filepath, "r", encoding="utf8") as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}
queries_filepath = "/mnt/sdc/shpark/training_data/queries.tsv"
with open(queries_filepath, "r", encoding="utf8") as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query

dev_corpus = {}
dev_collection_filepath = "/mnt/sdc/shpark/dev_data/collection.tsv"
with open(dev_collection_filepath, "r", encoding="utf8") as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        dev_corpus[pid] = passage


### Read the train queries, store in queries dict
dev_queries = {}
dev_queries_filepath = "/mnt/sdc/shpark/dev_data/queries.tsv"
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
train_eval_filepath = "/mnt/sdc/shpark/dev_data/triples.tsv"

with open(train_eval_filepath, "r", encoding="utf8") as fIn:
    for line in fIn:
        qid, pos_id, neg_id = line.strip().split("\t")

        if qid not in dev_samples and len(dev_samples) < num_dev_queries:
            dev_samples[qid] = {"query": dev_queries[qid], "positive": set(), "negative": set()}

        if qid in dev_samples:
            dev_samples[qid]["positive"].add(dev_corpus[pos_id])

            if len(dev_samples[qid]["negative"]) < num_max_dev_negatives:
                dev_samples[qid]["negative"].add(dev_corpus[neg_id])


# Read our training file
train_filepath = "/mnt/sdc/shpark/training_data/triples.tsv"

cnt = 0
with open(train_filepath, "r", encoding="utf8") as fIn:
    for line in tqdm.tqdm(fIn, unit_scale=True):
        qid, pos_id, neg_id = line.strip().split("\t")

        query = queries[qid]
        if (cnt % (pos_neg_ration + 1)) == 0:
            passage = corpus[pos_id]
            label = 1
        else:
            passage = corpus[neg_id]
            label = 0

        train_samples.append(InputExample(texts=[query, passage], label=label))
        cnt += 1

        if cnt >= max_train_samples:
            break

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
evaluator = CERerankingEvaluator(dev_samples, name="train-eval")

# Configure the training
warmup_steps = 5000
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    use_amp=True,
)

# Save latest model
model.save(model_save_path + "-latest")