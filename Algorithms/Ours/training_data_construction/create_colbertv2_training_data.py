import json
import torch
from tqdm import tqdm
from collections import defaultdict
from FlagEmbedding import LayerWiseFlagLLMReranker

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            data.append(json.loads(line.strip()))
    return data

batch_size = 512

triples = read_jsonl('/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Training_Dataset/edge/triples.jsonl')

corpus = {}
collection_filepath = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Training_Dataset/edge/collection.tsv"
with open(collection_filepath, "r", encoding="utf8") as fIn:
    for line in tqdm(fIn):
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}
queries_filepath = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Training_Dataset/edge/queries.tsv"
with open(queries_filepath, "r", encoding="utf8") as fIn:
    for line in tqdm(fIn):
        qid, query = line.strip().split("\t")
        queries[qid] = query


pid_set = set()
pid_list = []
qid_dict = defaultdict(lambda: {'positive_pid': None, 'negative_pids': set()})

for triple in tqdm(triples):
    qid = triple[0]
    positive_pid = triple[1]
    negative_pid = triple[2]
    
    # Set positive_pid if not already set
    if qid_dict[qid]['positive_pid'] is None:
        qid_dict[qid]['positive_pid'] = positive_pid
        pos_tuple = (qid, positive_pid, 'pos')
        if pos_tuple not in pid_set:
            pid_set.add(pos_tuple)
            pid_list.append(list(pos_tuple))
    
    # Add negative_pid only if there are less than 64
    if len(qid_dict[qid]['negative_pids']) < 64:
        qid_dict[qid]['negative_pids'].add(negative_pid)
        neg_tuple = (qid, negative_pid, 'neg')
        if neg_tuple not in pid_set:
            pid_set.add(neg_tuple)
            pid_list.append(list(neg_tuple))

cross_encoder_edge_retriever = LayerWiseFlagLLMReranker("/mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/Merged_BAAI_RERANKER_15_96_ckpt_400", use_fp16=True)

score_info = {}

for i in tqdm(range(0, len(pid_list), batch_size)):
    pid_batch = pid_list[i:i+batch_size]
    model_input = [[queries[str(qid)], corpus[str(pid)]] for qid, pid, _ in pid_batch]
    with torch.no_grad():
        edge_scores = cross_encoder_edge_retriever.compute_score(model_input, batch_size=batch_size, cutoff_layers=[40], max_length=256)
    
        for (qid, pid, label), score in zip(pid_batch, edge_scores):
            if qid not in score_info:
                score_info[qid] = {'pos': [], 'neg': []}

            if (pid, float(score)) not in score_info[qid][label]:
                score_info[qid][label].append((pid, float(score)))

training_dataset = []
for qid, passage_info in score_info.items():
    pos_passages = passage_info['pos'][:1]
    neg_passages = passage_info['neg'][:64]
    for pos_pid, pos_score in pos_passages:
        training_datum = [qid]
        training_datum.append([pos_pid, pos_score])
        for neg_pid, neg_score in neg_passages:
            training_datum.append([neg_pid, neg_score])
        training_dataset.append(training_datum)

with open('/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Training_Dataset/edge/training_dataset_w_score_ckpt_400.jsonl', 'w', encoding='utf-8') as file:
    for training_datum in training_dataset:
        file.write(json.dumps(training_datum) + '\n')