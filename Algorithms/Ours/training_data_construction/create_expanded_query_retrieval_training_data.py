import hydra
import json
from omegaconf import DictConfig
from pymongo import MongoClient
from tqdm import tqdm
import random
import os
from collections import defaultdict
random.seed(0)
def get_table_info(table_id, tables_dict):
    """Extracts table information such as title, column names, and rows."""
    raw_table = tables_dict.get(table_id)
    if not raw_table:
        return None, None, None

    table_lines = raw_table['text'].strip().split('\n')
    if not table_lines:
        return None, None, None

    title = ' '.join(table_id.split('_')[:-1])
    column_names = table_lines[0]
    rows = table_lines[1:]

    return title, column_names, rows

def get_passage_info(passage_id, passages_dict):
    """Extracts passage title and text."""
    raw_passage = passages_dict.get(passage_id)
    if not raw_passage:
        return None, None

    title = passage_id.replace('/wiki/', '').replace('_', ' ')
    return title, raw_passage

@hydra.main(config_path="conf", config_name="create_training_dataset_expanded_query")
def main(cfg: DictConfig):
    data_type = cfg.data_type
    triples_path = cfg.triples_path
    queries_path = cfg.queries_path
    collection_path = cfg.collection_path
    training_dict_path = cfg.training_dict_path

    with open('/mnt/sdf/OTT-QAMountSpace/Dataset/OTT-QA/preprocessed/all_passages.json', 'r') as f:
        raw_passages = json.load(f)

    raw_tables = {}
    with open('/mnt/sdf/OTT-QAMountSpace/Dataset/OTT-QA/raw_tables.json', 'r') as f:
        raw_table_list = json.load(f)

    for raw_table in raw_table_list:
        raw_tables[raw_table['table_id']] = raw_table

    if data_type == 'passage_to_table':
        if not os.path.exists('/mnt/sdf/OTT-QAMountSpace/Dataset/COS/Hard_negatives/qid_to_hard_negative_table_ids.json'):
            
            with open('/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_train_q_to_tables_with_bm25neg.json') as f:
                cos_single_retrieval_train_data = json.load(f)
            
            qid_to_hard_negative_table_ids = {}
            
            for cos_single_retrieval_train_datum in tqdm(cos_single_retrieval_train_data):
                qid = cos_single_retrieval_train_datum['id']
                
                hard_negative_table_id_set = set()
                for hard_negative_ctx in cos_single_retrieval_train_datum['hard_negative_ctxs']:
                    table_id = '_'.join(hard_negative_ctx['chunk_id'].split('_')[:-1])
                    hard_negative_table_id_set.add(table_id)
                
                qid_to_hard_negative_table_ids[qid] = list(hard_negative_table_id_set)
            
            with open('/mnt/sdf/OTT-QAMountSpace/Dataset/COS/Hard_negatives/qid_to_hard_negative_table_ids.json', 'w') as f:
                json.dump(qid_to_hard_negative_table_ids, f, ensure_ascii=False, indent=4)
        else:
            with open('/mnt/sdf/OTT-QAMountSpace/Dataset/COS/Hard_negatives/qid_to_hard_negative_table_ids.json') as f:
                qid_to_hard_negative_table_ids = json.load(f)
    else:
        if not os.path.exists('/mnt/sdf/OTT-QAMountSpace/Dataset/COS/Hard_negatives/qid_to_hard_negative_passage_ids.json'):
        
            with open('/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_train_expanded_retrieval.json') as f:
                cos_expanded_query_retrieval_train_data = json.load(f)
            
            qid_to_hard_negative_passage_ids = {}
            
            for cos_expanded_query_retrieval_train_datum in tqdm(cos_expanded_query_retrieval_train_data):
                qid = cos_expanded_query_retrieval_train_datum['id']
                
                hard_negative_passage_id_set = set()
                for positive_ctx in cos_expanded_query_retrieval_train_datum['positive_ctxs']:
                    
                    for hard_negative_ctx in positive_ctx['hard_negative_ctxs']:
                        passage_id = '/wiki/' + hard_negative_ctx['title'].replace(' ', '_')
                        hard_negative_passage_id_set.add(passage_id)
                
                qid_to_hard_negative_passage_ids[qid] = list(hard_negative_passage_id_set)
            
            with open('/mnt/sdf/OTT-QAMountSpace/Dataset/COS/Hard_negatives/qid_to_hard_negative_passage_ids.json', 'w') as f:
                json.dump(qid_to_hard_negative_passage_ids, f, ensure_ascii=False, indent=4)
        else:
            with open('/mnt/sdf/OTT-QAMountSpace/Dataset/COS/Hard_negatives/qid_to_hard_negative_passage_ids.json') as f:
                qid_to_hard_negative_passage_ids = json.load(f)

    with open('/home/shpark/OTT-QA/released_data/train.traced.json') as f:
        ott_qa_train_data = json.load(f)

    
    training_dict = defaultdict(lambda: {'positive': [], 'negative': []})
    triples_lines, queries_lines, collection_lines = [], [], []
    did, ex_qid = 0, 0

    # Processing data
    for datum in tqdm(ott_qa_train_data):
        question = datum['question']
        qid = datum['question_id']
        answer_node = datum['answer-node']
        table_id = datum['table_id']

        table_title, column_names, table_rows = get_table_info(table_id, raw_tables)
        if not table_rows:
            continue

        positive_list, positive_context_list = [], []

        for answer in answer_node:
            if answer[3] != 'passage':
                continue
            
            row_id, passage_id = answer[1][0], answer[2]
            row = table_rows[row_id] if 0 <= row_id < len(table_rows) else None

            if not row:
                continue

            table_segment = f"{table_title} [SEP] {column_names} [SEP] {row}"

            passage_title, passage_text = get_passage_info(passage_id, raw_passages)
            if not passage_text:
                continue
            
            if data_type == 'passage_to_table':
                positive_list.append(table_segment)
                positive_context_list.append(f"{passage_title} [SEP] {passage_text}")
            else:
                positive_list.append(f"{passage_title} [SEP] {passage_text}")
                positive_context_list.append(table_segment)

        negative_list = []
        if data_type == 'passage_to_table':
            hard_negative_table_ids = qid_to_hard_negative_table_ids.get(qid, [])
            for hn_table_id in hard_negative_table_ids:
                hn_title, hn_columns, hn_rows = get_table_info(hn_table_id, raw_tables)
                if not hn_rows:
                    continue

                row = random.choice(hn_rows)
                negative_list.append(f"{hn_title} [SEP] {hn_columns} [SEP] {row}")
        else:
            hard_negative_passage_ids = qid_to_hard_negative_passage_ids.get(qid, random.sample(list(raw_passages.keys()), 1))
            for hn_passage_id in hard_negative_passage_ids:
                hn_title, hn_text = get_passage_info(hn_passage_id, raw_passages)
                if not hn_text:
                    continue

                negative_list.append(f"{hn_title} [SEP] {hn_text}")

        start_query_id = ex_qid
        for element, context in zip(positive_list, positive_context_list):
            collection_lines.append(f"{did}\t{element}\n")
            expanded_query = f"{question} [SEP] {context}"
            queries_lines.append(f"{ex_qid}\t{expanded_query}\n")

            training_dict[str(ex_qid)]['positive'].append(did)
            ex_qid += 1
            did += 1

        for element in negative_list:
            collection_lines.append(f"{did}\t{element}\n")
            for q_id in range(start_query_id, ex_qid):
                training_dict[str(q_id)]['negative'].append(did)

            did += 1

        for q_id in range(start_query_id, ex_qid):
            for positive_id in training_dict[str(q_id)]['positive']:
                for negative_id in training_dict[str(q_id)]['negative']:
                    triples_lines.append(f"{q_id}\t{positive_id}\t{negative_id}\n")

    # Write output in bulk
    with open(triples_path, 'w', encoding='utf-8') as f:
        f.writelines(triples_lines)
    with open(queries_path, 'w', encoding='utf-8') as f:
        f.writelines(queries_lines)
    with open(collection_path, 'w', encoding='utf-8') as f:
        f.writelines(collection_lines)
    with open(training_dict_path, 'w', encoding='utf-8') as f:
        json.dump(training_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()