import hydra
import logging
import glob
import json
import pickle
import faiss
import torch
from pymongo import MongoClient
import numpy as np
from tqdm import tqdm
from typing import List
from torch import Tensor as T
from omegaconf import DictConfig, OmegaConf
from dpr.models import init_biencoder_components
from dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
from dpr.utils.tokenizers import SimpleTokenizer
from dpr.data.qa_validation import has_answer
import time
from DPR.run_chain_of_skills_hotpot import generate_question_vectors
import csv

logger = logging.getLogger()
setup_logger(logger)
def read_tsv_to_list_of_dicts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        graph_dict = {}
        for row in reader:
            if int(tuple(row)[0]) not in graph_dict:
                graph_dict[int(tuple(row)[0])] = {'retrieves':[], 'scores':[]}
            graph_dict[int(tuple(row)[0])]['retrieves'].append(int(tuple(row)[1]))
            graph_dict[int(tuple(row)[0])]['scores'].append(float(tuple(row)[3]))
        return graph_dict

def build_query(filename):
    data = json.load(open(filename))
    for sample in data:
        if 'hard_negative_ctxs' in sample:
            del sample['hard_negative_ctxs']
    return data 

def sort_page_ids_by_scores(page_ids, page_scores):
    # Combine page IDs and scores into a list of tuples
    combined_list = list(zip(page_ids, page_scores))

    # Sort the combined list by scores in descending order
    sorted_by_score = sorted(combined_list, key=lambda x: x[1], reverse=True)

    # Extract the sorted page IDs
    sorted_page_ids = [page_id for page_id, _ in sorted_by_score]

    return sorted_page_ids

@hydra.main(config_path="conf", config_name="subgraph_retriever_colbert")
def main(cfg: DictConfig):
    # mongodb setup
    client = MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password))
    mongodb = client[cfg.dbname]
    
    # load dataset
    table_collection = mongodb[cfg.table_collection_name]
    total_tables = table_collection.count_documents({})
    print(f"Loading {total_tables} tables...")
    all_tables = [doc for doc in tqdm(table_collection.find(), total=total_tables)]
    print("finish loading tables")    

    passage_collection = mongodb[cfg.passage_collection_name]
    total_passages = passage_collection.count_documents({})
    print(f"Loading {total_passages} passages...")
    all_passages = {doc['chunk_id']:doc for doc in tqdm(passage_collection.find(), total=total_passages)}
    print("finish loading passages")
    graphs = {}
    
    for graph_collection_name in cfg.graph_collection_name_list:
        graph_collection = mongodb[graph_collection_name]
        total_graphs = graph_collection.count_documents({})
        print(f"Loading {total_graphs} graphs...")
        
        for doc in tqdm(graph_collection.find(), total=total_graphs):
            doc['table_name'] = all_tables[doc['table_id']]['chunk_id']
            graphs[doc['chunk_id']] = doc

    print("finish loading graphs")
    doc_ids = json.load(open(cfg.doc_ids, 'r'))
    limits = [1, 5, 10, 20, 50, 100]

    answer_recall = [0]*len(limits)
    
    data = build_query(cfg.qa_dataset_path)
    
    k = 100                         
    b_size = 1
    tokenizer = SimpleTokenizer()
    new_data = []
    time_list = []
    search_results = read_tsv_to_list_of_dicts(cfg.colbert_query_results)
    for i in tqdm(range(0, len(data), b_size)):
        time1 = time.time()
        
        D, I = [search_results[i]['scores']], [search_results[i]['retrieves']]
        
        original_sample = data[i]
        all_included = []
        full_text_set = set()
        for j, ind in enumerate(I):
            for m, id in enumerate(ind):
                subgraph = graphs[doc_ids[str(id)]]
                table = all_tables[subgraph['table_id']]
                table_title = table['title']
                table_name = subgraph['table_name']
                full_text = table['text']
                rows = table['text'].split('\n')[1:]
                header = table['text'].split('\n')[0]
                doc_id = subgraph['chunk_id']
                row_id = int(doc_id.split('_')[1])
                if 'answers' in original_sample:
                    pasg_has_answer = has_answer(original_sample['answers'], full_text, tokenizer, 'string')
                else:
                    pasg_has_answer = False
                if len(all_included) == k:
                    break
                all_included.append({'id':subgraph['table_id'], 'title': table_title, 'text': full_text, 'has_answer': pasg_has_answer, 'score': float(D[j][m]), 'table_name': table_name})
                # star case
                if 'passage_score_list' in subgraph:
                    passage_id_list = sort_page_ids_by_scores(subgraph['passage_chunk_id_list'], subgraph['passage_score_list'])
                else:
                    passage_id_list = subgraph['passage_chunk_id_list']
                for passage_id in passage_id_list:
                    passage = all_passages[passage_id]
                    full_text = header + '\n' + rows[row_id] + '\n' + passage['title'] + ' ' + passage['text']
                    if full_text in full_text_set:
                        continue
                    full_text_set.add(full_text)
                    if 'answers' in original_sample:
                        pasg_has_answer = has_answer(original_sample['answers'], full_text, tokenizer, 'string')
                    else:
                        pasg_has_answer = False
                    if len(all_included) == k:
                        break
                    all_included.append({'id':subgraph['table_id'], 'title': table_title, 'text': full_text, 'has_answer': pasg_has_answer, 'score': float(D[j][m]), 'table_name': table_name})
        original_sample['ctxs'] = all_included
        time2 = time.time()
        time_list.append(time2-time1)
        for l, limit in enumerate(limits):
            
            if any([ctx['has_answer'] for ctx in all_included[:limit]]):
                answer_recall[l] += 1

        new_data.append(original_sample)
    for l, limit in enumerate(limits):
        print ('answer recall', limit, answer_recall[l]/len(data))
    print ('average time', np.mean(time_list))
    if cfg.search_space is None:
        result_path = cfg.result_path.split('.')[0] + '_' + cfg.hierarchical_level + '.json'
    else:
        result_path = cfg.result_path.split('.')[0] + '_' + cfg.hierarchical_level + '_' + cfg.search_space + '.json'
    
    with open(result_path, 'w') as f:
        json.dump(new_data, f, indent=4)
    
    if cfg.search_space is None:
        time_path = cfg.result_path.split('.')[0] + '_' + cfg.hierarchical_level + '_time.json'
    else:
        time_path = cfg.result_path.split('.')[0] + '_' + cfg.hierarchical_level + '_' + cfg.search_space + '_time.json'
    
    with open(time_path, 'w') as f:
        json.dump(time_list, f, indent=4)

if __name__ == "__main__":
    main()