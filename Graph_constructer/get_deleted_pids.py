import hydra
import logging
import json
import time
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm
from omegaconf import DictConfig
from dpr.options import setup_logger
from dpr.utils.tokenizers import SimpleTokenizer
from dpr.data.qa_validation import has_answer
from ColBERT.colbert.infra import ColBERTConfig
from ColBERT.colbert import Searcher
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
logger = logging.getLogger()
setup_logger(logger)

def build_query(filename):
    data = json.load(open(filename))
    for sample in data:
        if 'hard_negative_ctxs' in sample:
            del sample['hard_negative_ctxs']
    return data 

def min_max_normalize(scores):
    mMscaler = MinMaxScaler()
    scores = np.array(scores).reshape(len(scores),1).tolist()
    mMscaler.fit(scores)
    mMscaled_data = mMscaler.transform(scores)
    mMscaled_data = pd.DataFrame(mMscaled_data).values.squeeze().tolist()
    return mMscaled_data

def sort_page_ids_by_scores(page_ids, page_scores):
    # Combine page IDs and scores into a list of tuples
    combined_list = list(zip(page_ids, page_scores))

    # Sort the combined list by scores in descending order
    sorted_by_score = sorted(combined_list, key=lambda x: x[1], reverse=True)

    # Extract the sorted page IDs
    sorted_page_ids = [page_id for page_id, _ in sorted_by_score]

    return sorted_page_ids

# with open('/mnt/sdd/shpark/colbert/data/index_to_chunk_id_both.json', 'r') as f:
#     index_to_chunk_id_both = json.load(f)
# with open('/mnt/sdd/shpark/colbert/data/index_to_chunk_id_star.json', 'r') as f:
#     index_to_chunk_id_star = json.load(f)

def filter_fn(pid, values_to_remove):
    return pid[~torch.isin(pid, values_to_remove)].to("cuda")

@hydra.main(config_path="conf", config_name="subgraph_retriever_colbert")
def main(cfg: DictConfig):
    # mongodb setup
    client = MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password))
    mongodb = client[cfg.dbname]
    # edge_graph_collection = mongodb[cfg.edge_graph_collection_name]

    # total_edge_graphs = edge_graph_collection.count_documents({})
    # edge_graphs = [graph for graph in tqdm(edge_graph_collection.find(), total=total_edge_graphs)]
    
    graph_collection = mongodb["preprocess_table_graph_cos_apply_topk_star_5"]
    total_graphs = graph_collection.count_documents({})
    graphs = [graph for graph in tqdm(graph_collection.find(), total=total_graphs)]
    json_path = "/mnt/sdc/shpark/colbert/data/chunk_id_to_index_both_topk.json"
    chunk_id_to_index_both_topk = json.load(open(json_path))
    
    for selection_method in ['topp']:
        if selection_method == 'thr':
            parameters = [60.94, 56.85, 54.61, 52.75, 50.50]
        elif selection_method == 'topp':
            parameters = [0.5, 0.3, 0.1]
        elif selection_method == 'topk':
            parameters = [1, 2, 3, 4, 5]
        for parameter in parameters:
            pid_deleted_list = []
            if selection_method == 'topk':
                for i, graph in enumerate(tqdm(edge_graphs)):
                    if 'topk' in graph:
                        if graph['topk'] >= parameter:
                            pid_deleted_list.append(i)
            elif selection_method == 'topp':
                for i, graph in enumerate(tqdm(graphs)):
                    if 'mentions_in_row_info_dict' in graph:
                        for mention_id, mention_dict in graph['mentions_in_row_info_dict'].items():
                            normalizes_score_list = min_max_normalize(mention_dict['mention_linked_entity_score_list'])
                            acc_score = 0
                            temperature = 1
                            graph_scores = np.array([graph_score*temperature for graph_score in normalizes_score_list])
                            softmax_scores = np.exp(graph_scores) / np.sum(np.exp(graph_scores), axis=0)
                            softmax_scores = softmax_scores.tolist()
                            for row_k, mention_linked_entity_id in enumerate(mention_dict['mention_linked_entity_id_list']):
                                edge_id = graph['chunk_id'] + f"_{mention_id}" + f"_{row_k}"
                                try:
                                    pid = chunk_id_to_index_both_topk[edge_id]
                                except:
                                    acc_score += softmax_scores[row_k]
                                    continue
                                if acc_score > parameter:
                                    pid_deleted_list.append(int(pid))
                                acc_score += softmax_scores[row_k]
                
                # chunk_id_identifier = None
                # for i, graph in enumerate(tqdm(edge_graphs)):
                #     chunk_id = graph['chunk_id']
                #     current_chunk_id_identifier = '_'.join(chunk_id.split('_')[:3])
                #     if chunk_id_identifier is None or chunk_id_identifier != current_chunk_id_identifier:
                #         if chunk_id_identifier is not None:
                #             sorted_topp_list = sorted(topp_list, key=lambda x: x[1], reverse=True)
                #             acc_score = 0
                #             temperature = 0.1
                #             graph_scores = np.array([graph_score*temperature for _, graph_score in sorted_topp_list])
                #             softmax_scores = np.exp(graph_scores) / np.sum(np.exp(graph_scores), axis=0)
                #             softmax_scores = softmax_scores.tolist()
                #             exceed = False
                #             for id, (j, graph_score) in enumerate(sorted_topp_list):
                #                 if exceed:
                #                     pid_deleted_list.append(j)
                #                 if acc_score > parameter:
                #                     exceed = True
                #                 acc_score += softmax_scores[id]
                        
                #         topp_list = []
                #         chunk_id_identifier = current_chunk_id_identifier

                #     if 'linked_entity_score' in graph:
                #         topp_list.append((i, graph['linked_entity_score']))
            elif selection_method == 'thr':
                for i, graph in enumerate(tqdm(graphs)):
                    if 'mentions_in_row_info_dict' in graph:
                        for mention_id, mention_dict in graph['mentions_in_row_info_dict'].items():
                            normalizes_score_list = min_max_normalize(mention_dict['mention_linked_entity_score_list'])
                            for row_k, mention_linked_entity_id in enumerate(mention_dict['mention_linked_entity_id_list']):
                                edge_id = graph['chunk_id'] + f"_{mention_id}" + f"_{row_k}"
                                try:
                                    pid = chunk_id_to_index_both_topk[edge_id]
                                except:
                                    continue
                                if normalizes_score_list[row_k] < parameter:
                                    pid_deleted_list.append(int(pid))
                # for i, graph in enumerate(tqdm(edge_graphs)):
                #     if 'linked_entity_score' in graph:
                #         if graph['linked_entity_score'] < parameter:
                #             pid_deleted_list.append(i)
            parameter_name = str(parameter).replace('.', '_')
            path = f"/mnt/sdd/shpark/colbert/data/{selection_method}_{parameter_name}.json"
            with open(path, 'w') as f:
                json.dump(pid_deleted_list, f, indent=4)

if __name__ == "__main__":
    main()