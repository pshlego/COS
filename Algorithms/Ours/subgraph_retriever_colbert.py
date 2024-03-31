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

logger = logging.getLogger()
setup_logger(logger)

def min_max_normalize(scores):
    min_val = np.min(scores)
    max_val = np.max(scores)
    if min_val == max_val:
        return np.ones(len(scores))
    normalized = (scores - min_val) / (max_val - min_val)
    return normalized.squeeze()

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
    
    # load dataset
    all_tables = json.load(open(cfg.table_data_file_path))
    all_passages = {doc['chunk_id']:doc for doc in json.load(open(cfg.passage_data_file_path))}
    
    star_graphs = {}
    edge_graphs = {}
    # edge_graph_collection = mongodb[cfg.edge_graph_collection_name]
    # star_graph_collection = mongodb[cfg.star_graph_collection_name]
    # TODO: convert dictionary to list or vector
    for graph_collection_name in cfg.graph_collection_name_list:
        graph_collection = mongodb[graph_collection_name]
        total_graphs = graph_collection.count_documents({})
        print(f"Loading {total_graphs} graphs...")
        
        for doc in tqdm(graph_collection.find(), total=total_graphs):
            doc['table_name'] = all_tables[doc['table_id']]['chunk_id']
            if 'edge' in graph_collection_name:
                edge_graphs[doc['chunk_id']] = doc
            else:
                star_graphs[doc['chunk_id']] = doc

    print("finish loading graphs")
    doc_ids = json.load(open(cfg.doc_ids, 'r'))
    limits = [1, 5, 10, 20, 50, 100]

    
    data = build_query(cfg.qa_dataset_path)
    
    k = 100                         
    b_size = 1
    tokenizer = SimpleTokenizer()
    
    config = ColBERTConfig(
            root=cfg.collection_root_dir_path,
        )
    
    experiment_name = f"ottqa_{cfg.hierarchical_level}_512_topk" #_512" #_512
        
    colbert_searcher = Searcher(index=f"{experiment_name}.nbits{cfg.nbits}", config=config, index_root=cfg.index_root)
    edge_index_max = 80182286
    pid_deleted_list = []
    # total_edge_graphs = edge_graph_collection.count_documents({})
    for selection_method in ['thr']:
        if selection_method == 'thr':
            parameters = [0.975, 0.925, 0.875, 0.85, 0.825, 0.8] # [0.95, 0.9, 0.7, 0.5, 0.3, 0.1]
            # parameters = [60.94, 56.85, 54.61, 52.75]
            # parameters = [60.94, 56.85, 54.61, ]
        elif selection_method == 'topp':
            parameters = [0.9, 0.7, 0.5, 0.3, 0.1]
            # parameters = [0.9]
        elif selection_method == 'topk':
            parameters = [2, 3, 4, 5]
            # parameters = [1]
        for parameter in parameters:
            new_data = []
            time_list = []
            answer_recall = [0]*len(limits)
            parameter_name = str(parameter).replace('.', '_')
            if selection_method == 'topk':
                pid_deleted_list = json.load(open(f"/mnt/sdd/shpark/colbert/data/{selection_method}_{parameter_name}.json", 'r'))
            else:
                pid_deleted_list = json.load(open(f"/mnt/sdd/shpark/colbert/data/normalized_{selection_method}_{parameter_name}.json", 'r'))
            
            values_to_remove = torch.tensor(pid_deleted_list, dtype=torch.int32).to("cuda")
            
            for i in tqdm(range(0, len(data), b_size)):
                time1 = time.time()
                # torch.cuda.empty_cache()
                if cfg.scale is None:
                    ranking = colbert_searcher.search(text=data[i]['question'], k=100)
                else:
                    ranking = colbert_searcher.search(text=data[i]['question'], k=700, filter_fn=filter_fn, pid_deleted_list=values_to_remove)
                
                D, I = [ranking[2]], [ranking[0]]
                
                original_sample = data[i]
                all_included = []
                exist_table = {}
                exist_passage = {}
                
                for j, ind in enumerate(I):
                    for m, id in enumerate(ind):
                        # if id < edge_index_max:
                        #     subgraph = edge_graph_collection.find_one({'chunk_id': doc_ids[str(id)]})
                        # else:
                        #     subgraph = star_graph_collection.find_one({'chunk_id': doc_ids[str(id)]})
                        if id < edge_index_max:
                            subgraph = edge_graphs[doc_ids[str(id)]]
                        else:
                            subgraph = star_graphs[doc_ids[str(id)]]
                        
                        table_id = subgraph['table_id']
                        table = all_tables[table_id]
                        table_title = table['title']
                        table_name = subgraph['table_name']
                        full_text = table['text']
                        rows = table['text'].split('\n')[1:]
                        header = table['text'].split('\n')[0]
                        doc_id = subgraph['chunk_id']
                        row_id = int(doc_id.split('_')[1])

                        # star case
                        if 'mentions_in_row_info_dict' in subgraph:
                            passage_chunk_id_list = []
                            passage_score_list = []
                            if selection_method == 'topk':
                                for rid, r_dict in subgraph['mentions_in_row_info_dict'].items():
                                    for row_k, mention_linked_entity_id in enumerate(r_dict['mention_linked_entity_id_list']):
                                        #if row_k < 1:
                                        if row_k < parameter:
                                            passage_chunk_id_list.append(mention_linked_entity_id)
                                            passage_score_list.append(r_dict['mention_linked_entity_score_list'][row_k])
                            elif selection_method == 'topp':
                                for rid, r_dict in subgraph['mentions_in_row_info_dict'].items():
                                    normalizes_score_list = min_max_normalize(r_dict['mention_linked_entity_score_list'])
                                    acc_score = 0
                                    temperature = 1  # Consider removing if temperature is always 1, as it doesn't change the output
                                    exp_scores = np.exp(normalizes_score_list * temperature)
                                    softmax_scores = exp_scores / exp_scores.sum()  # More efficient softmax calculation
                                    for row_k, mention_linked_entity_id in enumerate(r_dict['mention_linked_entity_id_list']):
                                        if acc_score <= parameter:
                                            passage_chunk_id_list.append(mention_linked_entity_id)
                                            passage_score_list.append(r_dict['mention_linked_entity_score_list'][row_k])
                                        else:
                                            passage_chunk_id_list.append(mention_linked_entity_id)
                                            passage_score_list.append(r_dict['mention_linked_entity_score_list'][row_k])
                                            break
                                        acc_score += softmax_scores[row_k]
                            elif selection_method == 'thr':
                                for rid, r_dict in subgraph['mentions_in_row_info_dict'].items():
                                    normalizes_score_list = min_max_normalize(r_dict['mention_linked_entity_score_list'])
                                    for row_k, mention_linked_entity_id in enumerate(r_dict['mention_linked_entity_id_list']):
                                        if normalizes_score_list[row_k] >= parameter:
                                            passage_chunk_id_list.append(mention_linked_entity_id)
                                            passage_score_list.append(r_dict['mention_linked_entity_score_list'][row_k])
                                    
                            passage_id_list = sort_page_ids_by_scores(passage_chunk_id_list, passage_score_list)
                            level = 'star'
                            #TODO: how to filter unrelated passages
                        else:
                            if 'linked_entity_id' in subgraph:
                                passage_id_list = [subgraph['linked_entity_id']]
                                level = 'edge'
                            else:
                                passage_id_list = []
                                level = 'edge'
                            
                        if len(passage_id_list)==0:
                            continue
                        
                        if table_id not in exist_table:
                            exist_table[table_id] = 1
                            
                            if 'answers' in original_sample:
                                pasg_has_answer = has_answer(original_sample['answers'], full_text, tokenizer, 'string')
                            else:
                                pasg_has_answer = False
                            
                            if len(all_included) == k:
                                break
                            
                            all_included.append({'id':table_id, 'graph_id':doc_id, 'title': table_title, 'text': full_text, 'has_answer': pasg_has_answer, 'score': float(D[j][m]), 'table_name': table_name, 'linked_passage_list': passage_id_list, 'hierarchical_level': level})
                    
                        for passage_id in passage_id_list:
                            
                            if passage_id in exist_passage:
                                continue
                            
                            exist_passage[passage_id] = 1
                            passage = all_passages[passage_id]
                            full_text = header + '\n' + rows[row_id] + '\n' + passage['title'] + ' ' + passage['text']
                            
                            if 'answers' in original_sample:
                                pasg_has_answer = has_answer(original_sample['answers'], full_text, tokenizer, 'string')
                            else:
                                pasg_has_answer = False
                                
                            if len(all_included) == k:
                                break
                            
                            all_included.append({'id':table_id, 'graph_id':doc_id, 'title': table_title, 'text': full_text, 'has_answer': pasg_has_answer, 'score': float(D[j][m]), 'table_name': table_name, 'linked_passage_name': passage_id, 'hierarchical_level': level})
                    
                        if len(all_included) == k:
                            break

                original_sample['ctxs'] = all_included
                time2 = time.time()
                time_list.append(time2-time1)
                
                for l, limit in enumerate(limits):
                    
                    if any([ctx['has_answer'] for ctx in all_included[:limit]]):
                        answer_recall[l] += 1

                new_data.append(original_sample)
                torch.cuda.empty_cache()
            for l, limit in enumerate(limits):
                print ('answer recall', limit, answer_recall[l]/len(data))
            
            print ('average time', np.mean(time_list))
            
            result_path = cfg.result_path.split('.')[0] + '_' + cfg.hierarchical_level
            time_path = cfg.result_path.split('.')[0] + '_' + cfg.hierarchical_level
            
            if cfg.scale is not None:
                result_path += '_' + cfg.scale
                time_path += '_' + cfg.scale
            
            if selection_method == 'topk':
                result_path += f'_{selection_method}_{parameter_name}.json'
                time_path += f'_{selection_method}_{parameter_name}_time.json'
            elif selection_method == 'topp':
                result_path += f'_{selection_method}_{parameter_name}.json'
                time_path += f'_{selection_method}_{parameter_name}_time.json'
            elif selection_method == 'thr':
                result_path += f'_{selection_method}_{parameter_name}.json'
                time_path += f'_{selection_method}_{parameter_name}_time.json'
            else:
                result_path += '.json'
                time_path += '_time.json'

            with open(result_path, 'w') as f:
                json.dump(new_data, f, indent=4)

            with open(time_path, 'w') as f:
                json.dump(time_list, f, indent=4)

if __name__ == "__main__":
    main()