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

logger = logging.getLogger()
setup_logger(logger)

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
    all_tables = json.load(open(cfg.table_data_file_path))
    all_passages = {doc['chunk_id']:doc for doc in json.load(open(cfg.passage_data_file_path))}
    
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
    
    config = ColBERTConfig(
            root=cfg.collection_root_dir_path,
        )
    
    experiment_name = f"ottqa_{cfg.hierarchical_level}_512" #_512
        
    colbert_searcher = Searcher(index=f"{experiment_name}.nbits{cfg.nbits}", config=config, index_root=cfg.index_root)
    
    for i in tqdm(range(0, len(data), b_size)):
        time1 = time.time()
        
        if cfg.scale is None:
            ranking = colbert_searcher.search(text=data[i]['question'], k=100)
        else:
            ranking = colbert_searcher.search(text=data[i]['question'], k=500)
        
        D, I = [ranking[2]], [ranking[0]]
        
        original_sample = data[i]
        all_included = []
        exist_table = {}
        exist_passage = {}
        
        for j, ind in enumerate(I):
            for m, id in enumerate(ind):
                subgraph = graphs[doc_ids[str(id)]]
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
                if 'passage_score_list' in subgraph:
                    passage_id_list = sort_page_ids_by_scores(subgraph['passage_chunk_id_list'], subgraph['passage_score_list'])
                    level = 'star'
                else:
                    passage_id_list = subgraph['passage_chunk_id_list']
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
    
    for l, limit in enumerate(limits):
        print ('answer recall', limit, answer_recall[l]/len(data))
    
    print ('average time', np.mean(time_list))
    
    result_path = cfg.result_path.split('.')[0] + '_' + cfg.hierarchical_level
    time_path = cfg.result_path.split('.')[0] + '_' + cfg.hierarchical_level
    
    if cfg.scale is not None:
        result_path += '_' + cfg.scale
        time_path += '_' + cfg.scale
    
    result_path += '.json'
    time_path += '_time.json'

    with open(result_path, 'w') as f:
        json.dump(new_data, f, indent=4)

    with open(time_path, 'w') as f:
        json.dump(time_list, f, indent=4)

if __name__ == "__main__":
    main()