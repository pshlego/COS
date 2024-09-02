import hydra
import json
from omegaconf import DictConfig
from pymongo import MongoClient
from tqdm import tqdm
import random
import os

@hydra.main(config_path="conf", config_name="create_training_dataset")
def main(cfg: DictConfig):
    triples_path = cfg.triples_path
    queries_path = cfg.queries_path
    collection_path = cfg.collection_path
    training_dict_path = cfg.training_dict_path
    
    client = MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password))
    mongodb = client[cfg.dbname]
    
    with open('/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_dev_q_to_tables_with_bm25neg.json') as f:
        data = json.load(f)

    passage_collection = mongodb[cfg.passage_collection]
    total_passages = passage_collection.count_documents({})
    print(f"Loading {total_passages} passages...")
    passages = {doc['chunk_id']: doc for doc in tqdm(passage_collection.find(), total=total_passages)}

    table_collection = mongodb[cfg.table_collection]
    total_tables = table_collection.count_documents({})
    print(f"Loading {total_tables} tables...")
    tables = {doc['chunk_id']: i for i, doc in tqdm(enumerate(table_collection.find()), total=total_tables)}

    graph_collection = mongodb[cfg.graph_collection]
    total_graphs = graph_collection.count_documents({})
    print(f"Loading {total_graphs} graphs...")
    graphs = {doc['chunk_id']: doc for doc in tqdm(graph_collection.find(), total=total_graphs)}
    
    training_dict = {}
    did = 0
    triples_lines = []
    queries_lines = []
    collection_lines = []
    
    for qid, datum in tqdm(enumerate(data), total=len(data)):
        training_dict[str(qid)] = {}
        positive_graph_list = []
        negative_graph_list = []
        
        question = datum['question']
        positive_ctxs = datum['positive_ctxs']
        
        for positive_ctx in positive_ctxs:
            table_chunk_id = positive_ctx['chunk_id']
            table_title = positive_ctx['title']
            column_names = positive_ctx['text'].split('\n')[0]
            table_text = positive_ctx['text'].split('\n')[1:]
            
            if table_text[-1]=='':
                table_text = table_text[:-1]
            
            for answer in positive_ctx['answer_node']:
                passage_name = answer[0]
                real_row_id = answer[1][0]
                row_id = positive_ctx['rows'].index(real_row_id)
                row = table_text[row_id]
                
                try:
                    passage_text = passages[passage_name]['title'] + ' [SEP] ' + passages[passage_name]['text']
                except:
                    star_graph_id = f"{tables[table_chunk_id]}_{row_id}"
                    star_graph = graphs[star_graph_id]
                    if 'mentions_in_row_info_dict' in star_graph:
                        mentions_in_row_info_dict = star_graph['mentions_in_row_info_dict']
                        if mentions_in_row_info_dict=={}:
                            passage_text = ""
                        else:
                            random_mention = random.choice(list(mentions_in_row_info_dict.keys()))
                            passage_name = mentions_in_row_info_dict[random_mention]['mention_linked_entity_id_list'][0]
                            passage_text = passages[passage_name]['title'] + ' [SEP] ' + passages[passage_name]['text']
                    else:
                        passage_text = ""
                
                graph_text = table_title + ' [SEP] ' + column_names + ' [SEP] ' + row + ' [SEP] ' + passage_text
                positive_graph_list.append(graph_text)

        hard_negative_ctxs = datum['hard_negative_ctxs']
        for hard_negative_ctx in hard_negative_ctxs:
            table_chunk_id = hard_negative_ctx['chunk_id']
            table_title = hard_negative_ctx['title']
            column_names = hard_negative_ctx['text'].split('\n')[0]
            table_text = hard_negative_ctx['text'].split('\n')[1:]
            
            if table_text[-1]=='':
                table_text = table_text[:-1]
            
            row_id = random.choice(range(len(table_text)))
            row = table_text[row_id]
            star_graph_id = f"{tables[table_chunk_id]}_{row_id}"
            star_graph = graphs[star_graph_id]
            
            if 'mentions_in_row_info_dict' in star_graph:
                mentions_in_row_info_dict = star_graph['mentions_in_row_info_dict']
                star_passage_text = star_graph['text'].split('[SEP]')[:2*(len(list(mentions_in_row_info_dict.keys()))+1)]
                if mentions_in_row_info_dict=={}:
                    passage_text = ""
                else:
                    random_mention = random.choice(list(mentions_in_row_info_dict.keys()))
                    passage_name = mentions_in_row_info_dict[random_mention]['mention_linked_entity_id_list'][0]
                    passage_text = passages[passage_name]['title'] + ' [SEP] ' + passages[passage_name]['text']
            else:
                passage_text = ""
            
            #edge
            graph_text = table_title + ' [SEP] ' + column_names + ' [SEP] ' + row + ' [SEP] ' + passage_text
            
            #star
            star_graph_text = table_title + ' [SEP] ' + '[SEP]'.join(star_passage_text)
            negative_graph_list.append(graph_text)
            negative_graph_list.append(star_graph_text)

        question_line =f"{qid}\t{question}\n"
        queries_lines.append(question_line)
        training_dict[str(qid)]['positive'] = []
        training_dict[str(qid)]['negative'] = []
        
        for graph in positive_graph_list:
            collection_lines.append(f"{did}\t{graph}\n")
            training_dict[str(qid)]['positive'].append(did)
            did += 1
            
        for graph in negative_graph_list:
            collection_lines.append(f"{did}\t{graph}\n")
            training_dict[str(qid)]['negative'].append(did)
            did += 1
        
        for positive_id in training_dict[str(qid)]['positive']:
            for negative_id in training_dict[str(qid)]['negative']:
                triple_line = f"{qid}\t{positive_id}\t{negative_id}\n"
                triples_lines.append(triple_line)

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