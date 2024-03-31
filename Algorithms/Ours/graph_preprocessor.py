import hydra
import json
from omegaconf import DictConfig
from pymongo import MongoClient
from tqdm import tqdm
import os
import logging
import os
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import logging
logging.disable(logging.WARNING)

def preprocess_graph(cfg, mongodb):
    hierarchical_level = cfg.hierarchical_level
    
    if cfg.top_k_passages > 1:
        preprocessed_graph_path = cfg.preprocessed_graph_path.replace('.json', f'_{hierarchical_level}_{cfg.top_k_passages}.json')
    else:
        preprocessed_graph_path = cfg.preprocessed_graph_path.replace('.json', f'_{hierarchical_level}.json')
    
    preprocess_graph_collection_name = os.path.basename(preprocessed_graph_path).split('.')[0]
    collection_list = mongodb.list_collection_names()
    
    if preprocess_graph_collection_name not in collection_list:
        
        graph_collection = mongodb[cfg.graph_collection_name]
        
        if cfg.graph_collection_name not in mongodb.list_collection_names():
            with open(cfg.graph_path, 'r') as fin:
                data = json.load(fin)
            graph_collection.insert_many(data)
            total_graphs = graph_collection.count_documents({})
            print(f"Loading {total_graphs} graphs...")
            graph = {doc['table_chunk_id']: doc for doc in tqdm(graph_collection.find(), total=total_graphs)}
        else:
            total_graphs = graph_collection.count_documents({})
            print(f"Loading {total_graphs} graphs...")
            graph = {doc['table_chunk_id']: doc for doc in tqdm(graph_collection.find(), total=total_graphs)}
        
        print("finish loading graph")
        
        # Pre-fetch collections to minimize database queries
        table_collection = mongodb[cfg.table_collection_name]
        total_tables = table_collection.count_documents({})
        print(f"Loading {total_tables} tables...")
        all_tables = [doc for doc in tqdm(table_collection.find(), total=total_tables)]
        print("finish loading tables")
        
        passage_collection = mongodb[cfg.passage_collection_name]
        total_passages = passage_collection.count_documents({})
        print(f"Loading {total_passages} passages...")
        all_passages = {doc['chunk_id']: doc for doc in tqdm(passage_collection.find(), total=total_passages)}
        print("finish loading passages")
        
        table_mention_collection = mongodb[cfg.table_mention_collection_name]
        total_table_mentions = table_mention_collection.count_documents({})
        print(f"Loading {total_table_mentions} table mentions...")
        node_id_to_chunk_id = {doc['node_id']:doc['chunk_id'] for doc in tqdm(table_mention_collection.find(), total=total_table_mentions)}
        print("finish loading table mentions")
        
        node_list = []
        if hierarchical_level == 'star':
            
            for node_id, raw_table in tqdm(enumerate(all_tables), total=len(all_tables)):
                table_chunk_id = node_id_to_chunk_id[node_id]
                table_title = raw_table['title']
                column_names, *table_rows = raw_table['text'].split('\n')
                
                try:
                    node = graph[table_chunk_id]
                except:
                    for row_id, table_row in enumerate(table_rows):
                        row_node_context = f"{column_names} [SEP] {table_row}"
                        node_list.append({
                        'chunk_id': f'{node_id}_{row_id}',
                        'title': table_title,
                        'text': row_node_context,
                        'table_id': node_id
                        })
                    continue
                
                grounding_info = node['results']
                
                row_dict = {}
                
                for row_id, row in enumerate(table_rows):
                    row_dict[row_id] = []
                
                for mentions in grounding_info:
                    row_dict[mentions['row']].append(mentions)

                for row_id, mentions in row_dict.items():
                    row_entity_id_set = set()
                    row_linked_passage_context = []
                    mentions_in_row_info_dict = {}
                    
                    for topk in range(cfg.top_k_passages):
                        
                        for mention_id, mention in enumerate(mentions):
                            original_cell = mention['original_cell']
                            mention_linked_entity_id = mention['retrieved'][topk]

                            if str(mention_id) not in mentions_in_row_info_dict:
                                mentions_in_row_info_dict[str(mention_id)] = {
                                    'original_cell': original_cell,
                                    'mention_linked_entity_id_list': [],
                                    'mention_linked_entity_score_list': []
                                }
                            
                            raw_passage = all_passages[mention_linked_entity_id]
                            mention_linked_entity_text = f"{raw_passage['title']} [SEP] {raw_passage['text']}"
                            mention_linked_entity_score = mention['scores'][topk]
                            
                            mentions_in_row_info_dict[str(mention_id)]['mention_linked_entity_id_list'].append(mention_linked_entity_id)
                            mentions_in_row_info_dict[str(mention_id)]['mention_linked_entity_score_list'].append(mention_linked_entity_score)

                            # exculde duplicate passages
                            if mention_linked_entity_id not in row_entity_id_set:
                                row_entity_id_set.add(mention_linked_entity_id)
                                row_linked_passage_context.append(mention_linked_entity_text)
                    
                    row_node_context = f"{column_names} [SEP] {table_rows[row_id]} [SEP] {' [SEP] '.join(row_linked_passage_context)}"
                    
                    node_list.append({
                        'chunk_id': f'{node_id}_{row_id}',
                        'title': table_title,
                        'text': row_node_context,
                        'table_id': node_id,
                        'mentions_in_row_info_dict': mentions_in_row_info_dict
                    })

        elif hierarchical_level == 'edge':
            
            for node_id, raw_table in tqdm(enumerate(all_tables), total=len(all_tables)):
                table_chunk_id = node_id_to_chunk_id[node_id]
                table_title = raw_table['title']
                column_names, *table_rows = raw_table['text'].split('\n')
                
                try:
                    node = graph[table_chunk_id]
                except:
                    for row_id, table_row in enumerate(table_rows):
                        row_node_context = f"{column_names} [SEP] {table_row}"
                        node_list.append({
                        'chunk_id': f'{node_id}_{row_id}',
                        'title': table_title,
                        'text': row_node_context,
                        'table_id': node_id
                        })
                    continue
                
                grounding_info = node['results']
                
                row_dict = {}
                for row_id, row in enumerate(table_rows):
                    row_dict[row_id] = []
                
                for mentions in grounding_info:
                    row_dict[mentions['row']].append(mentions)
                
                for row_id, mentions in row_dict.items():
                    row_entity_id_set = set()
                    
                    for topk in range(cfg.top_k_passages):
                        
                        for mention_id, mention in enumerate(mentions):
                            original_cell = mention['original_cell']
                            mention_linked_entity_id = mention['retrieved'][topk]
                            
                            if mention_linked_entity_id not in row_entity_id_set:
                                row_entity_id_set.add(mention_linked_entity_id)
                            
                                raw_passage = all_passages[mention_linked_entity_id]
                                mention_linked_entity_text = f"{raw_passage['title']} [SEP] {raw_passage['text']}"
                                mention_linked_entity_score = mention['scores'][topk]
                                mention_node_context = f"{column_names} [SEP] {table_rows[row_id]} [SEP] " + mention_linked_entity_text
                                
                                node_list.append({
                                    'chunk_id': f'{node_id}_{row_id}_{mention_id}_{topk}',
                                    'title': table_title,
                                    'text': mention_node_context,
                                    'table_id': node_id,
                                    'mention_id': mention_id,
                                    'original_cell': original_cell,
                                    'linked_entity_id': mention_linked_entity_id,
                                    'linked_entity_score': mention_linked_entity_score,
                                    'topk': topk
                                })
                        
                        if len(mentions) == 0:
                            row_node_context = f"{column_names} [SEP] {table_rows[row_id]}"
                            node_list.append({
                                'chunk_id': f'{node_id}_{row_id}',
                                'title': table_title,
                                'text': row_node_context,
                                'table_id': node_id,
                            })
                            break

        # with open(preprocessed_graph_path, 'w') as fout:
        #     json.dump(node_list, fout, indent=4)
        
        preprocess_graph_collection = mongodb[preprocess_graph_collection_name]
        preprocess_graph_collection.insert_many(node_list)

@hydra.main(config_path="conf", config_name="graph_preprocessor")
def main(cfg: DictConfig):
    client = MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password))
    mongodb = client[cfg.dbname]
    preprocess_graph(cfg, mongodb)

if __name__ == "__main__":
    main()