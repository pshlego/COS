import hydra
import json
from tqdm import tqdm
from pymongo import MongoClient
from omegaconf import DictConfig
from ColBERT.colbert import Searcher
from ColBERT.colbert.infra import ColBERTConfig

@hydra.main(config_path="conf", config_name="graph_query_algorithm")
def main(cfg: DictConfig):
    # mongodb setup
    client = MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password))
    mongodb = client[cfg.dbname]
    
    # load dataset
    ## qa dataset
    print(f"Loading qa dataset...")
    qa_dataset = json.load(open(cfg.qa_dataset_path))

    ## two node graphs
    two_node_graph_contents = mongodb[cfg.two_node_graph_name]
    num_of_two_node_graphs = two_node_graph_contents.count_documents({})
    two_node_graph_key_to_content = {}
    print(f"Loading {num_of_two_node_graphs} graphs...")
    for two_node_graph_content in tqdm(two_node_graph_contents.find(), total=num_of_two_node_graphs):
        two_node_graph_key_to_content[two_node_graph_content['chunk_id']] = two_node_graph_content

    ## corpus
    print(f"Loading corpus...")
    table_key_to_content = json.load(open(cfg.table_data_path))
    
    passage_key_to_content = {}
    raw_passages = json.load(open(cfg.passage_data_path))
    for i, passage in enumerate(raw_passages):
        passage_key_to_content[passage['title']] = passage
    
    # load retrievers
    ## id mappings
    id_to_two_node_graph_key = json.load(open(cfg.two_node_graph_ids_path))
    id_to_table_key = json.load(open(cfg.table_ids_path))
    id_to_passage_key = json.load(open(cfg.passage_ids_path))
    
    ## colbert retrievers
    two_node_graph_config = ColBERTConfig(root=cfg.collection_two_node_graph_root_dir_path)
    table_config = ColBERTConfig(root=cfg.collection_table_root_dir_path)
    passage_config = ColBERTConfig(root=cfg.collection_passage_root_dir_path)

    two_node_graph_index_name = cfg.two_node_graph_index_name
    table_index_name = cfg.table_index_name
    passage_index_name = cfg.passage_index_name

    print(f"Loading index...")
    colbert_two_node_graph_retriever = Searcher(index=f"{two_node_graph_index_name}.nbits{cfg.nbits}", config=two_node_graph_config, index_root=cfg.two_node_graph_index_root_path)
    colbert_table_retriever = Searcher(index=f"{table_index_name}.nbits{cfg.nbits}", config=table_config, index_root=cfg.table_index_root_path)
    colbert_passage_retriever = Searcher(index=f"{passage_index_name}.nbits{cfg.nbits}", config=passage_config, index_root=cfg.passage_index_root_path)
    
    # load experimental settings
    top_k_of_two_node_graph = cfg.top_k_of_two_node_graph
    top_k_of_table_augmentation = cfg.top_k_of_table_augmentation
    top_k_of_passage_augmentation = cfg.top_k_of_passage_augmentation
    top_k_of_table = cfg.top_k_of_table
    top_k_of_passage = cfg.top_k_of_passage
    
    node_scoring_method = cfg.node_scoring_method
    
    integrated_graph_list = []
    
    # query
    print(f"Start querying...")
    num_of_queries = len(qa_dataset)
    for qidx, qa_datum in tqdm(enumerate(qa_dataset), total=num_of_queries):
        nl_question = qa_datum['question']
        answer = qa_datum['answers']
        
        ## two-node graph retrieval
        retrieved_two_node_graphs_info = colbert_two_node_graph_retriever.search(nl_question, top_k_of_two_node_graph)
        retrieved_two_node_graph_id_list = retrieved_two_node_graphs_info[0]
        retrieved_two_node_graph_score_list = retrieved_two_node_graphs_info[2]
        
        ## graph integration
        integrated_graph = {}
        for graphidx, retrieved_id in enumerate(retrieved_two_node_graph_id_list):
            retrieved_two_node_graph_content = two_node_graph_key_to_content[id_to_two_node_graph_key[str(retrieved_id)]]
            
            ## pass single node graph
            if 'linked_entity_id' not in retrieved_two_node_graph_content:
                continue
            
            graph_id = retrieved_two_node_graph_content['chunk_id']
            
            ### get table segment node info
            table_key = retrieved_two_node_graph_content['table_id']
            table = table_key_to_content[table_key]
            table_chunk_id = table['chunk_id']
            row_id = int(graph_id.split('_')[1])
            table_segment_node_id = f"{table_key}_{row_id}"
            
            ### get passage node info
            passage_id = retrieved_two_node_graph_content['linked_entity_id']
            
            ### get two node graph score
            two_node_graph_score = retrieved_two_node_graph_score_list[graphidx]
            
            if table_segment_node_id not in integrated_graph:
                integrated_graph[table_segment_node_id] = {'type': 'table segment', 'table_chunk_id':table_chunk_id,'linked_passage_nodes': [[passage_id, two_node_graph_score, 'two_node_graph_retrieval']]}
            else:
                integrated_graph[table_segment_node_id]['linked_passage_nodes'].append([passage_id, two_node_graph_score, 'two_node_graph_retrieval'])
            
            if passage_id not in integrated_graph:
                integrated_graph[passage_id] = {'type': 'passage', 'linked_table_segment_nodes': [[table_segment_node_id, two_node_graph_score, 'two_node_graph_retrieval']]}
            else:
                integrated_graph[passage_id]['linked_table_segment_nodes'].append([table_segment_node_id, two_node_graph_score, 'two_node_graph_retrieval'])
        
        ### node scoring
        for node_id, node_info in integrated_graph.items():
            
            if node_info['type'] == 'table segment':
                linked_nodes = node_info['linked_passage_nodes']
            elif node_info['type'] == 'passage':
                linked_nodes = node_info['linked_table_segment_nodes']
            
            linked_scores = [linked_node[1] for linked_node in linked_nodes]
            
            if node_scoring_method == 'min':
                node_score = min(linked_scores)
            elif node_scoring_method == 'max':
                node_score = max(linked_scores)
            elif node_scoring_method == 'mean':
                node_score = sum(linked_scores) / len(linked_scores)
            
            integrated_graph[node_id]['score'] = node_score
        
        ## augmentation retrieval
        ### get topk retrieved nodes
        topk_table_segment_nodes = []
        topk_passage_nodes = []
        for node_id, node_info in integrated_graph.items():
            if node_info['type'] == 'table segment':
                topk_table_segment_nodes.append([node_id, node_info['score']])
            elif node_info['type'] == 'passage':
                topk_passage_nodes.append([node_id, node_info['score']])

        topk_table_segment_nodes = sorted(topk_table_segment_nodes, key=lambda x: x[1], reverse=True)[:top_k_of_table_augmentation]
        topk_passage_nodes = sorted(topk_passage_nodes, key=lambda x: x[1], reverse=True)[:top_k_of_passage_augmentation]
        
        ## passage node augmentation
        for node_id, node_score in topk_table_segment_nodes:
            table_key = int(node_id.split('_')[0])
            table = table_key_to_content[table_key]
            table_title = table['title']

            row_id = int(node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            
            expanded_query = f"{nl_question} [SEP] {table_title} [SEP] {column_name} [SEP] {row_values}"
            retrieved_passage_info = colbert_passage_retriever.search(expanded_query, top_k_of_passage)
            retrieved_passage_id_list = retrieved_passage_info[0]
            retrieved_passage_score_list = retrieved_passage_info[2]
            
            for pid, retrieved_passage_id in enumerate(retrieved_passage_id_list):
                passage_node_id = id_to_passage_key[str(retrieved_passage_id)].replace('/wiki/', '').replace('_', ' ')

                if passage_node_id not in integrated_graph:
                    integrated_graph[node_id]['linked_passage_nodes'].append([passage_node_id, retrieved_two_node_graph_score_list[pid], 'passage_node_augmentation'])
                    integrated_graph[passage_node_id] = {'type': 'passage', 'linked_table_segment_nodes': [[node_id, retrieved_two_node_graph_score_list[pid], 'passage_node_augmentation']]}
                elif passage_node_id not in [node[0] for node in integrated_graph[node_id]['linked_passage_nodes']]:
                    integrated_graph[node_id]['linked_passage_nodes'].append([passage_node_id, retrieved_two_node_graph_score_list[pid], 'passage_node_augmentation'])
                    integrated_graph[passage_node_id]['linked_table_segment_nodes'].append([node_id, retrieved_two_node_graph_score_list[pid], 'passage_node_augmentation'])
                
        ## table segment node augmentation
        for node_id, node_score in topk_passage_nodes:
            passage = passage_key_to_content[node_id]
            passage_title = passage['title']
            passage_text = passage['text']
            
            expanded_query = f"{nl_question} [SEP] {passage_title} [SEP] {passage_text}"
            retrieved_table_segment_info = colbert_table_retriever.search(expanded_query, top_k_of_table)
            retrieved_table_segment_id_list = retrieved_table_segment_info[0]
            retrieved_table_segment_score_list = retrieved_table_segment_info[2]
            
            for tid, retrieved_table_id in enumerate(retrieved_table_segment_id_list):
                table_node_id = id_to_table_key[str(retrieved_table_id)]
                
                if table_node_id not in integrated_graph:
                    integrated_graph[node_id]['linked_table_segment_nodes'].append([retrieved_table_id, retrieved_two_node_graph_score_list[tid], 'table_segment_node_augmentation'])
                    integrated_graph[table_node_id] = {'type': 'table segment', 'linked_passage_nodes': [[node_id, retrieved_two_node_graph_score_list[tid]], 'table_segment_node_augmentation']}
                elif table_node_id not in [node[0] for node in integrated_graph[node_id]['linked_table_segment_nodes']]:
                    integrated_graph[node_id]['linked_table_segment_nodes'].append([table_node_id, retrieved_two_node_graph_score_list[tid], 'table_segment_node_augmentation'])
                    integrated_graph[table_node_id]['linked_passage_nodes'].append([node_id, retrieved_two_node_graph_score_list[tid], 'table_segment_node_augmentation'])
        
        ### node re-scoring
        for node_id, node_info in integrated_graph.items():
            if 'score' in node_info:
                continue

            if node_info['type'] == 'table segment':
                linked_nodes = node_info['linked_passage_nodes']
            elif node_info['type'] == 'passage':
                linked_nodes = node_info['linked_table_segment_nodes']
            
            linked_scores = [linked_node[1] for linked_node in linked_nodes]
            
            if node_scoring_method == 'min':
                node_score = min(linked_scores)
            elif node_scoring_method == 'max':
                node_score = max(linked_scores)
            elif node_scoring_method == 'mean':
                node_score = sum(linked_scores) / len(linked_scores)
            
            integrated_graph[node_id]['score'] = node_score

        ## save integrated graph
        integrated_graph_list.append(integrated_graph)
    
    ## save integrated graph
    print(f"Saving integrated graph...")
    json.dump(integrated_graph_list, open(cfg.integrated_graph_save_path, 'w'))
        
if __name__ == "__main__":
    main()
    
# for i, passage in enumerate(all_passages):
#     new_passages[passage['title']] = passage