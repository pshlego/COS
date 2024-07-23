import re
import copy
import json
import time
import hydra
import torch
import unicodedata
from tqdm import tqdm
from omegaconf import DictConfig
from ColBERT.colbert import Searcher
from ColBERT.colbert.infra import ColBERTConfig
from FlagEmbedding import LayerWiseFlagLLMReranker
from Algorithms.Ours.dpr.data.qa_validation import has_answer
from Algorithms.Ours.dpr.utils.tokenizers import SimpleTokenizer


@hydra.main(config_path="conf", config_name="graph_query_algorithm")
def main(cfg: DictConfig):
    # load qa dataset
    print(f"Loading qa dataset...")
    qa_dataset = json.load(open(cfg.qa_dataset_path))
    graph_query_engine = GraphQueryEngine(cfg)
    
    # query
    print(f"Start querying...")
    query_time_list = []
    retrieved_graph_list = []
    cnt = 0
    for qidx, qa_datum in tqdm(enumerate(qa_dataset), total=len(qa_dataset)):
        if cnt == 20:
            break
        nl_question = qa_datum['question']
        init_time = time.time()
        retrieved_graph = graph_query_engine.query(nl_question)
        end_time = time.time()
        query_time_list.append(end_time - init_time)
        retrieved_graph_list.append(retrieved_graph)
        cnt += 1
    
    # save integrated graph
    print(f"Saving integrated graph...")
    json.dump(retrieved_graph_list, open(cfg.integrated_graph_save_path, 'w'))
    json.dump(query_time_list, open(cfg.query_time_save_path, 'w'))
    
    # evaluation
    print(f"Start evaluation...")
    recall, error_cases = evaluate(retrieved_graph_list, qa_dataset, graph_query_engine)
    print(f"Recall: {recall}")
    json.dump(error_cases, open(cfg.error_cases_save_path, 'w'))
 


class GraphQueryEngine:
    def __init__(self, cfg):
        # load data
        ## load graphs
        edge_contents = []
        with open(cfg.edge_dataset_path, "r") as file:
            for line in file:
                edge_contents.append(json.loads(line))
        num_of_edges = len(edge_contents)
        
        self.edge_key_to_content = {}
        print(f"Loading {num_of_edges} graphs...")
        for id, edge_content in enumerate(edge_contents):
            self.edge_key_to_content[edge_content['chunk_id']] = edge_content
        
        ## load corpus
        print(f"Loading corpus...")
        self.table_key_to_content = {}
        self.table_title_to_table_key = {}
        table_contents = json.load(open(cfg.table_data_path))
        for table_key, table_content in enumerate(table_contents):
            self.table_key_to_content[str(table_key)] = table_content
            self.table_title_to_table_key[table_content['chunk_id']] = table_key
        
        self.passage_key_to_content = {}
        passage_contents = json.load(open(cfg.passage_data_path))
        for passage_content in passage_contents:
            self.passage_key_to_content[passage_content['title']] = passage_content

        # load retrievers
        ## id mappings
        self.id_to_edge_key = json.load(open(cfg.edge_ids_path))
        self.id_to_table_key = json.load(open(cfg.table_ids_path))
        self.id_to_passage_key = json.load(open(cfg.passage_ids_path))
        
        ## colbert retrievers
        print(f"Loading index...")
        self.colbert_edge_retriever = Searcher(index=f"{cfg.edge_index_name}.nbits{cfg.nbits}", config=ColBERTConfig(), collection=cfg.collection_edge_path, index_root=cfg.edge_index_root_path, checkpoint=cfg.edge_checkpoint_path)
        self.colbert_table_retriever = Searcher(index=f"{cfg.table_index_name}.nbits{cfg.nbits}", config=ColBERTConfig(), collection=cfg.collection_table_path, index_root=cfg.table_index_root_path, checkpoint=cfg.table_checkpoint_path)
        self.colbert_passage_retriever = Searcher(index=f"{cfg.passage_index_name}.nbits{cfg.nbits}", config=ColBERTConfig(), collection=cfg.collection_passage_path, index_root=cfg.passage_index_root_path, checkpoint=cfg.passage_checkpoint_path)
        self.cross_encoder_edge_retriever = LayerWiseFlagLLMReranker(cfg.reranker_checkpoint_path, use_fp16=True)
        
        # load experimental settings
        self.top_k_of_edge = cfg.top_k_of_edge
        self.top_k_of_table_segment_augmentation = cfg.top_k_of_table_segment_augmentation
        self.top_k_of_passage_augmentation = cfg.top_k_of_passage_augmentation
        self.top_k_of_table_segment = cfg.top_k_of_table_segment
        self.top_k_of_passage = cfg.top_k_of_passage

        self.node_scoring_method = cfg.node_scoring_method
        self.batch_size = cfg.batch_size



    def query(self, nl_question):
        
        # 1. Edge Retrieval
            # Scored edges
        retrieved_edges = self.retrieve_edges(nl_question)
        
        # retrieved_edges =
        # {
        #   "node_id_1": {
        #       "node_type": "table_segment" | "passage", 
        #       "linked_nodes" : [
        #           [target_node_id_1, score_1, retrieval_type_1, source_rank_1, target_rank_1],
        #           ...,
        #           [target_node_id_n, score_n, retrieval_type_n, source_rank_n, target_rank_n]
        #       ]
        #   },
        #   ...,
        #   "node_id_k": {...}
        # }
        
        # 2. Edge Reranking
        reranked_edges = self.reranking_edges(nl_question, retrieved_edges)
        
        # 3. Graph Integration
        integrated_graph = self.integrate_graphs(reranked_edges)
        
        # 4. Score Assignment
        self.assign_scores(integrated_graph)

        topk_table_segment_nodes = []
        topk_passage_nodes = []
        for node_id, node_info in integrated_graph.items():
            if node_info['type'] == 'table segment':
                topk_table_segment_nodes.append([node_id, node_info['score']])
            elif node_info['type'] == 'passage':
                topk_passage_nodes.append([node_id, node_info['score']])
                
        topk_table_segment_nodes = sorted(topk_table_segment_nodes, key=lambda x: x[1], reverse=True)[:self.top_k_of_table_segment_augmentation]
        topk_passage_nodes = sorted(topk_passage_nodes, key=lambda x: x[1], reverse=True)[:self.top_k_of_passage_augmentation]

        # 5.1 Passage Node Augmentation
        self.augment_node(integrated_graph, nl_question, topk_table_segment_nodes, 'table segment', 'passage')

        # 5.2 Table Segment Node Augmentation
        self.augment_node(integrated_graph, nl_question, topk_passage_nodes, 'passage', 'table segment')

        self.assign_scores(integrated_graph)
        
        retrieved_graphs = integrated_graph
        
        return retrieved_graphs
    
    def retrieve_edges(self, nl_question):
        
        retrieved_edges_info = self.colbert_edge_retriever.search(nl_question, 10000)
        
        retrieved_edge_id_list = retrieved_edges_info[0]
        retrieved_edge_score_list = retrieved_edges_info[2]
        
        retrieved_edge_contents = []
        for graphidx, retrieved_id in enumerate(retrieved_edge_id_list[:self.top_k_of_edge]):
            retrieved_edge_content = self.edge_key_to_content[self.id_to_edge_key[str(retrieved_id)]]

            # pass single node graph
            if 'linked_entity_id' not in retrieved_edge_content:
                continue
            
            retrieved_edge_content['edge_score'] = retrieved_edge_score_list[graphidx]
            retrieved_edge_contents.append(retrieved_edge_content)

        return retrieved_edge_contents

    def reranking_edges(self, nl_question, retrieved_edges):
        for retrieved_edge in retrieved_edges:
            edge_text = retrieved_edge['title'] + ' [SEP] ' + retrieved_edge['text']
        
        for i in range(0, len(retrieved_edges), self.batch_size):
            retrieved_edge_batch = retrieved_edges[i:i+self.batch_size]
            model_input = [[nl_question, retrieved_edge['title'] + ' [SEP] ' + retrieved_edge['text']] for retrieved_edge in retrieved_edge_batch]
            with torch.no_grad():
                edge_reranking_scores = self.cross_encoder_edge_retriever.compute_score(model_input, batch_size=self.batch_size, cutoff_layers=[40], max_length=256)
            
            for edge_reranking_score, retrieved_edge in zip(edge_reranking_scores, retrieved_edge_batch):
                retrieved_edge['edge_reranking_score'] = float(edge_reranking_score)

        return retrieved_edges

    def integrate_graphs(self, retrieved_graphs):
        integrated_graph = {}
        retrieval_type='edge_retrieval'
        # graph integration
        for retrieved_graph_content in retrieved_graphs:
            graph_chunk_id = retrieved_graph_content['chunk_id']

            # get table segment node info
            table_key = str(retrieved_graph_content['table_id'])
            row_id = graph_chunk_id.split('_')[1]
            table_segment_node_id = f"{table_key}_{row_id}"
            
            # get passage node info
            passage_id = retrieved_graph_content['linked_entity_id']
            
            # get two node graph score
            edge_reranking_score = retrieved_graph_content['edge_reranking_score']
            
            # add nodes
            self.add_node(integrated_graph, 'table segment', table_segment_node_id, passage_id, edge_reranking_score, retrieval_type)
            self.add_node(integrated_graph, 'passage', passage_id, table_segment_node_id, edge_reranking_score, retrieval_type)

            # node scoring
            self.assign_scores(integrated_graph)

        return integrated_graph

    def augment_node(self, graph, nl_question, topk_query_nodes, query_node_type, retrieved_node_type):
        for source_rank, (query_node_id, query_node_score) in enumerate(topk_query_nodes):
            expanded_query = self.get_expanded_query(nl_question, query_node_id, query_node_type)
            
            if query_node_type == 'table segment':
                retrieved_node_info = self.colbert_passage_retriever.search(expanded_query, 20)
                retrieved_id_list = retrieved_node_info[0]
                retrieved_score_list = retrieved_node_info[2]
                top_k = self.top_k_of_passage
            else:
                retrieved_node_info = self.colbert_table_retriever.search(expanded_query, 20)
                retrieved_id_list = retrieved_node_info[0]
                retrieved_score_list = retrieved_node_info[2]
                top_k = self.top_k_of_table_segment

            for target_rank, retrieved_id in enumerate(retrieved_id_list):
                if target_rank >= top_k:
                    break
                
                if query_node_type == 'table segment':
                    retrieved_node_id = self.id_to_passage_key[str(retrieved_id)]
                    augment_type = f'passage_node_augmentation'
                else:
                    retrieved_node_id = self.id_to_table_key[str(retrieved_id)]
                    augment_type = f'table_segment_node_augmentation'
                
                self.add_node(graph, query_node_type, query_node_id, retrieved_node_id, query_node_score, augment_type, source_rank, target_rank)
                self.add_node(graph, retrieved_node_type, retrieved_node_id, query_node_id, query_node_score, augment_type, target_rank, source_rank)

    def get_expanded_query(self, nl_question, node_id, query_node_type):
        if query_node_type == 'table segment':
            table_key = node_id.split('_')[0]
            table = self.table_key_to_content[table_key]
            table_title = table['title']
            
            row_id = int(node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            
            expanded_query = f"{nl_question} [SEP] {table_title} [SEP] {column_name} [SEP] {row_values}"
        else:
            passage = self.passage_key_to_content[node_id]
            passage_title = passage['title']
            passage_text = passage['text']
            
            expanded_query = f"{nl_question} [SEP] {passage_title} [SEP] {passage_text}"
        
        return expanded_query

    def add_node(self, graph, source_node_type, source_node_id, target_node_id, score, retrieval_type, source_rank=0, target_rank=0):
        # add source and target node
        if source_node_id not in graph:
            graph[source_node_id] = {'type': source_node_type, 'linked_nodes': [[target_node_id, score, retrieval_type, source_rank, target_rank]]}
        # add target node
        else:
            graph[source_node_id]['linked_nodes'].append([target_node_id, score, retrieval_type, source_rank, target_rank])
    
    def assign_scores(self, graph):
        for node_id, node_info in graph.items():
            linked_scores = [linked_node[1] for linked_node in node_info['linked_nodes']]
            
            if self.node_scoring_method == 'min':
                node_score = min(linked_scores)
            elif self.node_scoring_method == 'max':
                node_score = max(linked_scores)
            elif self.node_scoring_method == 'mean':
                node_score = sum(linked_scores) / len(linked_scores)

            graph[node_id]['score'] = node_score
    
def remove_accents_and_non_ascii(text):
    # Normalize the text to decompose combined characters
    normalized_text = unicodedata.normalize('NFD', text)
    # Remove all characters that are not ASCII
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining non-letter characters
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,!.?\-]', '', ascii_text)
    return cleaned_text

def evaluate(retrieved_graph_list, qa_dataset, graph_query_engine):
    recall_list = []
    error_cases = {}
    table_key_to_content = graph_query_engine.table_key_to_content
    passage_key_to_content = graph_query_engine.passage_key_to_content
    for retrieved_graph, qa_datum in zip(retrieved_graph_list, qa_dataset):
        node_count = 0
        edge_count = 0
        answers = qa_datum['answers']
        context = ""
        all_included = []
        retrieved_table_set = set()
        retrieved_passage_set = set()
        sorted_retrieved_graph = sorted(retrieved_graph.items(), key=lambda x: x[1]['score'], reverse=True)
        
        for node_id, node_info in sorted_retrieved_graph:
            if node_info['type'] == 'table segment':
                
                table_id = node_id.split('_')[0]
                table = table_key_to_content[table_id]
                chunk_id = table['chunk_id']
                node_info['chunk_id'] = chunk_id
                
                if table_id not in retrieved_table_set:
                    retrieved_table_set.add(table_id)
                    
                    if edge_count == 50:
                        continue
                    
                    context += table['text']
                    edge_count += 1
                    
                max_linked_node_id, max_score, _, _, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default=(None, 0, 'edge_reranking', 0, 0))
                
                if max_linked_node_id in retrieved_passage_set:
                    continue
                
                retrieved_passage_set.add(max_linked_node_id)
                passage_content = passage_key_to_content[max_linked_node_id]
                passage_text = passage_content['title'] + ' ' + passage_content['text']
                
                row_id = int(node_id.split('_')[1])
                table_rows = table['text'].split('\n')
                column_name = table_rows[0]
                row_values = table_rows[row_id+1]
                table_segment_text = column_name + '\n' + row_values
                
                edge_text = table_segment_text + '\n' + passage_text
                
                if edge_count == 50:
                    continue
                
                edge_count += 1
                context += edge_text
            
            elif node_info['type'] == 'passage':

                if node_id in retrieved_passage_set:
                    continue

                max_linked_node_id, max_score, _, _, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default=(None, 0, 'edge_reranking', 0, 0))
                table_id = max_linked_node_id.split('_')[0]
                table = table_key_to_content[table_id]
                
                if table_id not in retrieved_table_set:
                    retrieved_table_set.add(table_id)
                    if edge_count == 50:
                        continue
                    context += table['text']
                    edge_count += 1

                row_id = int(max_linked_node_id.split('_')[1])
                table_rows = table['text'].split('\n')
                column_name = table_rows[0]
                row_values = table_rows[row_id+1]
                table_segment_text = column_name + '\n' + row_values
                
                if edge_count == 50:
                    continue

                retrieved_passage_set.add(node_id)
                passage_content = passage_key_to_content[node_id]
                passage_text = passage_content['title'] + ' ' + passage_content['text']
                
                edge_text = table_segment_text + '\n' + passage_text
                context += edge_text
                edge_count += 1

            node_count += 1

        normalized_context = remove_accents_and_non_ascii(context)
        normalized_answers = [remove_accents_and_non_ascii(answer) for answer in answers]
        is_has_answer = has_answer(normalized_answers, normalized_context, SimpleTokenizer(), 'string')
        
        if is_has_answer:
            recall_list.append(1)
        else:
            recall_list.append(0)
            new_qa_datum = copy.deepcopy(qa_datum)
            new_qa_datum['retrieved_graph'] = retrieved_graph
            new_qa_datum['sorted_retrieved_graph'] = sorted_retrieved_graph[:node_count]
            if  "hard_negative_ctxs" in new_qa_datum:
                del new_qa_datum["hard_negative_ctxs"]
            
            error_cases[qa_datum['id']] = new_qa_datum


    recall = sum(recall_list) / len(recall_list)
    
    return recall, error_cases


if __name__ == "__main__":
    main()