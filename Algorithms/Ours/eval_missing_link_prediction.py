import os
import json
import time
import hydra
import torch
from tqdm import tqdm
from pymongo import MongoClient
from omegaconf import DictConfig
from ColBERT.colbert import Searcher
from ColBERT.colbert.infra import ColBERTConfig
from Ours.table_retriever import TableRetriever
from Ours.dpr.data.qa_validation import has_answer
from Ours.dpr.utils.tokenizers import SimpleTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from FlagEmbedding import LayerWiseFlagLLMReranker

class GraphQueryEngine:
    def __init__(self, cfg):
        # mongodb setup
        client = MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password))
        mongodb = client[cfg.dbname]

        # load dataset
        ## two node graphs
        # edge_contents = mongodb[cfg.edge_name]
        # num_of_edges = edge_contents.count_documents({})
        # self.edge_key_to_content = {}
        # self.table_key_to_edge_keys = {}
        # print(f"Loading {num_of_edges} graphs...")
        # for id, edge_content in tqdm(enumerate(edge_contents.find()), total=num_of_edges):
        #     self.edge_key_to_content[edge_content['chunk_id']] = edge_content
            
        #     if str(edge_content['table_id']) not in self.table_key_to_edge_keys:
        #         self.table_key_to_edge_keys[str(edge_content['table_id'])] = []

        #     self.table_key_to_edge_keys[str(edge_content['table_id'])].append(id)

        # load retrievers
        ## id mappings
        # self.id_to_edge_key = json.load(open(cfg.edge_ids_path))
        self.id_to_passage_key = json.load(open(cfg.passage_ids_path))
        
        ## colbert retrievers
        print(f"Loading index...")
        # self.colbert_edge_retriever = Searcher(index=f"{cfg.edge_index_name}.nbits{cfg.nbits}", config=ColBERTConfig(), collection=cfg.collection_edge_path, index_root=cfg.edge_index_root_path)
        self.colbert_passage_retriever = Searcher(index=f"{cfg.passage_index_name}.nbits{cfg.nbits}", config=ColBERTConfig(), collection=cfg.collection_passage_path, index_root=cfg.passage_index_root_path, checkpoint=cfg.passage_checkpoint_path)
        
        # load experimental settings
        self.top_k_of_passage = 2 #cfg.top_k_of_edge

        self.node_scoring_method = cfg.node_scoring_method
        self.batch_size = cfg.batch_size

    def query(self, nl_question, table, row_id, retrieval_time = 2):
        
        # 1. Edge Retrieval
        expanded_query = self.get_expanded_query(nl_question, table, row_id)

        retrieved_node_info = self.colbert_passage_retriever.search(expanded_query, 5)
        retrieved_id_list = retrieved_node_info[0]
        retrieved_score_list = retrieved_node_info[2]
        top_k = self.top_k_of_passage

        retrieved_passage_list = []
        
        for target_rank, retrieved_id in enumerate(retrieved_id_list):
            retrieved_node_id = self.id_to_passage_key[str(retrieved_id)]
            retrieved_passage_list.append(retrieved_node_id)

        return retrieved_passage_list[:top_k]
    
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
            
            # delete '_id' key
            if '_id' in retrieved_edge_content:
                del retrieved_edge_content['_id']
            
            retrieved_edge_content['edge_score'] = float(retrieved_edge_score_list[graphidx])
            retrieved_edge_contents.append(retrieved_edge_content)

        return retrieved_edge_contents

    def reranking_edges(self, nl_question, retrieved_graphs):
        edges = []
        edges_set = set()
        for node_id, node_info in retrieved_graphs.items():
            
            if node_info['type'] == 'table segment':
                table_id = node_id.split('_')[0]
                row_id = int(node_id.split('_')[1])
                table = self.table_key_to_content[table_id]
                table_title = table['title']
                table_rows = table['text'].split('\n')
                column_names = table_rows[0]
                row_values = table_rows[row_id+1]
                table_text = table_title + ' [SEP] ' + column_names + ' [SEP] ' + row_values

                for linked_node in node_info['linked_nodes']:
                    
                    linked_node_id = linked_node[0]
                    edge_id = f"{node_id}_{linked_node_id}"
                    
                    if edge_id not in edges_set:
                        edges_set.add(edge_id)
                    else:
                        continue
                    
                    passage_text = self.passage_key_to_content[linked_node_id]['text']
                    graph_text = table_text + ' [SEP] ' + passage_text
                    edges.append({'table_segment_node_id': node_id, 'passage_id': linked_node_id, 'text': graph_text})

        for i in range(0, len(edges), self.batch_size):
            edge_batch = edges[i:i+self.batch_size]
            model_input = [[nl_question, edge['text']] for edge in edge_batch]
            with torch.no_grad():
                edge_scores = self.cross_encoder_edge_retriever.compute_score(model_input, batch_size=200, cutoff_layers=[40], max_length=256)
            
                for edge, score in zip(edge_batch, edge_scores):
                    table_segment_node_id = edge['table_segment_node_id']
                    passage_id = edge['passage_id']
                    reranking_score = float(score)            
                    self.add_node(retrieved_graphs, 'table segment', table_segment_node_id, passage_id, reranking_score, 'edge_reranking')
                    self.add_node(retrieved_graphs, 'passage', passage_id, table_segment_node_id, reranking_score, 'edge_reranking')
    
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
            edge_score = retrieved_graph_content['edge_score']
            
            # add nodes
            self.add_node(integrated_graph, 'table segment', table_segment_node_id, passage_id, edge_score, retrieval_type)
            self.add_node(integrated_graph, 'passage', passage_id, table_segment_node_id, edge_score, retrieval_type)

            # node scoring
            self.assign_scores(integrated_graph)

        return integrated_graph

    def augment_node(self, graph, nl_question, topk_query_nodes, query_node_type, retrieved_node_type, retrieval_time):
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
                    augment_type = f'passage_node_augmentation_{retrieval_time}'
                else:
                    retrieved_node_id = self.id_to_table_key[str(retrieved_id)]
                    augment_type = f'table_segment_node_augmentation_{retrieval_time}'
                
                self.add_node(graph, query_node_type, query_node_id, retrieved_node_id, query_node_score, augment_type, source_rank, target_rank)
                self.add_node(graph, retrieved_node_type, retrieved_node_id, query_node_id, query_node_score, augment_type, target_rank, source_rank)

    def get_expanded_query(self, nl_question, table, row_id):
        # if query_node_type == 'table segment':
        # table_key = node_id.split('_')[0]
        # table = self.table_key_to_content[table_key]
        table_title = table['title']
        
        # row_id = int(node_id.split('_')[1])
        table_rows = table['text'].split('\n')
        column_name = table_rows[0]
        row_values = table_rows[row_id+1]
        
        expanded_query = f"{nl_question} [SEP] {table_title} [SEP] {column_name} [SEP] {row_values}"
        # else:
        #     passage = self.passage_key_to_content[node_id]
        #     passage_title = passage['title']
        #     passage_text = passage['text']
            
        #     expanded_query = f"{nl_question} [SEP] {passage_title} [SEP] {passage_text}"
        
        return expanded_query

    def add_node(self, graph, source_node_type, source_node_id, target_node_id, score, retrieval_type, source_rank=0, target_rank=0):
        # add source and target node
        if source_node_id not in graph:
            graph[source_node_id] = {'type': source_node_type, 'linked_nodes': [[target_node_id, score, retrieval_type, source_rank, target_rank]]}
        # add target node
        else:
            graph[source_node_id]['linked_nodes'].append([target_node_id, score, retrieval_type, source_rank, target_rank])
    
    def assign_scores(self, graph, retrieval_type = None):
        for node_id, node_info in graph.items():
            if retrieval_type is not None:
                filtered_retrieval_type = ['edge_retrieval', 'passage_node_augmentation_0', 'table_segment_node_augmentation_0']
                linked_scores = [linked_node[1] for linked_node in node_info['linked_nodes'] if linked_node[2] not in filtered_retrieval_type]
            else:
                linked_scores = [linked_node[1] for linked_node in node_info['linked_nodes']]
            
            if self.node_scoring_method == 'min':
                node_score = min(linked_scores)
            elif self.node_scoring_method == 'max':
                node_score = max(linked_scores)
            elif self.node_scoring_method == 'mean':
                node_score = sum(linked_scores) / len(linked_scores)
            
            graph[node_id]['score'] = node_score

@hydra.main(config_path="conf", config_name="graph_query_algorithm")
def main(cfg: DictConfig):
    # load qa dataset
    print(f"Loading qa dataset...")
    ALL_TABLES_PATH = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    with open(ALL_TABLES_PATH) as f:
        all_tables = json.load(f)
    
    table_chunk_to_table = {}
    for table_info in all_tables:
        table_chunk_to_table[table_info['chunk_id']] = table_info
    
    qa_dataset = json.load(open("/home/shpark/OTT_QA_Workspace/data_graph_error_case.json"))
    graph_query_engine = GraphQueryEngine(cfg)
    
    # query
    print(f"Start querying...")
    # Initialize recall arrays
    answer_recall_list = []
    for qidx, qa_datum in tqdm(enumerate(qa_dataset), total=len(qa_dataset)):
        
        nl_question = qa_datum['question']
        answers = qa_datum['answers']
        positive_passages = qa_datum['positive_passages']
        retrieved_node_list = []
        for positive_ctx in qa_datum['positive_ctxs']:
            rows = positive_ctx['rows']
            table_chunk_id = positive_ctx['chunk_id']
            table = table_chunk_to_table[table_chunk_id]
            for answer_node in positive_ctx['answer_node']:
                row_id = rows.index(answer_node[1][0])
                retrieved_nodes = graph_query_engine.query(nl_question, table, row_id, retrieval_time = 2)
                retrieved_node_list.extend(retrieved_nodes)

        if set(positive_passages).intersection(set(retrieved_node_list)) == set():
            answer_recall_list.append(0)
        else:
            answer_recall_list.append(1)
                
        # save integrated graph
        # print(f"Saving integrated graph...")
        # json.dump(retrieved_graphs, open(retrieved_graph_path, 'w'))

    print(f"Start evaluating...")
    print(f"Answer Recall: {sum(answer_recall_list) / len(answer_recall_list)}")

if __name__ == "__main__":
    main()
# 1. dev에서 table segment와 passage가 graph상에 연결되지 않은 경우를 확인
# 2. 해당 경우에 대해 gold table segment와 질의를 합쳐 passage를 검색했을 때 검색되는 경우를 확인
# 3. 검색되는 경우가 전체에서 얼마를 차지하는지 파악