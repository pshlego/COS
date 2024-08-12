import os
import re
import sys
import ast
import copy
import json
import time
import vllm
import hydra
import torch
import unicodedata
from tqdm import tqdm
from pymongo import MongoClient
from omegaconf import DictConfig
from transformers import set_seed
from ColBERT.colbert import Searcher
from ColBERT.colbert.infra import ColBERTConfig
from FlagEmbedding import LayerWiseFlagLLMReranker
from Ours.dpr.data.qa_validation import has_answer
from Ours.dpr.utils.tokenizers import SimpleTokenizer
from prompts import select_table_segment_prompt, select_passage_prompt
# VLLM Parameters
COK_VLLM_TENSOR_PARALLEL_SIZE = 4 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
COK_VLLM_GPU_MEMORY_UTILIZATION = 0.6 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.


ROW_ID_IDX = 1

# Reranking module
CUTOFF_LAYER = 28
EDGE_RERANKING_MAX_LENGTH = 256

MIN_EDGE_SCORE = -10000000

# LLM Module
LLM_GIVEN_EDGE_SCORE = 10000000

#
FINAL_MAX_EDGE_COUNT = 50
TABLE_AND_PASSAGE_TRIM_LENGTH = 96
PASSAGE_TRIM_LENGTH = 128




set_seed(0)


@hydra.main(config_path = "conf", config_name = "graph_query_algorithm_w_table_embedding")
def main(cfg: DictConfig):
    # load qa dataset
    print()
    print(f"[[ Loading qa dataset... ]]", end = "\n\n")
    qa_dataset = json.load(open(cfg.qa_dataset_path))
    graph_query_engine = GraphQueryEngine(cfg)
    
    # query
    print(f"Start querying...")
    for qidx, qa_datum in tqdm(enumerate(qa_dataset), total=len(qa_dataset)):
        # qid = qa_datum['id']
        # if qid not in error_qid_list:
        #     continue
        if qidx < 1378:
            continue
        
        nl_question = qa_datum['question']
        answers = qa_datum['answers']
        
        init_time = time.time()
        retrieved_graph = graph_query_engine.query(nl_question, qa_datum, retrieval_time = 2)
        end_time = time.time()
        qury_time = end_time - init_time
        
        to_print = {
            "qa data": qa_datum,
            "retrieved graph": retrieved_graph
        }
        
        with open(cfg.final_result_path, 'a+') as file:
            file.write(json.dumps(to_print) + '\n')
        
        with open(cfg.query_time_save_path, 'a+') as file:
            file.write(json.dumps(qury_time) + '\n')

    # save integrated graph
    print(f"Saving integrated graph...")
    # json.dump(retrieved_query_list, open(cfg.integrated_graph_save_path, 'w'))
    # json.dump(query_time_list, open(cfg.query_time_save_path, 'w'))
    # json.dump(error_cases, open(cfg.error_cases_save_path, 'w'))




class GraphQueryEngine:
    
    def __init__(self, cfg):

        # 1. Load data graph 1:
            # Edge id to node pair
        print(f"1. Loading edge contents...")
        edge_contents = []
        EDGES_NUM = 17151500
        with open(cfg.edge_dataset_path, "r") as file:
            for line in tqdm(file, total = EDGES_NUM):
                edge_contents.append(json.loads(line))
        num_of_edges = len(edge_contents)
        print("1. Loaded " + str(num_of_edges) + " edges!")
        self.edge_key_to_content = {}
        self.table_key_to_edge_keys = {}
        print("1. Processing edges...")
        for id, edge_content in tqdm(enumerate(edge_contents), total = len(edge_contents)):
            self.edge_key_to_content[edge_content['chunk_id']] = edge_content
            
            if str(edge_content['table_id']) not in self.table_key_to_edge_keys:
                self.table_key_to_edge_keys[str(edge_content['table_id'])] = []
            
            self.table_key_to_edge_keys[str(edge_content['table_id'])].append(id) 
        print("1. Processing edges complete", end = "\n\n")
        
        # 2. Load data graph 2:
            # Table id to linked passages
        SIZE_OF_DATA_GRAPH = 839810
        print("2. Loading data graph...")
        entity_linking_results = []
        with open(cfg.entity_linking_dataset_path, "r") as file:
            for line in tqdm(file, total = SIZE_OF_DATA_GRAPH):
                entity_linking_results.append(json.loads(line))
        num_of_entity_linking_results = len(entity_linking_results)
        print("2. Loaded " + str(num_of_entity_linking_results) + " tables!")
        self.table_chunk_id_to_linked_passages = {}
        print("2. Processing data graph...")
        for entity_linking_content in tqdm(entity_linking_results):
            self.table_chunk_id_to_linked_passages[entity_linking_content['table_chunk_id']] = entity_linking_content
        print("2. Processing data graph complete!", end = "\n\n")
        
        # 3. Load tables
        print("3. Loading tables...")
        self.table_key_to_content = {}
        self.table_title_to_table_key = {}
        table_contents = json.load(open(cfg.table_data_path))
        print("3. Loaded " + str(len(table_contents)) + " tables!")
        print("3. Processing tables...")
        for table_key, table_content in tqdm(enumerate(table_contents)):
            self.table_key_to_content[str(table_key)] = table_content
            self.table_title_to_table_key[table_content['chunk_id']] = table_key
        print("3. Processing tables complete!", end = "\n\n")
        
        # 4. Load passages
        print("4. Loading passages...")
        self.passage_key_to_content = {}
        passage_contents = json.load(open(cfg.passage_data_path))
        print("4. Loaded " + str(len(passage_contents)) + " passages!")
        print("4. Processing passages...")
        for passage_content in tqdm(passage_contents):
            self.passage_key_to_content[passage_content['title']] = passage_content
        print("4. Processing passages complete!", end = "\n\n")

        # 5. Load Retrievers
            ## id mappings
        print("5. Loading retrievers...")
        self.id_to_edge_key = json.load(open(cfg.edge_ids_path))
        self.id_to_table_key = json.load(open(cfg.table_ids_path))
        self.id_to_passage_key = json.load(open(cfg.passage_ids_path))
        print("5. Loaded retrievers complete!", end = "\n\n")
        
        # 6. Load ColBERT retrievers
        print("6. Loading index...")
        print("6 (1/4). Loading ColBERT edge retriever...")
        disablePrint()
        self.colbert_edge_retriever = Searcher(index=f"{cfg.edge_index_name}.nbits{cfg.nbits}", config=ColBERTConfig(), collection=cfg.collection_edge_path, index_root=cfg.edge_index_root_path, checkpoint=cfg.edge_checkpoint_path)
        enablePrint()
        print("6 (1/4). Loaded index complete!")
        print("6 (2/4). Loading ColBERT table retriever...")
        disablePrint()
        self.colbert_table_retriever = Searcher(index=f"{cfg.table_index_name}.nbits{cfg.nbits}", config=ColBERTConfig(), collection=cfg.collection_table_path, index_root=cfg.table_index_root_path, checkpoint=cfg.table_checkpoint_path)
        enablePrint()
        print("6 (2/4). Loaded ColBERT table retriever complete!")
        print("6 (3/4). Loading ColBERT passage retriever...")
        disablePrint()
        self.colbert_passage_retriever = Searcher(index=f"{cfg.passage_index_name}.nbits{cfg.nbits}", config=ColBERTConfig(), collection=cfg.collection_passage_path, index_root=cfg.passage_index_root_path, checkpoint=cfg.passage_checkpoint_path)
        enablePrint()
        print("6 (3/4). Loaded ColBERT passage retriever complete!")
        print("6 (4/4). Loading reranker...")
        self.cross_encoder_edge_retriever = LayerWiseFlagLLMReranker(cfg.reranker_checkpoint_path, use_fp16=True, device='cuda:1')
        print("6 (4/4). Loading reranker complete!")
        print("6. Loaded index complete!", end = "\n\n")
        
        # 7. Load LLM
        print("7. Loading large language model...")
        self.llm = vllm.LLM(
            cfg.llm_checkpoint_path,
            worker_use_ray = True,
            tensor_parallel_size = COK_VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization = COK_VLLM_GPU_MEMORY_UTILIZATION, 
            trust_remote_code = True,
            dtype = "half", # note: bfloat16 is not supported on nvidia-T4 GPUs
            max_model_len = 6400, # input length + output length
            enforce_eager = True,
        )
        print("7. Loaded large language model complete!", end = "\n\n")
        
        # 8. Load tokenizer
        print("8. Loading tokenizer...")
        self.tokenizer = self.llm.get_tokenizer()
        print("8. Loaded tokenizer complete!", end = "\n\n")
        
        ## prompt templates
        self.select_table_segment_prompt = select_table_segment_prompt
        self.select_passage_prompt = select_passage_prompt
        
        # load experimental settings
        self.top_k_of_edge = cfg.top_k_of_edge
        self.top_k_of_table_segment_augmentation = cfg.top_k_of_table_segment_augmentation
        self.top_k_of_passage_augmentation = cfg.top_k_of_passage_augmentation
        self.top_k_of_table_segment = cfg.top_k_of_table_segment
        self.top_k_of_passage = cfg.top_k_of_passage
        self.top_k_of_table_select_w_llm = cfg.top_k_of_table_select_w_llm
        self.top_k_of_table_segment_select_w_llm = cfg.top_k_of_table_segment_select_w_llm
        self.top_k_of_entity_linking = cfg.top_k_of_entity_linking
        
        self.node_scoring_method = cfg.node_scoring_method
        self.batch_size = cfg.edge_reranker_batch_size
        self.missing_edge_prediction_type = cfg.missing_edge_prediction_type















    #############################################################################
    # Query                                                                     #
    # Input: NL Question                                                        #
    # Input: retrieval_time (Number of iterations)                              #
    # Output: Retrieved Graphs (Subgraph of data graph relevant to the NL       #
    #                           question)                                       #  
    # ------------------------------------------------------------------------- # 
    #############################################################################
    @profile
    def query(self, nl_question, qa_datum, retrieval_time = 2):
        
        # 1. Edge Retrieval
        retrieved_edges = self.retrieve_edges(nl_question)
        
        # 2. Graph Integration
        integrated_graph = self.integrate_graphs(retrieved_edges)
        retrieval_type = None
        
        for i in range(retrieval_time):
            
            # From second iteration
            if i >= 1:
                self.reranking_edges(nl_question, integrated_graph)
                retrieval_type = 'edge_reranking'
                self.assign_scores(integrated_graph, retrieval_type)
            
            # Is not last iteration
            if i < retrieval_time:
                topk_table_segment_nodes = []
                topk_passage_nodes = []
                for node_id, node_info in integrated_graph.items():
                    if node_info['type'] == 'table segment':
                        topk_table_segment_nodes.append([node_id, node_info['score']])
                    elif node_info['type'] == 'passage':
                        topk_passage_nodes.append([node_id, node_info['score']])

                    # Default: 10
                topk_table_segment_nodes = sorted(topk_table_segment_nodes, key=lambda x: x[1], reverse=True)[:self.top_k_of_table_segment_augmentation[i]]
                    # Default: 0
                topk_passage_nodes = sorted(topk_passage_nodes, key=lambda x: x[1], reverse=True)[:self.top_k_of_passage_augmentation]
                
                
                # Expanded Query Retrieval
                if self.missing_edge_prediction_type == 'entity_linking':
                    self.entity_linking(integrated_graph, nl_question, topk_table_segment_nodes, 'table segment', 'passage', i)
                elif self.missing_edge_prediction_type == 'both':
                    self.missing_link_prediction_w_both(integrated_graph, nl_question, topk_table_segment_nodes, 'table segment', 'passage', i)
                else:
                    # 3.1 Passage Node Augmentation
                    self.augment_node(integrated_graph, nl_question, topk_table_segment_nodes,  'table segment', 'passage', i)
                    # 3.2 Table Segment Node Augmentation
                    self.augment_node(integrated_graph, nl_question, topk_passage_nodes,        'passage', 'table segment', i)
                
                self.assign_scores(integrated_graph, retrieval_type)

            # Is last iteration
            if i == retrieval_time - 1:
                table_segment_id_to_augmented_nodes, table_id_to_augmented_nodes = self.get_table(integrated_graph)

                selected_table_segment_list = self.get_linked_passages(table_segment_id_to_augmented_nodes)
                
                if table_id_to_augmented_nodes != {}:
                    table_id_to_row_id_to_linked_passage_ids, table_id_to_table_info \
                                                        = self.combine_linked_passages(table_id_to_augmented_nodes)
                    
                    selected_table_segment_list_from_llm = self.select_table_segments(
                                                                nl_question, 
                                                                table_id_to_row_id_to_linked_passage_ids,
                                                                table_id_to_table_info
                                                            )
                    selected_table_segment_list.extend(selected_table_segment_list_from_llm)
                
                self.select_passages(nl_question, selected_table_segment_list, integrated_graph)
            
            self.assign_scores(integrated_graph, retrieval_type)
            retrieved_graphs = integrated_graph
        
        return retrieved_graphs
    
    
    
    
    
    
    
    
    
    
    
    
    #############################################################################
    # RetrieveEdges                                                             #
    # Input: NL Question                                                        #
    # Output: Retrieved Edges                                                   #
    # ------------------------------------------------------------------------- #
    # Retrieve `top_k_of_edge` number of edges (table segment - passage)        #
    # relevant to the input NL question.                                        #
    #############################################################################
    @profile
    def retrieve_edges(self, nl_question):
        
        retrieved_edges_info = self.colbert_edge_retriever.search(nl_question, 10000)
        
        retrieved_edge_id_list = retrieved_edges_info[0]
        retrieved_edge_score_list = retrieved_edges_info[2]
        
        retrieved_edge_contents = []
        for graphidx, retrieved_id in enumerate(retrieved_edge_id_list):
            retrieved_edge_content = self.edge_key_to_content[self.id_to_edge_key[str(retrieved_id)]]

            # pass single node graph
            if 'linked_entity_id' not in retrieved_edge_content and 'table_segment_node_list' not in retrieved_edge_content:
                continue
            
            retrieved_edge_content['edge_score'] = retrieved_edge_score_list[graphidx]
            retrieved_edge_contents.append(retrieved_edge_content)
            
            if len(retrieved_edge_contents) == self.top_k_of_edge:
                break

        return retrieved_edge_contents

    #############################################################################
    # IntegrateGraphs                                                           #
    # Input: retrieved_graphs                                                   #
    # Output: integrated_graph with scores assigned to each node                #
    # ------------------------------------------------------------------------- #
    # Integrate `retrieved_graphs`(list of edges relevant to the NL question)   #
    # into `self.graph`, a dictionary containing the information about the      #
    # nodes, via addNode function.                                              #
    # The nodes are assigned scores via `assign_scores` function.               #
    #############################################################################
    @profile
    def integrate_graphs(self, retrieved_graphs):
        integrated_graph = {}
        retrieval_type = 'edge_retrieval'
        # graph integration
        for retrieved_graph_content in retrieved_graphs:
            # get passage node info
            if 'linked_entity_id' in retrieved_graph_content:
                graph_chunk_id = retrieved_graph_content['chunk_id']

                # get table segment node info
                table_key = str(retrieved_graph_content['table_id'])
                row_id = graph_chunk_id.split('_')[1]
                table_segment_node_id = f"{table_key}_{row_id}"
                passage_id = retrieved_graph_content['linked_entity_id']
                
                # get two node graph score
                edge_score = retrieved_graph_content['edge_score']
                
                # add nodes
                self.add_node(integrated_graph, 'table segment', table_segment_node_id, passage_id, edge_score, retrieval_type)
                self.add_node(integrated_graph, 'passage', passage_id, table_segment_node_id, edge_score, retrieval_type)
            elif 'table_segment_node_list' in retrieved_graph_content:
                table_key = f"{retrieved_graph_content['table_id']}"
                table_node_id = table_key
                table_segment_node_list = retrieved_graph_content['table_segment_node_list']
                for table_segment_node_id in table_segment_node_list:
                    row_id = table_segment_node_id.split('_')[1]
                    edge_score = retrieved_graph_content['edge_score']
                    self.add_node(integrated_graph, 'table', table_node_id, table_segment_node_id, edge_score, "table_to_table_segment")
                    self.add_node(integrated_graph, 'table segment', table_segment_node_id, table_node_id, edge_score, "table_to_table_segment")

            # node scoring
            self.assign_scores(integrated_graph)

        return integrated_graph



    #############################################################################
    # RerankingEdges                                                            #
    # Input: NL Question                                                        #
    # InOut: Retrieved Graphs - actually just any `graph`                       #
    # ------------------------------------------------------------------------- #
    # Recalculate scores for edges in the `retrieved_graphs` using the cross    #
    # encoder edge reranker.                                                    #
    # The scores are assigned via `addNode` function.                           #
    #############################################################################
    # TODO: change after on (add_node -> change score)
    @profile
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
                    
                    if linked_node[2] == "table_to_table_segment":
                        linked_node_id = linked_node[0]
                        edge_id = linked_node_id
                        
                        if edge_id not in edges_set:
                            edges_set.add(edge_id)
                        else:
                            continue

                        table_content = self.table_key_to_content[str(edge_id)]
                        graph_text = table_content['title'] + ' [SEP] ' + table_content['text']
                        edges.append({'table_id': edge_id, 'table_segment_node_id':node_id, 'text': graph_text.replace('\n','')})
                        
                    else:
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
            edge_batch = edges[i : i + self.batch_size]
            model_input = [[nl_question, edge['text']] for edge in edge_batch]
            with torch.no_grad():
                
                
                edge_scores = self.cross_encoder_edge_retriever.compute_score(model_input, batch_size = self.batch_size, cutoff_layers = [CUTOFF_LAYER], max_length = EDGE_RERANKING_MAX_LENGTH)
                
                if len(model_input) == 1:
                    edge_scores = [edge_scores]
                
                for edge, score in zip(edge_batch, edge_scores):
                    if 'table_id' in edge:
                        table_id = edge['table_id']
                        table_segment_node_id_list = [linked_node[0] for linked_node in retrieved_graphs[table_id]['linked_nodes'] if linked_node[2] == 'table_to_table_segment']
                        for table_segment_node_id in table_segment_node_id_list:
                            reranking_score = float(score)
                            self.add_node(retrieved_graphs, 'table', table_id, table_segment_node_id, reranking_score, 'table_to_table_segment_reranking')
                            self.add_node(retrieved_graphs, 'table segment', table_segment_node_id, table_id, reranking_score, 'table_to_table_segment_reranking')
                    else:
                        table_segment_node_id = edge['table_segment_node_id']
                        passage_id = edge['passage_id']
                        reranking_score = float(score)            
                        self.add_node(retrieved_graphs, 'table segment', table_segment_node_id, passage_id, reranking_score, 'edge_reranking')
                        self.add_node(retrieved_graphs, 'passage', passage_id, table_segment_node_id, reranking_score, 'edge_reranking')
    
    
    
    #############################################################################
    # AssignScores                                                              #
    # InOut: graph                                                              #
    # Input: retrieval_type (default = None)                                    #
    # ------------------------------------------------------------------------- #
    # Assign scores to the nodes in the graph using weights of the edges        #
    # connected to each node                                                    #
    #  - min: minimum score of the linked nodes                                 #
    #  - max: maximum score of the linked nodes                                 #
    #  - mean: mean score of the linked nodes                                   #
    # The score is assigned in graph[node_id]["score"]                          #
    #############################################################################
    @profile
    def assign_scores(self, graph, retrieval_type = None):
        # Filter linked scores based on retrieval_type if provided
        for node_id, node_info in graph.items():
            if retrieval_type is not None:
                if node_info['type'] == 'table segment':
                    filtered_retrieval_type = ['edge_retrieval', 'passage_node_augmentation_0', 'table_segment_node_augmentation_0', 'node_augmentation_0', 'table_to_table_segment', 'table_to_table_segment_reranking']
                else:
                    filtered_retrieval_type = ['edge_retrieval', 'passage_node_augmentation_0', 'table_segment_node_augmentation_0', 'node_augmentation_0', 'table_to_table_segment']
            
                linked_scores = [linked_node[1] for linked_node in node_info['linked_nodes'] if linked_node[2] not in filtered_retrieval_type]
            else:
                # if node_info['type'] == 'table segment':
                #     filtered_retrieval_type = ['table_to_table_segment']
                # else:
                #     filtered_retrieval_type = []
                
                #linked_scores = [linked_node[1] for linked_node in node_info['linked_nodes'] if linked_node[2] not in filtered_retrieval_type]
                linked_scores = [linked_node[1] for linked_node in node_info['linked_nodes']]

            # Assign scores based on the selected method
            if self.node_scoring_method == 'min':
                node_score = min(linked_scores) if linked_scores else MIN_EDGE_SCORE
            elif self.node_scoring_method == 'max':
                node_score = max(linked_scores) if linked_scores else MIN_EDGE_SCORE
            elif self.node_scoring_method == 'mean':
                node_score = sum(linked_scores) / len(linked_scores) if linked_scores else MIN_EDGE_SCORE
            elif self.node_scoring_method == 'PageRank':
                # Calculate PageRank for each node (simplified example)
                page_rank_scores = self.calculate_pagerank(graph)
                node_score = page_rank_scores.get(node_id, 0)
            
            # Assign the computed score to the node
            graph[node_id]['score'] = node_score

    def calculate_pagerank(self, graph, damping_factor=0.85, max_iterations=10, tol=1.0e-6):
        # Initialize scores
        n = len(graph)
        ranks = {node_id: 1.0 / n for node_id in graph}
        new_ranks = ranks.copy()

        for _ in range(max_iterations):
            for node_id in graph:
                rank_sum = 0
                for linked_node_id, linked_score, augment_type, source_rank, target_rank in graph[node_id]['linked_nodes']:
                    if linked_node_id in ranks:
                        rank_sum += ranks[linked_node_id] / len(graph[linked_node_id]['linked_nodes'])
                new_ranks[node_id] = (1 - damping_factor) / n + damping_factor * rank_sum
            
            # Check for convergence
            if max(abs(new_ranks[node_id] - ranks[node_id]) for node_id in graph) < tol:
                break
            
            ranks = new_ranks.copy()
        
        return ranks
    
    
    
    
    #############################################################################
    # AugmentNode                                                               #
    # InOut: graph                                                              #
    # Input: nl_question                                                        #
    # Input: topk_query_nodes (`k` nodes which show the highest scores)         #
    # Input: query_node_type                                                    #
    # Input: retrieved_node_type                                                #
    # Input: retrieval_time                                                     #
    # ------------------------------------------------------------------------- #
    # Add 20 edges for target nodes which are the top k nodes in the query      #
    #############################################################################
    @profile
    def augment_node(self, graph, nl_question, topk_query_nodes, query_node_type, retrieved_node_type, retrieval_time):
        
        for source_rank, (query_node_id, query_node_score) in enumerate(topk_query_nodes):
            
            expanded_query = self.get_expanded_query(nl_question, query_node_id, query_node_type)
            
            if query_node_type == 'table segment':
                top_k = self.top_k_of_passage[retrieval_time]
                retrieved_node_info = self.colbert_passage_retriever.search(expanded_query, top_k)
                retrieved_id_list = retrieved_node_info[0]
            else:
                top_k = self.top_k_of_table_segment
                retrieved_node_info = self.colbert_table_retriever.search(expanded_query, top_k)
                retrieved_id_list = retrieved_node_info[0]

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
    
    
    def entity_linking(self, graph, nl_question, topk_query_nodes, query_node_type, retrieved_node_type, retrieval_time):
        edge_count_list = []
        edge_total_list = []
        linked_passage_total_id_list = []
        top_k = self.top_k_of_passage[retrieval_time]
        
        for source_rank, (query_node_id, query_node_score) in enumerate(topk_query_nodes):
            table_key = query_node_id.split('_')[0]
            table = self.table_key_to_content[table_key]
            chunk_id = table['chunk_id']
            row_id = int(query_node_id.split('_')[1])
            entity_linking_result = self.table_chunk_id_to_linked_passages[chunk_id]
            linked_passage_list = []
            for mention_info in entity_linking_result['results']:
                row = mention_info['row']
                if str(row_id) == str(row):
                    linked_passage_list.extend(mention_info['retrieved'][1:self.top_k_of_entity_linking])

            linked_passage_list = list(set(linked_passage_list))
            table_title = table['title']
            table_column_names = table['text'].split('\n')[0]
            table_row_values = table['text'].split('\n')[row_id+1]
            table_text = f"{table_title} [SEP] {table_column_names} [SEP] {table_row_values}"
            
            edge_text_list = []
            for linked_passage_id in linked_passage_list:
                passage_content = self.passage_key_to_content[linked_passage_id]
                passage_text = f"{passage_content['title']} [SEP] {passage_content['text']}"
                edge_text = f"{table_text} [SEP] {passage_text}"
                edge_text_list.append(edge_text)
            
            edge_count_list.append(len(edge_text_list))
            edge_total_list.extend(edge_text_list)
            linked_passage_total_id_list.extend(linked_passage_list)
        if retrieval_time == 0:
            edges = self.colbert_edge_retriever.checkpoint.doc_tokenizer.tensorize(edge_total_list)
            queries = self.colbert_edge_retriever.checkpoint.query_tokenizer.tensorize([nl_question])
            encoded_Q = self.colbert_edge_retriever.checkpoint.query(*queries)
            Q_duplicated = encoded_Q.repeat_interleave(len(edge_total_list), dim=0).contiguous()
            encoded_D, encoded_D_mask = self.colbert_edge_retriever.checkpoint.doc(*edges, keep_dims='return_mask')
            pred_scores = self.colbert_edge_retriever.checkpoint.score(Q_duplicated, encoded_D, encoded_D_mask)
        else:
            model_input = [[nl_question, edge] for edge in edge_total_list]
            with torch.no_grad():
                pred_scores = self.cross_encoder_edge_retriever.compute_score(model_input, batch_size = len(model_input), cutoff_layers = [CUTOFF_LAYER], max_length = EDGE_RERANKING_MAX_LENGTH)
                pred_scores = torch.tensor(pred_scores)
                if len(model_input) == 1:
                    pred_scores = [pred_scores]
                    pred_scores = torch.tensor(pred_scores)

        # decompose pred_scores by using edge_count_list
        start_idx = 0
        for source_rank, (query_node_id, query_node_score) in enumerate(topk_query_nodes):
            end_idx = start_idx + edge_count_list[source_rank]
            # and we have to sort the scores in descending order
            sorted_idx = torch.argsort(pred_scores[start_idx:end_idx], descending=True)[:top_k]
            for target_rank, idx in enumerate(sorted_idx):
                retrieved_node_id = linked_passage_total_id_list[start_idx + idx]
                augment_type = f'node_augmentation_{retrieval_time}'
                #get score from pred_scores list()
                query_node_score = float(pred_scores[start_idx + idx])
                self.add_node(graph, query_node_type, query_node_id, retrieved_node_id, query_node_score, augment_type, source_rank, target_rank)
                self.add_node(graph, retrieved_node_type, retrieved_node_id, query_node_id, query_node_score, augment_type, target_rank, source_rank)
            start_idx = end_idx
        

    def missing_link_prediction_w_both(self, graph, nl_question, topk_query_nodes, query_node_type, retrieved_node_type, retrieval_time):
        edge_count_list = []
        edge_total_list = []
        retrieved_passage_total_id_list = []
        top_k = self.top_k_of_passage[retrieval_time]
        
        for source_rank, (query_node_id, query_node_score) in enumerate(topk_query_nodes):
            table_key = query_node_id.split('_')[0]
            table = self.table_key_to_content[table_key]
            chunk_id = table['chunk_id']
            row_id = int(query_node_id.split('_')[1])
            entity_linking_result = self.table_chunk_id_to_linked_passages[chunk_id]
            linked_passage_list = []
            for mention_info in entity_linking_result['results']:
                row = mention_info['row']
                if str(row_id) == str(row):
                    linked_passage_list.extend(mention_info['retrieved'][1:self.top_k_of_entity_linking])

            linked_passage_list = list(set(linked_passage_list))
            
            expanded_query = self.get_expanded_query(nl_question, query_node_id, query_node_type)

            top_k = self.top_k_of_passage[retrieval_time]
            retrieved_node_info = self.colbert_passage_retriever.search(expanded_query, top_k)
            retrieved_id_list = retrieved_node_info[0]
            retrieved_passage_list = []
            for target_rank, retrieved_id in enumerate(retrieved_id_list):
                retrieved_node_id = self.id_to_passage_key[str(retrieved_id)]
                retrieved_passage_list.append(retrieved_node_id)
            
            retrieved_passage_list = retrieved_passage_list + linked_passage_list
            retrieved_passage_list = list(set(retrieved_passage_list))
            
            table_title = table['title']
            table_column_names = table['text'].split('\n')[0]
            table_row_values = table['text'].split('\n')[row_id+1]
            table_text = f"{table_title} [SEP] {table_column_names} [SEP] {table_row_values}"
            
            edge_text_list = []
            for linked_passage_id in retrieved_passage_list:
                passage_content = self.passage_key_to_content[linked_passage_id]
                passage_text = f"{passage_content['title']} [SEP] {passage_content['text']}"
                edge_text = f"{table_text} [SEP] {passage_text}"
                edge_text_list.append(edge_text)
            
            edge_count_list.append(len(edge_text_list))
            edge_total_list.extend(edge_text_list)
            retrieved_passage_total_id_list.extend(retrieved_passage_list)
            
        if retrieval_time == 0:
            edges = self.colbert_edge_retriever.checkpoint.doc_tokenizer.tensorize(edge_total_list)
            queries = self.colbert_edge_retriever.checkpoint.query_tokenizer.tensorize([nl_question])
            encoded_Q = self.colbert_edge_retriever.checkpoint.query(*queries)
            Q_duplicated = encoded_Q.repeat_interleave(len(edge_total_list), dim=0).contiguous()
            encoded_D, encoded_D_mask = self.colbert_edge_retriever.checkpoint.doc(*edges, keep_dims='return_mask')
            pred_scores = self.colbert_edge_retriever.checkpoint.score(Q_duplicated, encoded_D, encoded_D_mask)
        else:
            model_input = [[nl_question, edge] for edge in edge_total_list]
            with torch.no_grad():
                pred_scores = self.cross_encoder_edge_retriever.compute_score(model_input, batch_size = len(model_input), cutoff_layers = [CUTOFF_LAYER], max_length = EDGE_RERANKING_MAX_LENGTH)
                pred_scores = torch.tensor(pred_scores)
                if len(model_input) == 1:
                    pred_scores = [pred_scores]
                    pred_scores = torch.tensor(pred_scores)

        # decompose pred_scores by using edge_count_list
        start_idx = 0
        for source_rank, (query_node_id, query_node_score) in enumerate(topk_query_nodes):
            end_idx = start_idx + edge_count_list[source_rank]
            # and we have to sort the scores in descending order
            sorted_idx = torch.argsort(pred_scores[start_idx:end_idx], descending=True)[:top_k]
            for target_rank, idx in enumerate(sorted_idx):
                retrieved_node_id = retrieved_passage_total_id_list[start_idx + idx]
                augment_type = f'node_augmentation_{retrieval_time}'
                #get score from pred_scores list()
                query_node_score = float(pred_scores[start_idx + idx])
                self.add_node(graph, query_node_type, query_node_id, retrieved_node_id, query_node_score, augment_type, source_rank, target_rank)
                self.add_node(graph, retrieved_node_type, retrieved_node_id, query_node_id, query_node_score, augment_type, target_rank, source_rank)
            start_idx = end_idx

        #retrieved_passage_list = [retrieved_passage_list[i] for i in torch.argsort(pred_scores, descending=True)][:top_k]
        
            

    #############################################################################
    # getExpandedQuery                                                          #
    # Input: nl_question                                                        #
    # Input: node_id                                                            #
    # Input: query_node_type                                                    #
    # Output: expanded_query                                                    #
    # ------------------------------------------------------------------------- #
    # generate an expanded query (string to query) using the information of the #
    # node (either can be table segment or passage).                            #
    #############################################################################
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #############################################################################
    # getTable                                                                  #
    # Input: retrieved_graph                                                    #
    # Output: table_key_list                                                    #
    # Output: node_id_list                                                      #
    # Output: table_key_to_augmented_nodes                                      #
    # ------------------------------------------------------------------------- #
    # Using the retrieved_graph, get the tables according to the scores of the  #
    # table segment-typed nodes.                                                #
    # Sorted in the reveresed order of the scores, aggregate the original table #
    # node of what were the table segment nodes.                                #
    #############################################################################
    def get_table(self, retrieved_graph):
        node_id_list = []
        table_key_list = []
        table_key_to_augmented_nodes = {}
        table_segment_key_list = []
        table_segment_key_to_augmented_nodes = {}
        
        sorted_retrieved_graph = sorted(retrieved_graph.items(), key=lambda x: x[1]['score'], reverse=True)
        
        for node_id, node_info in sorted_retrieved_graph:
            
            if node_info['type'] == 'table segment' and (len(table_segment_key_to_augmented_nodes) + len(table_key_to_augmented_nodes)) < self.top_k_of_table_segment_select_w_llm:
            #if node_info['type'] == 'table segment' and (len(table_segment_key_to_augmented_nodes)) < self.top_k_of_table_select_w_llm:
                table_segment_key_to_augmented_nodes[node_id] = list(set([node_info[0] for node_info in node_info['linked_nodes'] if 'table_to_table_segment' not in node_info[2]]))
            elif node_info['type'] == 'table' and (len(table_segment_key_to_augmented_nodes) + len(table_key_to_augmented_nodes)) < self.top_k_of_table_select_w_llm:
            #elif node_info['type'] == 'table' and (len(table_key_to_augmented_nodes)) < self.top_k_of_table_select_w_llm:
                table_segment_node_list = list(set([node_info[0] for node_info in node_info['linked_nodes']]))
                for table_segment_node_id in table_segment_node_list:
                    linked_nodes = retrieved_graph[table_segment_node_id]['linked_nodes']
                    row_id = table_segment_node_id.split('_')[1]
                    table_key_to_augmented_nodes[node_id] = {}
                    table_key_to_augmented_nodes[node_id][row_id] = list(set([node_info[0] for node_info in linked_nodes if 'table_to_table_segment' not in node_info[2]]))

        return table_segment_key_to_augmented_nodes, table_key_to_augmented_nodes
    

    #############################################################################
    # combineLinkedPassages                                                     #
    # Input: table_id_to_linked_nodes                                           #
    # Output: table_id_to_row_id_to_linked_passage_ids                          #
    # Output: table_id_to_table_info                                            #
    # ------------------------------------------------------------------------- #
    # 
    #############################################################################
    def combine_linked_passages(self, table_id_to_linked_nodes):

        table_id_to_row_id_to_linked_passage_ids = {}
        table_id_to_table_info = {}
        
        for table_id, table_segment_to_linked_nodes in table_id_to_linked_nodes.items():
            if table_id not in table_id_to_row_id_to_linked_passage_ids:
                table_id_to_row_id_to_linked_passage_ids[table_id] = {}
            
            # 1. Calculate table info and put it in `table_id_to_table_info`
            if self.table_key_to_content[table_id]['chunk_id'] not in self.table_chunk_id_to_linked_passages:
                continue
            
            linked_passage_info = self.table_chunk_id_to_linked_passages[self.table_key_to_content[table_id]['chunk_id']]
            table_title = linked_passage_info['question'].split(' [SEP] ')[0]
            table_column_name = linked_passage_info['question'].split(' [SEP] ')[-1].split('\n')[0]
            table_rows = linked_passage_info['question'].split(' [SEP] ')[-1].split('\n')[1:]
            table_rows = [row for row in table_rows if row != ""]
            table_info = {"title": table_title, "column_name": table_column_name, "rows": table_rows}
            
            table_id_to_table_info[table_id] = table_info
            
            # 2. Calculate table row to table linked passages and put it in
                #       `table_id_to_row_id_to_linked_passage_ids`
            linked_passages = linked_passage_info['results']
            for linked_passage_info in linked_passages:
                row_id = str(linked_passage_info['row'])
                
                if row_id not in table_id_to_row_id_to_linked_passage_ids[table_id]:
                    table_id_to_row_id_to_linked_passage_ids[table_id][row_id] = []

                table_id_to_row_id_to_linked_passage_ids[table_id][row_id].append(linked_passage_info['retrieved'][0])
                
            for row_id, linked_nodes in table_segment_to_linked_nodes.items():
                
                if row_id not in table_id_to_row_id_to_linked_passage_ids[table_id]:
                    table_id_to_row_id_to_linked_passage_ids[table_id][row_id] = []
                
                for node in list(set(linked_nodes)):
                    try:    table_id_to_row_id_to_linked_passage_ids[table_id][row_id].append(node)
                    except: continue
            
        return table_id_to_row_id_to_linked_passage_ids, table_id_to_table_info
    
    def get_linked_passages(self, table_segment_id_to_augmented_nodes):
        selected_table_segment_list = []
        table_id_to_row_id_to_linked_passage_ids = {}
        for table_segment_id, augmented_nodes in table_segment_id_to_augmented_nodes.items():
            table_id = table_segment_id.split('_')[0]
            row_id = table_segment_id.split('_')[1]
            if table_id not in table_id_to_row_id_to_linked_passage_ids:
                table_id_to_row_id_to_linked_passage_ids[table_id] = {}
            
            # linked_passage_info = self.table_chunk_id_to_linked_passages[self.table_key_to_content[table_id]['chunk_id']]
            # linked_passages = linked_passage_info['results']
            # for linked_passage_info in linked_passages:
            #     if row_id == str(linked_passage_info['row']):
            #         if row_id not in table_id_to_row_id_to_linked_passage_ids[table_id]:
            #             table_id_to_row_id_to_linked_passage_ids[table_id][row_id] = []
                    
            #         table_id_to_row_id_to_linked_passage_ids[table_id][row_id].append(linked_passage_info['retrieved'][0])
            
            if row_id not in table_id_to_row_id_to_linked_passage_ids[table_id]:
                table_id_to_row_id_to_linked_passage_ids[table_id][row_id] = []
            
            table_id_to_row_id_to_linked_passage_ids[table_id][row_id].extend(augmented_nodes)
            
            selected_table_segment_list.append(
                {
                    "table_segment_node_id": table_segment_id, 
                    "linked_passages": list(set(table_id_to_row_id_to_linked_passage_ids[table_id][row_id]))
                }
            )
        
        return selected_table_segment_list
    
    #############################################################################
    # selectTableSegments                                                       #
    # Input: nl_question                                                        #
    # Input: table_id_to_row_id_to_linked_passage_ids                           #
    # Input: table_id_to_table_info                                             #
    # Output: selected_table_segment_list                                       #
    # ------------------------------------------------------------------------- #
    # For each table, examine its entire table info and linked passages to pick #
    # the most relevant table segments, using the llm.                          #
    #############################################################################
    @profile
    def select_table_segments(self, nl_question, table_id_to_row_id_to_linked_passage_ids, table_id_to_table_info):
        
        prompt_list = []
        table_id_list = []
        
        for table_id in table_id_to_table_info.keys():
                    
            ######################################################################################################
            
            table_and_linked_passages = self.stringify_table_and_linked_passages(
                                                table_id_to_table_info[table_id],
                                                table_id_to_row_id_to_linked_passage_ids[table_id]
                                            )
            contents_for_prompt = {'question': nl_question, 'table_and_linked_passages': table_and_linked_passages}
            prompt = self.get_prompt(contents_for_prompt)
            prompt_list.append(prompt)
            table_id_list.append(table_id)
            
        responses = self.llm.generate(
                                    prompt_list,
                                    vllm.SamplingParams(
                                        n = 1,  # Number of output sequences to return for each prompt.
                                        top_p = 0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                                        temperature = 0.5,  # randomness of the sampling
                                        skip_special_tokens = True,  # Whether to skip special tokens in the output.
                                        max_tokens = 64,  # Maximum number of tokens to generate per output sequence.
                                        logprobs = 1
                                    ),
                                    use_tqdm = False
                                )
        
        selected_table_segment_list = []
        for table_id, response in zip(table_id_list, responses):
            selected_rows = response.outputs[0].text
            if selected_rows is None:
                continue
            
            try:
                selected_rows = ast.literal_eval(selected_rows)
                selected_rows = [string.strip() for string in selected_rows]
            except:
                selected_rows = [selected_rows.strip()]
                
            for selected_row in selected_rows:
                try:
                    row_id = selected_row.split('_')[ROW_ID_IDX]
                    row_id = str(int(row_id) - 1)
                    _ = table_id_to_row_id_to_linked_passage_ids[table_id][row_id]
                except:
                    continue
                
                selected_table_segment_list.append(
                    {
                        "table_segment_node_id": f"{table_id}_{row_id}", 
                        "linked_passages": table_id_to_row_id_to_linked_passage_ids[table_id][row_id]
                    }
                )
            
        return selected_table_segment_list
    
    
    #############################################################################
    # stringifyTableAndLinkedPassages                                           #
    # Input: table_info                                                         #
    # Input: row_id_to_linked_passage_ids                                       #
    # Output: table_and_linked_passages (string)                                #
    # ------------------------------------------------------------------------- #
    # Make the table and its linked passages in a string format.                #
    #############################################################################
    def stringify_table_and_linked_passages(self, table_info, row_id_to_linked_passage_ids):
        # 1. Stringify table metadata
        table_and_linked_passages = ""
        table_and_linked_passages += f"Table Name: {table_info['title']}\n"
        table_and_linked_passages += f"Column Name: {table_info['column_name'].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ')}\n\n"
        
        # 2. Stringify each row + its linked passages
        for row_id, row_content in enumerate(table_info['rows']):
            table_and_linked_passages += f"Row_{row_id + 1}: {row_content.replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ')}\n"
                                                                
            if str(row_id) in row_id_to_linked_passage_ids:
                table_and_linked_passages += f"Passages linked to Row_{row_id + 1}:\n"
                
                for linked_passage_id in list(set(row_id_to_linked_passage_ids[str(row_id)])):
                    
                    linked_passage_content = self.passage_key_to_content[linked_passage_id]['text']
                    
                    tokenized_content = self.tokenizer.encode(linked_passage_content)
                    trimmed_tokenized_content = tokenized_content[ : TABLE_AND_PASSAGE_TRIM_LENGTH]
                    trimmed_content = self.tokenizer.decode(trimmed_tokenized_content)
                    table_and_linked_passages += f"- {trimmed_content}\n"
                    
            table_and_linked_passages += "\n\n"

        return table_and_linked_passages

    
    #############################################################################
    # SelectPassages                                                            #
    # Input: NL Question                                                        #
    # Input: selected_table_segment_list                                        #
    # InOut: integrated_graph                                                   #
    # ------------------------------------------------------------------------- #
    # For each selected table segment, put its content and its linked passages  #
    #   into the prompt, and get the response from the LLM.                     #
    # Parse the response, then update the `integrated_graph` with the selected  #
    #   table segments and its linked passages.                                 #
    # They are marked as `llm_based_selection`, and they are given edge scores  #
    #   of 100,000.                                                             #
    #############################################################################
    @profile
    def select_passages(self, nl_question, selected_table_segment_list, integrated_graph):
        
        prompt_list = []
        table_segment_node_id_list = []
        
        # 1. Generate prompt which contains the table segment and its linked passages
        for selected_table_segment in selected_table_segment_list:
            table_segment_node_id = selected_table_segment['table_segment_node_id']
            linked_passages = selected_table_segment['linked_passages']
            table_key = table_segment_node_id.split('_')[0]
            table = self.table_key_to_content[table_key]
            table_title = table['title']
            row_id = int(table_segment_node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id + 1]
            table_segment_text = column_name + '\n' + row_values
            table_segment_content = {"title": table_title, "content": table_segment_text}
            linked_passage_contents = []
            for linked_passage_id in linked_passages:
                if linked_passage_id not in self.passage_key_to_content:
                    continue
                
                linked_passage_contents.append({"title":linked_passage_id, "content": self.passage_key_to_content[linked_passage_id]['text']})
                
            graph = {"table_segment": table_segment_content, "linked_passages": linked_passage_contents}

            table_segment = graph['table_segment']
            table_segment_content = f"Table Title: {table_segment['title']}" + "\n" + table_segment['content']\
                                                                                                    .replace(' , ', '[special tag]')\
                                                                                                    .replace(', ', ' | ')\
                                                                                                    .replace('[special tag]', ' , ')
            
            linked_passages = graph['linked_passages']
            linked_passage_contents = ""
            for linked_passage in linked_passages:
                title = linked_passage['title']
                content = linked_passage['content']
                tokenized_content = self.tokenizer.encode(content)
                trimmed_tokenized_content = tokenized_content[ : PASSAGE_TRIM_LENGTH]
                trimmed_content = self.tokenizer.decode(trimmed_tokenized_content)
                linked_passage_contents += f"Title: {title}. Content: {trimmed_content}\n\n"

            contents_for_prompt = {"question": nl_question, "table_segment": table_segment_content, "linked_passages": linked_passage_contents}
            prompt = self.get_prompt(contents_for_prompt)
            prompt_list.append(prompt)
            table_segment_node_id_list.append(table_segment_node_id)
        
        # 2. Run LLM
        responses = self.llm.generate(
                                    prompt_list,
                                    vllm.SamplingParams(
                                        n = 1,  # Number of output sequences to return for each prompt.
                                        top_p = 0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                                        temperature = 0.5,  # randomness of the sampling
                                        skip_special_tokens = True,  # Whether to skip special tokens in the output.
                                        max_tokens = 128,  # Maximum number of tokens to generate per output sequence.
                                        logprobs = 1
                                    ),
                                    use_tqdm = False
                                )
        
        # 3. Parse LLM results and add the top 
        for table_segment_node_id, response in zip(table_segment_node_id_list, responses):
            selected_passage_id_list = response.outputs[0].text
            try:    selected_passage_id_list = ast.literal_eval(selected_passage_id_list)
            except: selected_passage_id_list = [selected_passage_id_list]
            
            
            if selected_passage_id_list is None:
                continue
            
            for selected_passage_id in selected_passage_id_list:
                if selected_passage_id not in self.passage_key_to_content: continue
                self.add_node(integrated_graph, 'table segment', table_segment_node_id, selected_passage_id,   LLM_GIVEN_EDGE_SCORE, 'llm_selected')
                self.add_node(integrated_graph, 'passage',       selected_passage_id,   table_segment_node_id, LLM_GIVEN_EDGE_SCORE, 'llm_selected')
    

    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #############################################################################
    # AddNode                                                                   #
    # InOut: graph                                                              #
    # Input: source_node_type                                                   #
    # Input: source_node_id                                                     #
    # Input: score                                                              #
    # Input: retrieval_type                                                     #
    # Input: source_rank (default = 0)                                          #
    # Input: target_rank (default = 0)                                          #
    # ------------------------------------------------------------------------- #
    # `graph` is a dictionary in which its key becomes a source node id and     #
    # its value is a dictionary containing the information about the nodes      #
    # linked to the source node.                                                #
    #   `type`: type of the source node (table segment | passage)               #
    #   `linked_nodes`: a list of linked nodes to the source node. Each linked  #
    #                   node is a list containing                               #
    #                       - target node id                                    #
    #                       - score                                             #
    #                       - retrieval type                                    #
    #                       - source rank                                       #
    #                       - target rank                                       #
    #############################################################################
    def add_node(self, graph, source_node_type, source_node_id, target_node_id, score, retrieval_type, source_rank = 0, target_rank = 0):
        # add source and target node
        if source_node_id not in graph:
            graph[source_node_id] = {'type': source_node_type, 'linked_nodes': [[target_node_id, score, retrieval_type, source_rank, target_rank]]}
        # add target node
        else:
            graph[source_node_id]['linked_nodes'].append([target_node_id, score, retrieval_type, source_rank, target_rank])
    
    #############################################################################
    # GetPrompt                                                                 #
    #############################################################################
    def get_prompt(self, contents_for_prompt):
        if 'linked_passages' in contents_for_prompt:
            prompt = self.select_passage_prompt.format(**contents_for_prompt)
        else:
            prompt = self.select_table_segment_prompt.format(**contents_for_prompt)
        
        return prompt
    
    

    
    
    














# def evaluate(retrieved_graph_list, qa_dataset, graph_query_engine):
def evaluate(retrieved_graph, qa_data, graph_query_engine):
    
    table_key_to_content = graph_query_engine.table_key_to_content
    passage_key_to_content = graph_query_engine.passage_key_to_content
    filtered_retrieval_type = ['edge_reranking', "passage_node_augmentation_1"]
    
    # 1. Revise retrieved graph
    revised_retrieved_graph = {}
    for node_id, node_info in retrieved_graph.items():                  

        if node_info['type'] == 'table segment':
            linked_nodes = [x for x in node_info['linked_nodes'] 
                                if x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation_1' and (x[3] < 10) and (x[4] < 2)) 
                                    or x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[4] < 1) and (x[3] < 1)) 
                                    or x[2] == 'edge_reranking'
                            ]
        elif node_info['type'] == 'passage':
            linked_nodes = [x for x in node_info['linked_nodes'] 
                                if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < 1) and (x[4] < 1)) 
                                or x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation_1' and (x[4] < 10) and (x[3] < 2)) 
                                or x[2] == 'edge_reranking'
                            ]
        if len(linked_nodes) == 0: continue
        
        revised_retrieved_graph[node_id] = copy.deepcopy(node_info)
        revised_retrieved_graph[node_id]['linked_nodes'] = linked_nodes
        
        linked_scores = [linked_node[1] for linked_node in linked_nodes]
        node_score = max(linked_scores)
        revised_retrieved_graph[node_id]['score'] = node_score
    retrieved_graph = revised_retrieved_graph


    # 2. Evaluate with revised retrieved graph
    node_count = 0
    edge_count = 0
    answers = qa_data['answers']
    context = ""
    retrieved_table_set = set()
    retrieved_passage_set = set()
    sorted_retrieved_graph = sorted(retrieved_graph.items(), key = lambda x: x[1]['score'], reverse = True)
    
    for node_id, node_info in sorted_retrieved_graph:
        
        if node_info['type'] == 'table segment':
            
            table_id = node_id.split('_')[0]
            table = table_key_to_content[table_id]
            chunk_id = table['chunk_id']
            node_info['chunk_id'] = chunk_id
            
            if table_id not in retrieved_table_set:
                retrieved_table_set.add(table_id)
                
                if edge_count == FINAL_MAX_EDGE_COUNT:
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
            
            if edge_count == FINAL_MAX_EDGE_COUNT:
                continue
            
            edge_count += 1
            context += edge_text
        
        elif node_info['type'] == 'passage':

            if node_id in retrieved_passage_set:
                continue

            max_linked_node_id, _, _, _, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default = (None, 0, 'edge_reranking', 0, 0))
            table_id = max_linked_node_id.split('_')[0]
            table = table_key_to_content[table_id]
            
            if table_id not in retrieved_table_set:
                retrieved_table_set.add(table_id)
                if edge_count == FINAL_MAX_EDGE_COUNT:
                    continue
                context += table['text']
                edge_count += 1

            row_id = int(max_linked_node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            table_segment_text = column_name + '\n' + row_values
            
            if edge_count == FINAL_MAX_EDGE_COUNT:
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
        recall = 1
        error_analysis = {}
    else:
        recall = 0
        error_analysis = copy.deepcopy(qa_data)
        error_analysis['retrieved_graph'] = retrieved_graph
        error_analysis['sorted_retrieved_graph'] = sorted_retrieved_graph[:node_count]
        if  "hard_negative_ctxs" in error_analysis:
            del error_analysis["hard_negative_ctxs"]

    return recall, error_analysis

def remove_accents_and_non_ascii(text):
    # Normalize the text to decompose combined characters
    normalized_text = unicodedata.normalize('NFD', text)
    # Remove all characters that are not ASCII
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining non-letter characters
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,!.?\-]', '', ascii_text)
    return cleaned_text



def filter_fn(pid, values_to_remove):
    return pid[~torch.isin(pid, values_to_remove)].to("cuda")

def get_context(retrieved_graph, graph_query_engine):
    context = ""
    retrieved_table_set = set()
    retrieved_passage_set = set()
    sorted_retrieved_graph = sorted(retrieved_graph.items(), key=lambda x: x[1]['score'], reverse=True)
    table_key_to_content = graph_query_engine.table_key_to_content
    for node_id, node_info in sorted_retrieved_graph:
        if node_info['type'] == 'table segment':
            table_id = node_id.split('_')[0]
            table = table_key_to_content[table_id]
            
            if table_id not in retrieved_table_set:
                retrieved_table_set.add(table_id)
                context += table['text']
        
            max_linked_node_id, max_score, a, b, c = max(node_info['linked_nodes'], key=lambda x: x[1], default=(None, 0, 'edge_retrieval', 0, 0))

            if max_linked_node_id in retrieved_passage_set:
                continue
            
            retrieved_passage_set.add(max_linked_node_id)
            passage_content = graph_query_engine.passage_key_to_content[max_linked_node_id]
            passage_text = passage_content['title'] + ' ' + passage_content['text']
            
            row_id = int(node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            table_segment_text = column_name + '\n' + row_values
            
            edge_text = table_segment_text + '\n' + passage_text
            context += edge_text
            
        elif node_info['type'] == 'passage':
            if node_id in retrieved_passage_set:
                continue
            
            retrieved_passage_set.add(node_id)
            passage_content = graph_query_engine.passage_key_to_content[node_id]
            passage_text = passage_content['title'] + ' ' + passage_content['text']
            #'Who created the series in which the character of Robert , played by actor Nonso Anozie , appeared ? [SEP] Nonso Anozie [SEP] Year, Title, Role, Notes [SEP] 2007, Prime Suspect 7 : The Final Act, Robert, Episode : Part 1'
            max_linked_node_id, max_score, _, _, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default=(None, 0, 'edge_retrieval', 0, 0))
            table_id = max_linked_node_id.split('_')[0]
            table = table_key_to_content[table_id]
            
            if table_id not in retrieved_table_set:
                retrieved_table_set.add(table_id)
                context += table['text']
                
            row_id = int(max_linked_node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            table_segment_text = column_name + '\n' + row_values
            
            edge_text = table_segment_text + '\n' + passage_text
            context += edge_text

    return context


def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__









if __name__ == "__main__":
    main()