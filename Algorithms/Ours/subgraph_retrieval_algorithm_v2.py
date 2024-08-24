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
import concurrent.futures
from omegaconf import DictConfig
from transformers import set_seed
from ColBERT.colbert import Searcher
from ColBERT.colbert.infra import ColBERTConfig
from FlagEmbedding import LayerWiseFlagLLMReranker
from Ours.dpr.data.qa_validation import has_answer
from Ours.dpr.utils.tokenizers import SimpleTokenizer
from prompts import select_table_segment_prompt, select_passage_prompt

set_seed(0)

@hydra.main(config_path = "conf", config_name = "subgraph_retrieval_algorithm")
def main(cfg: DictConfig):
    # load qa dataset
    print()
    print(f"[[ Loading qa dataset... ]]", end = "\n\n")
    qa_dataset = json.load(open(cfg.qa_dataset_path))
    positive_id_list = json.load(open("/home/shpark/OTT_QA_Workspace/positive_id_list.json"))
    negative_id_list = json.load(open("/home/shpark/OTT_QA_Workspace/negative_id_list.json"))
    graph_query_engine = GraphQueryEngine(cfg)
    # query
    print(f"Start querying...")
    for qidx, qa_datum in tqdm(enumerate(qa_dataset), total=len(qa_dataset)):
        if qa_datum['id'] in negative_id_list:
            print("negative example")
        elif qa_datum['id'] in positive_id_list:
            print("positive example")
        else:
            continue
        # if qidx < 1095:
        #     continue
        nl_question = qa_datum['question']

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
    
    print(f"Querying done.")






class GraphQueryEngine:
    def __init__(self, cfg):
        """_summary_

        Args:
            cfg (DictConfig): Configurations for the graph query engine
        """
        # 1. Load data graph: Table id to linked passages
        print("1. Loading data graph...")
        SIZE_OF_DATA_GRAPH = 839810
        self.table_chunk_id_to_linked_passages = {}
        with open(cfg.entity_linking_dataset_path, "r") as file:
            for line in tqdm(file, total = SIZE_OF_DATA_GRAPH):
                entity_linking_content = json.loads(line)
                self.table_chunk_id_to_linked_passages[entity_linking_content['table_chunk_id']] = entity_linking_content
        print("1. Loaded " + str(SIZE_OF_DATA_GRAPH) + " table chunks!", end = "\n\n")
        
        
        # 2. Load edge contents: Edge id to node pair
        print(f"2. Loading edge and table contents...")
        EDGES_NUM = 17992395
        self.edge_and_table_key_to_content = {}
        with open(cfg.edge_dataset_path, "r") as file:
            for line in tqdm(file, total = EDGES_NUM):
                edge_and_table_content = json.loads(line)
                self.edge_and_table_key_to_content[edge_and_table_content['chunk_id']] = edge_and_table_content
        print("2. Loaded " + str(EDGES_NUM) + " edges!", end = "\n\n")
        
        
        # 3. Load tables
        print("3. Loading tables...")
        self.table_key_to_content = {}
        table_contents = json.load(open(cfg.table_data_path))
        TABLES_NUM = len(table_contents)
        for table_key, table_content in tqdm(enumerate(table_contents), total = TABLES_NUM):
            self.table_key_to_content[str(table_key)] = table_content
        print("3. Loaded " + str(TABLES_NUM) + " tables!", end = "\n\n")
        
        
        # 4. Load passages
        print("4. Loading passages...")
        self.passage_key_to_content = {}
        passage_contents = json.load(open(cfg.passage_data_path))
        PASSAGES_NUM = len(passage_contents)
        for passage_content in tqdm(passage_contents):
            self.passage_key_to_content[passage_content['title']] = passage_content
        print("4. Loaded " + str(PASSAGES_NUM) + " passages!", end = "\n\n")
        
        
        # 5. Load Retrievers
        print("5. Loading retrievers...")
        print("5.1. Loading id mappings...")
        self.id_to_edge_and_table_key = json.load(open(cfg.edge_and_table_ids_path))
        self.id_to_table_key = json.load(open(cfg.table_ids_path))
        self.id_to_passage_key = json.load(open(cfg.passage_ids_path))
        print("5.1. Loaded id mappings!")
        print("5.2. Loading index...")
        print("5.2 (1/3). Loading edge & table index...")
        disablePrint()
        self.colbert_edge_and_table_retriever = Searcher(index=cfg.edge_and_table_index_name, config=ColBERTConfig(), collection=cfg.collection_edge_and_table_path, index_root=cfg.index_root_path, checkpoint=cfg.edge_and_table_checkpoint_path)
        enablePrint()
        print("5.2 (1/3). Loaded edge & table index complete!")
        print("5.2 (2/3). Loading table index...")
        disablePrint()
        self.colbert_table_retriever = Searcher(index=cfg.table_index_name, config=ColBERTConfig(), collection=cfg.collection_table_path, index_root=cfg.index_root_path, checkpoint=cfg.table_checkpoint_path)
        enablePrint()
        print("5.2 (2/3). Loaded table index complete!")
        print("5.2 (3/3). Loading passage index...")
        disablePrint()
        self.colbert_passage_retriever = Searcher(index=cfg.passage_index_name, config=ColBERTConfig(), collection=cfg.collection_passage_path, index_root=cfg.index_root_path, checkpoint=cfg.passage_checkpoint_path)
        enablePrint()
        print("5.2 (3/3). Loaded passage index complete!")
        print("5.2. Loaded index complete!")
        print("5.3. Loading reranker...")
        self.process_num = cfg.process_num
        print(f"5.3. Using {self.process_num} processes for reranking...")
        self.reranker_list = []
        for i in range(self.process_num):
            print(f"5.3. Loading reranker {i}...")
            reranker = LayerWiseFlagLLMReranker(cfg.reranker_checkpoint_path, use_fp16=True, device=f'cuda:{i}')
            print(f"5.3. Loading reranker {i} complete!")
            self.reranker_list.append(reranker)
        self.cutoff_layer = cfg.cutoff_layer
        self.reranking_max_length = cfg.reranking_max_length
        print("5.3. Loading reranker complete!")
        print("5. Loaded retrievers!", end = "\n\n")
        
        
        # 6. Load LLM
        print("6. Loading LLM...")
        self.llm = vllm.LLM(
            cfg.llm_checkpoint_path,
            worker_use_ray = True,
            tensor_parallel_size = cfg.tensor_parallel_size, 
            gpu_memory_utilization = cfg.gpu_memory_utilization, 
            trust_remote_code = True,
            dtype = "half", # note: bfloat16 is not supported on nvidia-T4 GPUs
            max_model_len = cfg.mex_model_length, # input length + output length
            enforce_eager = True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.select_passage_prompt = select_passage_prompt
        self.select_table_segment_prompt = select_table_segment_prompt
        self.table_and_linked_passages_trim_length = cfg.table_and_linked_passages_trim_length
        self.passage_trim_length = cfg.passage_trim_length
        print("6. Loaded large language model complete!", end = "\n\n")
        
        
        # 7. load experimental settings
        ## edges & tables retrieval
        self.top_k_of_edges_and_tables = cfg.top_k_of_edges_and_tables
        
        ## missing edge prediction
        self.node_scoring_method = cfg.node_scoring_method
        self.missing_edge_prediction_type = cfg.missing_edge_prediction_type
        self.top_k_of_missing_edges = cfg.top_k_of_missing_edges
        ### expanded query retrieval
        self.top_k_of_table_segment_query = cfg.top_k_of_table_segment_query
        self.top_k_of_table_segment_query_entity_linking = cfg.top_k_of_table_segment_query_entity_linking
        self.top_k_of_passage_query = cfg.top_k_of_passage_query
        self.top_k_of_table_segment_target = cfg.top_k_of_table_segment_target
        self.top_k_of_passage_target = cfg.top_k_of_passage_target
        ### entity linking
        self.top_k_of_entity_linking = cfg.top_k_of_entity_linking
        
        ## node selection with large language model
        self.top_k_of_table_segment_select_w_llm = cfg.top_k_of_table_segment_select_w_llm
        self.top_k_of_table_select_w_llm = cfg.top_k_of_table_select_w_llm
        self.top_k_of_passage_select_w_llm = cfg.top_k_of_passage_select_w_llm
        
        self.max_edge_score = cfg.max_edge_score
        self.min_edge_score = cfg.min_edge_score


    #@profile
    def query(self, nl_question, qa_datum, retrieval_time = 2):
        """_summary_

        Args:
            nl_question (str): Natural language question
            retrieval_time (int, optional): Number of iterations. Defaults to 2.

        Returns:
            relevant_nodes (dict): Retrieved nodes (Nodes from the subgraph of data graph relevant to the natural language question)
        """
        
        # 1. Reduce Search Space
        self.qa_datum = qa_datum
        reduced_search_space = {}
        for iteration in range(retrieval_time):
            reduced_search_space = self.search_space_reduction(nl_question, reduced_search_space, iteration)    
        
        # 2. Find Relevant Nodes With LLM

        self.find_relevant_nodes(nl_question, reduced_search_space)

        return reduced_search_space


    #@profile
    def search_space_reduction(self, nl_question, integrated_graph, iteration):
        """_summary_

        Args:
            nl_question (str): Natural language question
            integrated_graph (dict): Integrated graph
            iteration (int): Iteration number
        """
        retrieval_type = None
        if iteration == 0:
            # Retrieve edges and tables
            edges_and_tables = self.retrieve_edges_and_tables(nl_question)
            integrated_graph = self.integrate_into_graph(edges_and_tables)
        else:
            # Rerank edges and tables
            self.rerank_edges_and_tables(nl_question, integrated_graph)
            retrieval_type = 'edge_reranking'
            self.assign_scores(integrated_graph, retrieval_type)
        
        # Predict missing edges
        self.predict_missing_edges(nl_question, integrated_graph, iteration)
        self.assign_scores(integrated_graph, retrieval_type)
        
        return integrated_graph

    #@profile
    def retrieve_edges_and_tables(self, nl_question):
        """Retrieve `top_k_of_edges_and_tables` number of edges (table segment - passage) and tables relevant to the input NL question.

        Args:
            nl_question (str): Natural language question

        Returns:
            edges_and_tables (list): Retrieved edges and tables
        """
        
        # Retrieve edges and tables
        retrieved_edges_and_tables_info = self.colbert_edge_and_table_retriever.search(nl_question, 10000)
        retrieved_edges_and_tables_id_list = retrieved_edges_and_tables_info[0]
        retrieved_edges_and_tables_score_list = retrieved_edges_and_tables_info[2]
        
        edges_and_tables = []
        for idx, retrieved_edge_and_table_id in enumerate(retrieved_edges_and_tables_id_list):
            edge_and_table_key = self.id_to_edge_and_table_key[str(retrieved_edge_and_table_id)]
            retrieved_edge_and_table_content = self.edge_and_table_key_to_content[edge_and_table_key]
            
            # pass single node graph
            if 'linked_entity_id' not in retrieved_edge_and_table_content and 'table_segment_node_list' not in retrieved_edge_and_table_content:
                continue
            
            retrieved_edge_and_table_content['score'] = retrieved_edges_and_tables_score_list[idx]
            edges_and_tables.append(retrieved_edge_and_table_content)
            
            if len(edges_and_tables) == self.top_k_of_edges_and_tables:
                break
        
        return edges_and_tables
    
    def integrate_into_graph(self, edges_and_tables):
        """Integrate `edges_and_tables`(list of edges and tables relevant to the NL question) 
        into `integrated_graph`, a dictionary containing the information about the nodes, 
        via add_node function. The nodes are assigned scores via `assign_scores` function.

        Args:
            edges_and_tables (list): edges and tables

        Returns:
            integrated_graph (dict): integrated graph
        """
        integrated_graph = {}
        
        # graph integration
        for element in edges_and_tables:
            # get passage node info
            if 'linked_entity_id' in element:
                # get edge info
                table_key = str(element['table_id'])
                row_id = element['chunk_id'].split('_')[1]
                table_segment_node_id = f"{table_key}_{row_id}"
                passage_id = element['linked_entity_id']
                
                # get edge score
                edge_score = element['score']
                
                # add nodes
                self.add_node(integrated_graph, 'table segment', table_segment_node_id, passage_id, edge_score, 'edge_and_table_retrieval')
                self.add_node(integrated_graph, 'passage', passage_id, table_segment_node_id, edge_score, 'edge_and_table_retrieval')

            elif 'table_segment_node_list' in element:
                # get table info
                table_key = f"{element['table_id']}"
                table_node_id = table_key
                table_segment_node_list = element['table_segment_node_list']
                
                # get table score
                table_score = element['score']
                for table_segment_node_id in table_segment_node_list:
                    # add nodes
                    self.add_node(integrated_graph, 'table', table_node_id, table_segment_node_id, table_score, "table_to_table_segment")
                    self.add_node(integrated_graph, 'table segment', table_segment_node_id, table_node_id, table_score, "table_to_table_segment")

        # node scoring
        self.assign_scores(integrated_graph)

        return integrated_graph
    
    #@profile
    def rerank_edges_and_tables(self, nl_question, integrated_graph):
        """Recalculate scores for edges in the `retrieved_graphs` using the cross
        encoder edge reranker. The scores are assigned via `add_node` function.

        Args:
            nl_question (str): Natural language question
            integrated_graph (dict): Integrated graph
        """
        torch.cuda.empty_cache()
        edges_and_tables = []
        edges_and_tables_key_set = set()
        
        for node_id, node_info in integrated_graph.items():
            
            if node_info['type'] == 'table segment':
                table_id = node_id.split('_')[0]
                row_id = int(node_id.split('_')[1])
                table_content = self.table_key_to_content[table_id]
                table_title = table_content['title']
                table_rows = table_content['text'].split('\n')
                column_names = table_rows[0]
                row_values = table_rows[row_id+1]
                table_text = table_title + ' [SEP] ' + column_names + ' [SEP] ' + row_values

                for linked_node in node_info['linked_nodes']:
                    
                    if linked_node[2] != "table_to_table_segment":
                        linked_node_id = linked_node[0]
                        edge_key = f"{node_id}_{linked_node_id}"
                        
                        if edge_key not in edges_and_tables_key_set:
                            edges_and_tables_key_set.add(edge_key)
                        else:
                            continue
                        
                        passage_text = self.passage_key_to_content[linked_node_id]['text']
                        edge_text = table_text + ' [SEP] ' + passage_text
                        edges_and_tables.append({'table_segment_id': node_id, 'passage_id': linked_node_id, 'text': edge_text})
                        
            elif node_info['type'] == 'table':
                table_id = node_id
                table_content = self.table_key_to_content[table_id]
                table_title = table_content['title']
                table_rows = table_content['text'].split('\n')
                column_names = table_rows[0]
                row_values = table_rows[1:]
                table_text = table_title + ' [SEP] ' + column_names + ' [SEP] ' + '\n'.join(row_values)
                
                if table_id not in edges_and_tables_key_set:
                    edges_and_tables_key_set.add(table_id)
                else:
                    continue
                    
                edges_and_tables.append({'table_id': table_id, 'text': table_text})

        model_input = [[nl_question, element['text']] for element in edges_and_tables]
        devided_model_input = []
        for rank in range(self.process_num):
            devided_model_input.append(model_input[rank*len(model_input)//self.process_num:(rank+1)*len(model_input)//self.process_num])
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_list = []
            reranking_scores = []
            
            for rank in range(self.process_num):
                future = executor.submit(self.rerank_edges_and_tables_worker, (self.reranker_list[rank], devided_model_input[rank]))
                future_list.append(future)
            
            concurrent.futures.wait(future_list)
            for future in future_list:
                reranking_scores.extend(future.result())

        for element, reranking_score in zip(edges_and_tables, reranking_scores):
            if 'table_id' in element:
                table_id = element['table_id']
                table_segment_id_list = [linked_node[0] for linked_node in integrated_graph[table_id]['linked_nodes'] if linked_node[2] == 'table_to_table_segment']
                for table_segment_id in table_segment_id_list:
                    reranking_score = float(reranking_score)
                    self.add_node(integrated_graph, 'table', table_id, table_segment_id, reranking_score, 'table_reranking')
                    self.add_node(integrated_graph, 'table_segment', table_segment_id, table_id, reranking_score, 'table_reranking')

            else:
                table_segment_id = element['table_segment_id']
                passage_id = element['passage_id']
                reranking_score = float(reranking_score)            
                self.add_node(integrated_graph, 'table segment', table_segment_id, passage_id, reranking_score, 'edge_reranking')
                self.add_node(integrated_graph, 'passage', passage_id, table_segment_id, reranking_score, 'edge_reranking')
                    
    def rerank_edges_and_tables_worker(self, reranker_and_model_input):
        reranker = reranker_and_model_input[0]
        model_input = reranker_and_model_input[1]
        if len(model_input) == 0:
            return []
        
        reranking_scores = reranker.compute_score(model_input, batch_size=60, cutoff_layers=[self.cutoff_layer], max_length=self.reranking_max_length)
        
        if len(model_input) == 1:
            reranking_scores = [reranking_scores]
        
        return reranking_scores


    #@profile
    def predict_missing_edges(self, nl_question, integrated_graph, iteration):
        """_summary_

        Args:
            nl_question (str): Natural language question
            integrated_graph (dict): Integrated graph
            iteration (int): Iteration number
        """
        
        table_segment_query_nodes = []
        passage_query_nodes = []
        for node_id, node_info in integrated_graph.items():
            if node_info['type'] == 'table segment':
                table_segment_query_nodes.append([node_id, node_info['score']])
            elif node_info['type'] == 'passage':
                passage_query_nodes.append([node_id, node_info['score']])
        
        sorted_table_segment_query_nodes = sorted(table_segment_query_nodes, key=lambda x: x[1], reverse=True)[:self.top_k_of_table_segment_query[iteration]]
        sorted_passage_query_nodes = sorted(passage_query_nodes, key=lambda x: x[1], reverse=True)[:self.top_k_of_passage_query[iteration]]
        
        if self.missing_edge_prediction_type == 'entity_linking':
            self.entity_linking(integrated_graph, nl_question, sorted_table_segment_query_nodes, 'table segment', 'passage', iteration)
        elif self.missing_edge_prediction_type == 'expanded_query_retrieval':
            self.expanded_query_retrieval(integrated_graph, nl_question, sorted_table_segment_query_nodes,  'table segment', 'passage', iteration)
        else:
            self.hybrid_missing_edge_prediction(integrated_graph, nl_question, sorted_table_segment_query_nodes,  'table segment', 'passage', iteration)
    
    
    #@profile
    def entity_linking(self, graph, nl_question, topk_query_nodes, query_node_type, target_node_type, iteration):
        edge_count_list = []
        edge_total_list = []
        linked_passage_total_id_list = []
        # top_k = self.top_k_of_missing_edges[iteration]
        
        for source_rank, (query_node_id, query_node_score) in enumerate(topk_query_nodes):
            table_key = query_node_id.split('_')[0]
            table_content = self.table_key_to_content[table_key]
            chunk_id = table_content['chunk_id']
            row_id = int(query_node_id.split('_')[1])
            entity_linking_result = self.table_chunk_id_to_linked_passages[chunk_id]
            linked_passage_list = []
            for mention_info in entity_linking_result['results']:
                row = mention_info['row']
                if str(row_id) == str(row):
                    linked_passage_list.extend(mention_info['retrieved'][1:self.top_k_of_entity_linking[iteration]])
            
            linked_passage_list = list(set(linked_passage_list))
            
            table_title = table_content['title']
            table_column_names = table_content['text'].split('\n')[0]
            table_row_values = table_content['text'].split('\n')[row_id+1]
            table_text = f"{table_title} [SEP] {table_column_names} [SEP] {table_row_values}"
            
            edge_text_list = []
            for linked_passage_id in linked_passage_list:
                passage_text = self.passage_key_to_content[linked_passage_id]['text']
                edge_text = f"{table_text} [SEP] {passage_text}"
                edge_text_list.append(edge_text)
                
            edge_count_list.append(len(edge_text_list))
            edge_total_list.extend(edge_text_list)
            linked_passage_total_id_list.extend(linked_passage_list)
            
        if iteration == 0:
            edges = self.colbert_edge_and_table_retriever.checkpoint.doc_tokenizer.tensorize(edge_total_list)
            queries = self.colbert_edge_and_table_retriever.checkpoint.query_tokenizer.tensorize([nl_question])
            encoded_Q = self.colbert_edge_and_table_retriever.checkpoint.query(*queries)
            Q_duplicated = encoded_Q.repeat_interleave(len(edge_total_list), dim=0).contiguous()
            encoded_D, encoded_D_mask = self.colbert_edge_and_table_retriever.checkpoint.doc(*edges, keep_dims='return_mask')
            pred_scores = self.colbert_edge_and_table_retriever.checkpoint.score(Q_duplicated, encoded_D, encoded_D_mask)
        else:
            # predict scores with reranker
            model_input = [[nl_question, edge] for edge in edge_total_list]
            devided_model_input = []

            for rank in range(self.process_num):
                devided_model_input.append(model_input[rank*len(model_input)//self.process_num:(rank+1)*len(model_input)//self.process_num])
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_list = []
                reranking_scores = []
                for rank in range(self.process_num):
                    future = executor.submit(self.rerank_edges_and_tables_worker, (self.reranker_list[rank], devided_model_input[rank]))
                    future_list.append(future)
                
                concurrent.futures.wait(future_list)
                for future in future_list:
                    reranking_score = future.result()
                    if len(model_input) == 1:
                        reranking_score = [reranking_score]
                    reranking_scores.extend(reranking_score)
                
            pred_scores = torch.tensor(reranking_scores)

        # decompose pred_scores by using edge_count_list
        start_idx = 0
        for source_rank, (query_node_id, query_node_score) in enumerate(topk_query_nodes):
            end_idx = start_idx + edge_count_list[source_rank]
            # and we have to sort the scores in descending order
            sorted_idx = torch.argsort(pred_scores[start_idx:end_idx], descending=True)#[:top_k]
            for target_rank, idx in enumerate(sorted_idx):
                target_node_id = linked_passage_total_id_list[start_idx + idx]
                augment_type = f'node_augmentation_{iteration}'
                #get score from pred_scores list()
                query_node_score = float(pred_scores[start_idx + idx])
                self.add_node(graph, query_node_type, query_node_id, target_node_id, query_node_score, augment_type, source_rank, target_rank)
                self.add_node(graph, target_node_type, target_node_id, query_node_id, query_node_score, augment_type, target_rank, source_rank)

            start_idx = end_idx
            
    #@profile
    def expanded_query_retrieval(self, graph, nl_question, topk_query_nodes, query_node_type, target_node_type, iteration):
        edge_count_list = []
        edge_total_list = []
        retrieved_passage_total_id_list = []
        # top_k = self.top_k_of_missing_edges[iteration]
        
        for source_rank, (query_node_id, query_node_score) in enumerate(topk_query_nodes):
            
            expanded_query = self.get_expanded_query(nl_question, query_node_id, query_node_type)
            
            if query_node_type == 'table segment':
                table_key = query_node_id.split('_')[0]
                row_id = int(query_node_id.split('_')[1])
                table = self.table_key_to_content[table_key]
                table_title = table['title']
                table_column_names = table['text'].split('\n')[0]
                table_row_values = table['text'].split('\n')[row_id+1]
                query_node_text = f"{table_title} [SEP] {table_column_names} [SEP] {table_row_values}"
                #top_k = self.top_k_of_passage_target[iteration]
                retrieved_node_info = self.colbert_passage_retriever.search(expanded_query, self.top_k_of_passage_target[iteration])
                retrieved_id_list = retrieved_node_info[0]
            else:
                passage = self.passage_key_to_content[query_node_id]
                query_node_text = f"{passage['title']} [SEP] {passage['text']}"
                #top_k = self.top_k_of_table_segment_target[iteration]
                retrieved_node_info = self.colbert_table_retriever.search(expanded_query, self.top_k_of_table_segment_target[iteration])
                retrieved_id_list = retrieved_node_info[0]

            retrieved_node_id_list = []
            for target_rank, retrieved_id in enumerate(retrieved_id_list):
                if query_node_type == 'table segment':
                    retrieved_node_id = self.id_to_passage_key[str(retrieved_id)]
                else:
                    retrieved_node_id = self.id_to_table_key[str(retrieved_id)]
                retrieved_node_id_list.append(retrieved_node_id)
            
            edge_text_list = []
            for retrieved_node_id in retrieved_node_id_list:
                if target_node_type == 'passage':
                    passage_text = self.passage_key_to_content[retrieved_node_id]['text']
                    target_text = passage_text
                else:
                    table_content = self.table_key_to_content[retrieved_node_id]
                    table_title = table_content['title']
                    table_rows = table_content['text'].split('\n')
                    column_names = table_rows[0]
                    row_values = table_rows[1:]
                    target_text = table_title + ' [SEP] ' + column_names + ' [SEP] ' + '\n'.join(row_values)
                
                edge_text = f"{query_node_text} [SEP] {target_text}"
                edge_text_list.append(edge_text)
            
            edge_count_list.append(len(edge_text_list))
            edge_total_list.extend(edge_text_list)
            retrieved_passage_total_id_list.extend(retrieved_node_id_list)
            
        if iteration == 0:
            edges = self.colbert_edge_and_table_retriever.checkpoint.doc_tokenizer.tensorize(edge_total_list)
            queries = self.colbert_edge_and_table_retriever.checkpoint.query_tokenizer.tensorize([nl_question])
            encoded_Q = self.colbert_edge_and_table_retriever.checkpoint.query(*queries)
            Q_duplicated = encoded_Q.repeat_interleave(len(edge_total_list), dim=0).contiguous()
            encoded_D, encoded_D_mask = self.colbert_edge_and_table_retriever.checkpoint.doc(*edges, keep_dims='return_mask')
            pred_scores = self.colbert_edge_and_table_retriever.checkpoint.score(Q_duplicated, encoded_D, encoded_D_mask)
        else:
            # predict scores with reranker
            model_input = [[nl_question, edge] for edge in edge_total_list]
            devided_model_input = []

            for rank in range(self.process_num):
                devided_model_input.append(model_input[rank*len(model_input)//self.process_num:(rank+1)*len(model_input)//self.process_num])
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_list = []
                reranking_scores = []
                for rank in range(self.process_num):
                    future = executor.submit(self.rerank_edges_and_tables_worker, (self.reranker_list[rank], devided_model_input[rank]))
                    future_list.append(future)
                
                concurrent.futures.wait(future_list)
                for future in future_list:
                    reranking_score = future.result()
                    if len(model_input) == 1:
                        reranking_score = [reranking_score]
                    reranking_scores.extend(reranking_score)
                
            pred_scores = torch.tensor(reranking_scores)

        # decompose pred_scores by using edge_count_list
        start_idx = 0
        for source_rank, (query_node_id, query_node_score) in enumerate(topk_query_nodes):
            end_idx = start_idx + edge_count_list[source_rank]
            # and we have to sort the scores in descending order
            sorted_idx = torch.argsort(pred_scores[start_idx:end_idx], descending=True)#[:top_k]
            for target_rank, idx in enumerate(sorted_idx):
                target_node_id = retrieved_passage_total_id_list[start_idx + idx]
                augment_type = f'node_augmentation_{iteration}'
                #get score from pred_scores list()
                query_node_score = float(pred_scores[start_idx + idx])
                self.add_node(graph, query_node_type, query_node_id, target_node_id, query_node_score, augment_type, source_rank, target_rank)
                self.add_node(graph, target_node_type, target_node_id, query_node_id, query_node_score, augment_type, target_rank, source_rank)

            start_idx = end_idx
            
    #@profile
    def hybrid_missing_edge_prediction(self, graph, nl_question, topk_query_nodes, query_node_type, target_node_type, iteration):
        edge_count_list = []
        edge_total_list = []
        retrieved_passage_total_id_list = []
        #top_k = self.top_k_of_missing_edges[iteration]
        column_select_list = []
        row_value_select_list = []
        value_select_list = []
        select_count_list = []
        deleted_node_list = []
        for source_rank, (query_node_id, query_node_score) in enumerate(topk_query_nodes):
            table_key = query_node_id.split('_')[0]
            table_content = self.table_key_to_content[table_key]
            chunk_id = table_content['chunk_id']
            row_id = int(query_node_id.split('_')[1])
            table_title = table_content['title']
            table_column_names = table_content['text'].split('\n')[0]
            table_row_values = table_content['text'].split('\n')[row_id+1]

            refined_column_name_list = []
            column_name_list = table_content['text'].replace(' , ', '[comma]').split('\n')[0].split(', ')
            for column_name in column_name_list:
                column_name = column_name.replace('[comma]', ' , ')
                refined_column_name_list.append(column_name)
            
            row_value_list = table_content['text'].replace(' , ', '[comma]').split('\n')[row_id+1].split(', ')
            refined_row_value_list = []
            for row_value in row_value_list:
                row_value = row_value.replace('[comma]', ' , ')
                refined_row_value_list.append(row_value)

            if len(refined_column_name_list) != len(refined_row_value_list):
                deleted_node_list.append([query_node_id, query_node_score])
                continue

            for column_name in refined_column_name_list:
                column_select_list.append(f"{table_title} [SEP] {column_name}")
            
            for column_name, row_value in zip(refined_column_name_list, refined_row_value_list):
                row_value_select_list.append(f"{table_title} [SEP] {column_name} {row_value}")
                value_select_list.append(row_value)
            
            select_count_list.append(len(refined_column_name_list))

        select_column_list = column_select_list + row_value_select_list
        if len(select_column_list) != 0:
            if iteration == 0:
                edges = self.colbert_edge_and_table_retriever.checkpoint.doc_tokenizer.tensorize(select_column_list)
                queries = self.colbert_edge_and_table_retriever.checkpoint.query_tokenizer.tensorize([nl_question])
                encoded_Q = self.colbert_edge_and_table_retriever.checkpoint.query(*queries)
                Q_duplicated = encoded_Q.repeat_interleave(len(select_column_list), dim=0).contiguous()
                encoded_D, encoded_D_mask = self.colbert_edge_and_table_retriever.checkpoint.doc(*edges, keep_dims='return_mask')
                pred_scores_for_selecting = self.colbert_edge_and_table_retriever.checkpoint.score(Q_duplicated, encoded_D, encoded_D_mask)
            else:
                # predict scores with reranker
                model_input = [[nl_question, select_column] for select_column in select_column_list]
                devided_model_input = []

                for rank in range(self.process_num):
                    devided_model_input.append(model_input[rank*len(model_input)//self.process_num:(rank+1)*len(model_input)//self.process_num])
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_list = []
                    reranking_scores = []
                    for rank in range(self.process_num):
                        future = executor.submit(self.rerank_edges_and_tables_worker, (self.reranker_list[rank], devided_model_input[rank]))
                        future_list.append(future)
                    
                    concurrent.futures.wait(future_list)
                    for future in future_list:
                        reranking_score = future.result()
                        if len(model_input) == 1:
                            reranking_score = [reranking_score]
                        reranking_scores.extend(reranking_score)
                    
                pred_scores_for_selecting = torch.tensor(reranking_scores)
            
            start_idx = 0
            column_pred_scores = pred_scores_for_selecting[:len(column_select_list)]
            row_value_pred_scores = pred_scores_for_selecting[len(column_select_list):]
            final_list = []
            # apply deletion to topk_query_nodes
            for deleted_node in deleted_node_list:
                topk_query_nodes.remove(deleted_node)
            
            for source_rank, (query_node_id, query_node_score) in enumerate(topk_query_nodes):
                end_idx = start_idx + select_count_list[source_rank]
                column_sorted_idx = torch.argsort(column_pred_scores[start_idx:end_idx], descending=True)[0]
                value_soted_idx = torch.argsort(row_value_pred_scores[start_idx:end_idx], descending=True)[0]
                final_column_id_list = list(set([column_sorted_idx, value_soted_idx]))
                seleted_value_list =[]
                for final_column_id in final_column_id_list:
                    seleted_value_list.append(value_select_list[start_idx + final_column_id])
                final_list.append(seleted_value_list)
                start_idx = end_idx
        else:
            final_list = []
            
        
        for source_rank, (query_node_id, query_node_score) in enumerate(topk_query_nodes):
            if len(final_list) != 0:
                selected_value_list = final_list[source_rank]
                table_key = query_node_id.split('_')[0]
                table_content = self.table_key_to_content[table_key]
                chunk_id = table_content['chunk_id']
                row_id = int(query_node_id.split('_')[1])
                table_title = table_content['title']
                table_column_names = table_content['text'].split('\n')[0]
                table_row_values = table_content['text'].split('\n')[row_id+1]
                # Entity Linking
                entity_linking_result = self.table_chunk_id_to_linked_passages[chunk_id]
                linked_passage_list = []
                for mention_info in entity_linking_result['results']:
                    row = mention_info['row']
                    original_cell = mention_info['original_cell']
                    if str(row_id) == str(row) and original_cell in ' '.join(selected_value_list).lower():
                        mention_list = []
                        # min_score = min(mention_info['scores'])
                        # max_score = max(mention_info['scores'])
                        
                        for mention, score in zip(mention_info['retrieved'][1:self.top_k_of_entity_linking[iteration]], mention_info['scores'][1:self.top_k_of_entity_linking[iteration]]):
                            # min_max_scaled_score = (score - min_score) / (max_score - min_score)
                            # if min_max_scaled_score >= 0.5:
                            # if score >= 54.7:
                            mention_list.append(mention)
                            # mention_list.append(mention)
                        linked_passage_list.extend(mention_list)
                
                linked_passage_list = list(set(linked_passage_list))
            else:
                linked_passage_list = []
            # Expanded Query Retrieval
            expanded_query = self.get_expanded_query(nl_question, query_node_id, query_node_type)
            
            retrieved_node_info = self.colbert_passage_retriever.search(expanded_query, self.top_k_of_passage_target[iteration])
            retrieved_id_list = retrieved_node_info[0]
            retrieved_passage_list = []
            for target_rank, retrieved_id in enumerate(retrieved_id_list):
                retrieved_node_id = self.id_to_passage_key[str(retrieved_id)]
                retrieved_passage_list.append(retrieved_node_id)
            
            if source_rank < self.top_k_of_table_segment_query_entity_linking[iteration]:
                retrieved_passage_list = retrieved_passage_list + linked_passage_list
            else:
                retrieved_passage_list = retrieved_passage_list
            
            retrieved_passage_list = list(set(retrieved_passage_list))
            table_text = f"{table_title} [SEP] {table_column_names} [SEP] {table_row_values}"
            
            edge_text_list = []
            for linked_passage_id in retrieved_passage_list:
                passage_text = self.passage_key_to_content[linked_passage_id]['text']
                edge_text = f"{table_text} [SEP] {passage_text}"
                edge_text_list.append(edge_text)
            
            edge_count_list.append(len(edge_text_list))
            edge_total_list.extend(edge_text_list)
            retrieved_passage_total_id_list.extend(retrieved_passage_list)
            
        if iteration == 0:
            edges = self.colbert_edge_and_table_retriever.checkpoint.doc_tokenizer.tensorize(edge_total_list)
            queries = self.colbert_edge_and_table_retriever.checkpoint.query_tokenizer.tensorize([nl_question])
            encoded_Q = self.colbert_edge_and_table_retriever.checkpoint.query(*queries)
            Q_duplicated = encoded_Q.repeat_interleave(len(edge_total_list), dim=0).contiguous()
            encoded_D, encoded_D_mask = self.colbert_edge_and_table_retriever.checkpoint.doc(*edges, keep_dims='return_mask')
            pred_scores = self.colbert_edge_and_table_retriever.checkpoint.score(Q_duplicated, encoded_D, encoded_D_mask)
        else:
            # predict scores with reranker
            model_input = [[nl_question, edge] for edge in edge_total_list]
            devided_model_input = []

            for rank in range(self.process_num):
                devided_model_input.append(model_input[rank*len(model_input)//self.process_num:(rank+1)*len(model_input)//self.process_num])
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_list = []
                reranking_scores = []
                for rank in range(self.process_num):
                    future = executor.submit(self.rerank_edges_and_tables_worker, (self.reranker_list[rank], devided_model_input[rank]))
                    future_list.append(future)
                
                concurrent.futures.wait(future_list)
                for future in future_list:
                    reranking_score = future.result()
                    if len(model_input) == 1:
                        reranking_score = [reranking_score]
                    reranking_scores.extend(reranking_score)
                
            pred_scores = torch.tensor(reranking_scores)

        # decompose pred_scores by using edge_count_list
        start_idx = 0
        for source_rank, (query_node_id, query_node_score) in enumerate(topk_query_nodes):
            end_idx = start_idx + edge_count_list[source_rank]
            # and we have to sort the scores in descending order
            sorted_idx = torch.argsort(pred_scores[start_idx:end_idx], descending=True)#[:top_k]
            for target_rank, idx in enumerate(sorted_idx):
                retrieved_node_id = retrieved_passage_total_id_list[start_idx + idx]
                augment_type = f'node_augmentation_{iteration}'
                #get score from pred_scores list()
                query_node_score = float(pred_scores[start_idx + idx])
                self.add_node(graph, query_node_type, query_node_id, retrieved_node_id, query_node_score, augment_type, source_rank, target_rank)
                self.add_node(graph, target_node_type, retrieved_node_id, query_node_id, query_node_score, augment_type, target_rank, source_rank)
            
            start_idx = end_idx


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
    
    
    #@profile
    def find_relevant_nodes(self, nl_question, reduced_search_space):
        table_segment_id_to_augmented_nodes, table_id_to_augmented_nodes = self.get_table(reduced_search_space)

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
        
        self.select_passages(nl_question, selected_table_segment_list, reduced_search_space)
        retrieval_type = 'edge_reranking'
        self.assign_scores(reduced_search_space, retrieval_type)
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
        table_key_to_augmented_nodes = {}
        table_segment_key_to_augmented_nodes = {}
        
        sorted_retrieved_graph = sorted(retrieved_graph.items(), key=lambda x: x[1]['score'], reverse=True)
        
        for node_id, node_info in sorted_retrieved_graph:
            
            if node_info['type'] == 'table segment' and (len(table_segment_key_to_augmented_nodes) + len(table_key_to_augmented_nodes)) < self.top_k_of_table_segment_select_w_llm:
                table_segment_key_to_augmented_nodes[node_id] = list(set([node_info[0] for node_info in node_info['linked_nodes'] if node_info[2] not in ['table_to_table_segment', 'table_reranking']]))
            elif node_info['type'] == 'table' and (len(table_segment_key_to_augmented_nodes) + len(table_key_to_augmented_nodes)) < self.top_k_of_table_select_w_llm:
                table_segment_node_list = list(set([node_info[0] for node_info in node_info['linked_nodes']]))
                for table_segment_node_id in table_segment_node_list:
                    linked_nodes = retrieved_graph[table_segment_node_id]['linked_nodes']
                    row_id = table_segment_node_id.split('_')[1]
                    table_key_to_augmented_nodes[node_id] = {}
                    table_key_to_augmented_nodes[node_id][row_id] = list(set([node_info[0] for node_info in linked_nodes if node_info[2] not in ['table_to_table_segment', 'table_reranking']]))

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
    #@profile
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
                    row_id = selected_row.split('_')[1]
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
                    trimmed_tokenized_content = tokenized_content[ : self.table_and_linked_passages_trim_length]
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
    #@profile
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
                trimmed_tokenized_content = tokenized_content[ : self.passage_trim_length]
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
                self.add_node(integrated_graph, 'table segment', table_segment_node_id, selected_passage_id,   self.max_edge_score, 'llm_selected')
                self.add_node(integrated_graph, 'passage',       selected_passage_id,   table_segment_node_id, self.max_edge_score, 'llm_selected')
    

    
    

    
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

    def assign_scores(self, graph, retrieval_type = None):
        # Filter linked scores based on retrieval_type if provided
        for node_id, node_info in graph.items():
            if retrieval_type is not None:
                if node_info['type'] == 'table segment':
                    filtered_retrieval_type = ['edge_and_table_retrieval', 'node_augmentation_0', 'table_to_table_segment', 'table_reranking']
                else:
                    filtered_retrieval_type = ['edge_and_table_retrieval', 'node_augmentation_0', 'table_to_table_segment']
            
                linked_scores = [linked_node[1] for linked_node in node_info['linked_nodes'] if linked_node[2] not in filtered_retrieval_type]
            else:
                linked_scores = [linked_node[1] for linked_node in node_info['linked_nodes']]

            # Assign scores based on the selected method
            if self.node_scoring_method == 'min':
                node_score = min(linked_scores) if linked_scores else self.min_edge_score
            elif self.node_scoring_method == 'max':
                node_score = max(linked_scores) if linked_scores else self.min_edge_score
            elif self.node_scoring_method == 'mean':
                node_score = sum(linked_scores) / len(linked_scores) if linked_scores else self.min_edge_score




            
            # Assign the computed score to the node
            graph[node_id]['score'] = node_score

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
def evaluate(retrieved_graph, qa_data, graph_query_engine, score_function = 'max'):
    
    table_key_to_content = graph_query_engine.table_key_to_content
    passage_key_to_content = graph_query_engine.passage_key_to_content
    # table_key_to_content = graph_query_engine.table_key_to_content
    # passage_key_to_content = graph_query_engine.passage_key_to_content
    # filtered_retrieval_type = ['edge_reranking', "passage_node_augmentation_1"]
    # filtered_retrieval_type_1 = ['edge_reranking']
    filtered_retrieval_type = ['edge_reranking', "node_augmentation_1", "passage_node_augmentation_1", 'llm_selected']
    filtered_retrieval_type_1 = ['edge_reranking', 'llm_selected']#, 'llm_selected'
    filtered_retrieval_type_2 = ["node_augmentation_1", "passage_node_augmentation_1"]
    # filtered_retrieval_type = ['edge_retrieval', "passage_node_augmentation_0", "llm_selected"]
    # filtered_retrieval_type = ['edge_retrieval', "passage_node_augmentation_1"]
    # filtered_retrieval_type_1 = ['edge_retrieval', "llm_selected"]
    # 1. Revise retrieved graph
    revised_retrieved_graph = {}
    for node_id, node_info in retrieved_graph.items():                  

        if node_info['type'] == 'table segment':
            linked_nodes = [x for x in node_info['linked_nodes'] 
                                if x[2] in filtered_retrieval_type and (x[2] in filtered_retrieval_type_2 and (x[3] < 10) and (x[4] < 10000)) 
                                    or x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[4] < 1) and (x[3] < 1)) 
                                    or x[2] in filtered_retrieval_type_1#['edge_reranking', "llm_selected"]
                            ]
        elif node_info['type'] == 'passage':
            linked_nodes = [x for x in node_info['linked_nodes'] 
                                if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < 1) and (x[4] < 1)) 
                                or x[2] in filtered_retrieval_type and (x[2] in filtered_retrieval_type_2 and (x[4] < 10) and (x[3] < 10000)) 
                                or x[2] in filtered_retrieval_type_1#['edge_reranking', "llm_selected"]
                            ]
        else:
            continue
        
        if len(linked_nodes) == 0: continue
        
        revised_retrieved_graph[node_id] = copy.deepcopy(node_info)
        revised_retrieved_graph[node_id]['linked_nodes'] = linked_nodes
        
        linked_scores = [linked_node[1] for linked_node in linked_nodes]

        if score_function == 'max':
            node_score = max(linked_scores)
        elif score_function == 'avg':
            node_score = sum(linked_scores) / len(linked_scores)
        else:
            node_score = min(linked_scores)

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
        if edge_count < 50:
            node_count += 1
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

            max_linked_node_id, _, _, _, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default = (None, 0, 'edge_reranking', 0, 0))
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

        # node_count += 1

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