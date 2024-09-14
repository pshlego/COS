import re
import json
import copy
import unicodedata
from Ours.dpr.data.qa_validation import has_answer
from Ours.dpr.utils.tokenizers import SimpleTokenizer
import ast
import json
import time
import hydra
import torch
import requests
from tqdm import tqdm
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import set_seed
from llm_based_retrieval import LlmNodeSelector
from prompt.prompts import select_table_segment_prompt, select_passage_prompt

set_seed(0)

@hydra.main(config_path = "conf", config_name = "bipartite_subgraph_retrieval")
def main(cfg: DictConfig):
    # load qa dataset
    print()
    print(f"[[ Loading qa dataset... ]]", end = "\n\n")
    qa_dataset = json.load(open(cfg.qa_dataset_path))
    print(f"Number of questions: {len(qa_dataset)}", end = "\n\n")
    
    # load graph query engine
    print(f"[[ Loading graph query engine... ]]", end = "\n\n")
    graph_query_engine = GraphQueryEngine(cfg)
    print(f"Graph query engine loaded successfully!", end = "\n\n")

    print(f"Start querying...")
    for qidx, qa_datum in tqdm(enumerate(qa_dataset), total = len(qa_dataset)):
        question = qa_datum['question']
        init_time = time.time()
        retrieved_graph = graph_query_engine.query(question)
        end_time = time.time()
        query_time = end_time - init_time
        
        # save retrieved graph
        to_print = {
            "qa data": qa_datum,
            "retrieved graph": retrieved_graph
        }
        
        with open(cfg.final_result_path, 'a+') as file:
            file.write(json.dumps(to_print) + '\n')
        
        with open(cfg.query_time_save_path, 'a+') as file:
            file.write(json.dumps(query_time) + '\n')

    print(f"Querying done.")

class GraphQueryEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.edge_retriever_addr = "http://localhost:5000/edge_retrieve"#"http://localhost:5000/edge_retrieve"
        self.table_segment_retriever_addr = "http://localhost:5001/table_segment_retrieve"
        self.passage_retriever_addr = "http://localhost:5002/passage_retrieve"
        self.reranker_addr = "http://localhost:5003/rerank"
        self.llm_addr = "http://localhost:5004/generate"
        self.trim_addr = "http://localhost:5004/trim"
        
        # 3. Load tables
        print("1. Loading tables...")
        table_contents = json.load(open(cfg.table_data_path))
        TABLES_NUM = len(table_contents)
        self.table_key_to_content = {str(table_key): table_content for table_key, table_content in tqdm(enumerate(table_contents), total = TABLES_NUM)}
        print("1. Loaded " + str(TABLES_NUM) + " tables!", end = "\n\n")


        # 4. Load passages
        print("2. Loading passages...")
        passage_contents = json.load(open(cfg.passage_data_path))
        PASSAGES_NUM = len(passage_contents)
        self.passage_key_to_content = {str(passage_content['title']): passage_content for passage_content in tqdm(passage_contents)}
        self.passage_id_to_passage_title = {str(passage_content['chunk_id']): passage_content['title'] for passage_content in tqdm(passage_contents)}

        print("2. Loaded " + str(PASSAGES_NUM) + " passages!", end = "\n\n")
        
        
        self.select_passage_prompt = select_passage_prompt
        self.select_table_segment_prompt = select_table_segment_prompt

        self.llm_node_selector = LlmNodeSelector(cfg, self.table_key_to_content, self.passage_key_to_content)
    @profile
    def query(self, question):
        # 1. Search bipartite subgraph candidates
        bipartite_subgraph_candidates = self.search_bipartite_subgraph_candidates(question)
        
        # 2. Find bipartite subgraph
        self.find_bipartite_subgraph(question, bipartite_subgraph_candidates)
        
        return bipartite_subgraph_candidates

    @profile
    def search_bipartite_subgraph_candidates(self, question):
        # 1.1 Retrieve edges
        retrieved_edges = self.retrieve_edges(question)

        # 1.2 Rerank edges
        reranked_edges = self.rerank_edges(question, retrieved_edges)

        # 1.3 Integrate edges into bipartite subgraph candidates
        bipartite_subgraph_candidates = self.integrate_into_graph(reranked_edges)

        # 1.4 Assign scores to nodes
        self.assign_scores_to_nodes(question, bipartite_subgraph_candidates)
        
        # 1.5 Augment nodes to bipartite subgraph candidates
        self.augment_nodes(question, bipartite_subgraph_candidates)
        
        return bipartite_subgraph_candidates
    @profile
    def retrieve_edges(self, question):
        response = requests.post(
            self.edge_retriever_addr,
            json={
                "query": question,
                "k": self.cfg.top_k_of_retrieved_edges
            },
            timeout=None,
        ).json()
        
        retrieved_edges = response['edge_content_list']
        # For MMQA, we need to convert linked_entity_id to linked_entity_title
        for retrieved_edge in retrieved_edges:
            retrieved_edge['linked_entity_id'] = self.passage_id_to_passage_title[retrieved_edge['linked_entity_id']]
        
        return retrieved_edges
    @profile
    def rerank_edges(self, question, retrieved_edges):
        model_input = []
        for element in retrieved_edges:
            table_key = element['chunk_id'].split('_')[0]
            row_id = int(element['chunk_id'].split('_')[1])
            table_content = self.table_key_to_content[table_key]
            table_title = table_content['title']
            table_rows = table_content['text'].split('\n')
            column_names = table_rows[0]
            row_values = table_rows[row_id+1]
            table_text = table_title + ' [SEP] ' + column_names + ' [SEP] ' + row_values
            passage_key = element['linked_entity_id']
            passage_text = self.passage_key_to_content[passage_key]['text']
            edge_text = table_text + ' [SEP] ' + passage_text
            model_input.append([question, edge_text])
        
        response = requests.post(
            self.reranker_addr,
            json={
                "model_input": model_input,
                "max_length": self.cfg.reranking_edge_max_length
            },
            timeout=None,
        ).json()
        
        model_input = response['model_input']
        reranking_scores = response['reranking_scores']
        
        for retrieved_edge, reranking_score in tqdm(zip(retrieved_edges, reranking_scores), total = len(retrieved_edges)):
            retrieved_edge['reranking_score'] = float(reranking_score)
        
        # Sort edges by reranking score
        reranked_edges = sorted(retrieved_edges, key = lambda x: x['reranking_score'], reverse = True)[:self.cfg.top_k_of_reranked_edges]
        
        return reranked_edges
    @profile
    def integrate_into_graph(self, reranked_edges):
        bipartite_subgraph_candidates = {}
        
        for reranked_edge in reranked_edges:
            if 'linked_entity_id' in reranked_edge:
                # get edge info
                table_key = str(reranked_edge['table_id'])
                row_id = reranked_edge['chunk_id'].split('_')[1]
                table_segment_node_id = f"{table_key}_{row_id}"
                passage_id = reranked_edge['linked_entity_id']
                
                # get edge score
                edge_score = reranked_edge['reranking_score']

                # add nodes
                self.add_node(bipartite_subgraph_candidates, 'table segment', table_segment_node_id, passage_id, edge_score, 'edge_reranking')
                self.add_node(bipartite_subgraph_candidates, 'passage', passage_id, table_segment_node_id, edge_score, 'edge_reranking')
        
        return bipartite_subgraph_candidates
    @profile
    def assign_scores_to_nodes(self, question, bipartite_subgraph_candidates):
        if self.cfg.node_scoring_method == 'direct':
            node_text_list = []
            node_id_list = []
            for node_id, node_info in bipartite_subgraph_candidates.items():
                if 'score' in node_info:
                    continue
                
                if node_info['type'] == 'table segment':
                    table_id = node_id.split('_')[0]
                    row_id = int(node_id.split('_')[1])
                    table_content = self.table_key_to_content[table_id]
                    table_title = table_content['title']
                    table_rows = table_content['text'].split('\n')
                    column_names = table_rows[0]
                    row_values = table_rows[row_id+1]
                    table_text = table_title + ' [SEP] ' + column_names + ' [SEP] ' + row_values
                    node_text_list.append(table_text)
                    node_id_list.append(node_id)
                elif node_info['type'] == 'passage':
                    passage_content = self.passage_key_to_content[node_id]
                    passage_text = passage_content['title'] + ' [SEP] ' + passage_content['text']
                    node_text_list.append(passage_text)
                    node_id_list.append(node_id)
            
            model_input = [[question, node_text] for node_text in node_text_list]
            
            response = requests.post(
                self.reranker_addr,
                json={
                    "model_input": model_input,
                    "max_length": self.cfg.reranking_node_max_length
                },
                timeout=None,
            ).json()

            model_input = response['model_input']
            reranking_scores = response['reranking_scores']

            # Assign the computed score to the node
            for node_id, node_score in zip(node_id_list, reranking_scores):
                bipartite_subgraph_candidates[node_id]['score'] = node_score
            
        else:
            self.approximate_with_edge_scores(bipartite_subgraph_candidates)
    @profile
    def approximate_with_edge_scores(self, bipartite_subgraph_candidates):
        # Filter linked scores based on retrieval_type if provided
        for node_id, node_info in bipartite_subgraph_candidates.items():
            linked_scores = [linked_node[1] for linked_node in node_info['linked_nodes']]

            # Assign scores based on the selected method
            if self.node_scoring_method == 'min':
                node_score = min(linked_scores) if linked_scores else self.cfg.min_edge_score
            elif self.node_scoring_method == 'max':
                node_score = max(linked_scores) if linked_scores else self.cfg.min_edge_score
            elif self.node_scoring_method == 'mean':
                node_score = sum(linked_scores) / len(linked_scores) if linked_scores else self.cfg.min_edge_score
                
            # Assign the computed score to the node
            bipartite_subgraph_candidates[node_id]['score'] = node_score

    @profile
    def augment_nodes(self, nl_question, integrated_graph):
        node_list = []
        for node_id, node_info in integrated_graph.items():
            node_list.append((node_id, node_info['score'], node_info['type']))
        
        # Softmax를 이용해 Query Node의 Score를 확률값으로 변환
        node_scores = torch.tensor([node[1] for node in node_list])
        node_probs = F.softmax(node_scores, dim=0)
        
        # 확률값과 node_id, type을 함께 저장
        topk_query_nodes = sorted(zip(node_list, node_probs), key=lambda x: x[1], reverse=True)[:self.cfg.beam_size]
        
        self.expanded_query_retrieval(integrated_graph, nl_question, topk_query_nodes)

    @profile
    def expanded_query_retrieval(self, graph, nl_question, topk_query_nodes):
        final_prob_list = []  # 최종 확률값을 저장할 리스트
        
        for (query_node, query_node_prob) in topk_query_nodes:
            query_node_id, query_node_score, query_node_type = query_node
            
            expanded_query = self.get_expanded_query(nl_question, query_node_id, query_node_type)
            
            if query_node_type == 'table segment':
                target_node_type = 'passage'
                table_key = query_node_id.split('_')[0]
                row_id = int(query_node_id.split('_')[1])
                table = self.table_key_to_content[table_key]
                table_title = table['title']
                table_column_names = table['text'].split('\n')[0]
                table_row_values = table['text'].split('\n')[row_id+1]
                query_node_text = f"{table_title} [SEP] {table_column_names} [SEP] {table_row_values}"
                response = requests.post(
                    self.passage_retriever_addr,
                    json={
                        "query": expanded_query,
                        "k": self.cfg.beam_size
                    },
                    timeout=None,
                ).json()
                
                retrieved_node_id_list = response['retrieved_key_list']
                retrieved_score_list = response['retrieved_score_list']
            else:
                target_node_type = 'table segment'
                passage = self.passage_key_to_content[query_node_id]
                query_node_text = f"{passage['title']} [SEP] {passage['text']}"
                response = requests.post(
                    self.table_segment_retriever_addr,
                    json={
                        "query": expanded_query,
                        "k": self.cfg.beam_size
                    },
                    timeout=None,
                ).json()
                
                retrieved_node_id_list = response['retrieved_key_list']
                retrieved_score_list = response['retrieved_score_list']

            # retrieved_score_list를 확률값으로 변환
            retrieved_scores = torch.tensor(retrieved_score_list)
            retrieved_probs = F.softmax(retrieved_scores, dim=0)

            # Query Node 확률과 retrieved 확률을 곱해서 final_prob 계산
            for idx, target_node_id in enumerate(retrieved_node_id_list):
                final_prob = query_node_prob * retrieved_probs[idx]
                final_prob_list.append((query_node_id, target_node_id, final_prob, query_node_type, target_node_type))
        
        # 최종 확률값을 기준으로 정렬하여 상위 beam size만큼 선택
        final_prob_list = sorted(final_prob_list, key=lambda x: x[2], reverse=True)[:self.cfg.beam_size]

        edge_total_list = []
        for query_node_id, target_node_id, final_prob, query_node_type, target_node_type in final_prob_list:
            if query_node_type == 'table segment':
                table_key = query_node_id.split('_')[0]
                row_id = int(query_node_id.split('_')[1])
                table = self.table_key_to_content[table_key]
                table_title = table['title']
                table_column_names = table['text'].split('\n')[0]
                table_row_values = table['text'].split('\n')[row_id+1]
                query_node_text = f"{table_title} [SEP] {table_column_names} [SEP] {table_row_values}"
                passage_text = self.passage_key_to_content[target_node_id]['text']
                target_text = passage_text
            else:
                passage = self.passage_key_to_content[query_node_id]
                query_node_text = f"{passage['title']} [SEP] {passage['text']}"
                table_key = target_node_id.split('_')[0]
                row_id = int(target_node_id.split('_')[1])
                table_content = self.table_key_to_content[table_key]
                table_title = table_content['title']
                table_rows = table_content['text'].split('\n')
                column_names = table_rows[0]
                row_values = table_rows[1:][row_id]
                target_text = table_title + ' [SEP] ' + column_names + ' [SEP] ' + row_values

            edge_text = f"{query_node_text} [SEP] {target_text}"
            edge_total_list.append(edge_text)

        # predict scores with reranker for final beam edges
        model_input = [[nl_question, edge] for edge in edge_total_list]
        
        response = requests.post(
            self.reranker_addr,
            json={
                "model_input": model_input,
                "max_length": self.cfg.reranking_edge_max_length
            },
            timeout=None,
        ).json()

        reranking_scores = response['reranking_scores']
        
        # reranking_scores와 final_prob_list를 결합
        for i, (query_node_id, target_node_id, final_prob, query_node_type, target_node_type) in enumerate(final_prob_list):
            reranked_prob = float(torch.tensor(reranking_scores[i]))
            query_rank = [node[0] for node in final_prob_list].index(query_node_id)
            self.add_node(graph, query_node_type, query_node_id, target_node_id, reranked_prob, 'node_augmentation', query_rank, i)
            self.add_node(graph, target_node_type, target_node_id, query_node_id, reranked_prob, 'node_augmentation', i, query_rank)

    @profile
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

    @profile
    def find_bipartite_subgraph(self, question, bipartite_subgraph_candidates):
        bipartite_subgraph_candidate_list, table_id_to_row_id_to_linked_passage_ids = self.get_bipartite_subgraph_candidate_list(bipartite_subgraph_candidates)
        
        is_aggregate = self.llm_node_selector.detect_aggregation_query(question)
        if is_aggregate:
            selected_rows = self.llm_node_selector.select_row_wise(question, table_id_to_row_id_to_linked_passage_ids)
            if len(selected_rows) != 0:
                for table_id, row_id, linked_passage_ids in selected_rows:
                    table_segment_id = f"{table_id}_{row_id}"
                    if table_segment_id not in [bipartite_subgraph_candidate['table_segment_id'] for bipartite_subgraph_candidate in bipartite_subgraph_candidate_list]:
                        bipartite_subgraph_candidate_list.append({"table_segment_id":table_segment_id, "linked_passage_ids": linked_passage_ids})

        table_segment_id_to_passage_id_list = self.llm_node_selector.select_passage_wise(question, bipartite_subgraph_candidate_list)
        
        for table_segment_id, passage_id_list in table_segment_id_to_passage_id_list.items():
            for passage_id in passage_id_list:
                self.add_node(bipartite_subgraph_candidates, 'table segment', table_segment_id, passage_id, self.cfg.max_edge_score, 'llm_selected')
                self.add_node(bipartite_subgraph_candidates, 'passage', passage_id, table_segment_id, self.cfg.max_edge_score, 'llm_selected')

    def llm_based_pruning(self, question, bipartite_subgraph_candidates):
        bipartite_subgraph_candidate_list, table_id_to_row_id_to_linked_passage_ids = self.get_bipartite_subgraph_candidate_list(bipartite_subgraph_candidates)
        
        is_aggregate = self.llm_node_selector.detect_aggregation_query(question)
        if is_aggregate:
            selected_rows = self.llm_node_selector.select_row_wise(question, table_id_to_row_id_to_linked_passage_ids)
            if len(selected_rows) != 0:
                for table_id, row_id, linked_passage_ids in selected_rows:
                    table_segment_id = f"{table_id}_{row_id}"
                    if table_segment_id not in [bipartite_subgraph_candidate['table_segment_id'] for bipartite_subgraph_candidate in bipartite_subgraph_candidate_list]:
                        bipartite_subgraph_candidate_list.append({"table_segment_id":table_segment_id, "linked_passage_ids": linked_passage_ids})

        table_segment_id_to_passage_id_list = self.llm_node_selector.prune_passage_wise(question, bipartite_subgraph_candidate_list)
        
        for table_segment_id, passage_id_list in table_segment_id_to_passage_id_list.items():
            for passage_id in passage_id_list:
                self.delete_node(bipartite_subgraph_candidates, 'table segment', table_segment_id, passage_id)
                self.delete_node(bipartite_subgraph_candidates, 'passage', passage_id, table_segment_id)


    @profile
    def get_bipartite_subgraph_candidate_list(self, bipartite_subgraph_candidates):
        table_segment_id_to_linked_passage_ids = {}
        table_id_to_row_id_to_linked_passage_ids = {}
        bipartite_subgraph_candidate_list = []
        
        sorted_node_list = sorted(bipartite_subgraph_candidates.items(), key=lambda x: max([node_info[1] for node_info in x[1]['linked_nodes']]), reverse=True)
        
        for node_id, node_info in sorted_node_list:
            if node_info['type'] == 'table segment':
                table_segment_id_to_linked_passage_ids[node_id] = list(set([node_info[0] for node_info in node_info['linked_nodes']]))
        
        for table_segment_id, linked_passage_ids in table_segment_id_to_linked_passage_ids.items():
            table_id = table_segment_id.split('_')[0]
            row_id = table_segment_id.split('_')[1]
            
            if table_id not in table_id_to_row_id_to_linked_passage_ids:
                table_id_to_row_id_to_linked_passage_ids[table_id] = {}
            
            if row_id not in table_id_to_row_id_to_linked_passage_ids[table_id]:
                table_id_to_row_id_to_linked_passage_ids[table_id][row_id] = []
                
            table_id_to_row_id_to_linked_passage_ids[table_id][row_id].extend(linked_passage_ids)
            
            bipartite_subgraph_candidate_list.append(
                {
                    "table_segment_id": table_segment_id, 
                    "linked_passage_ids": list(set(table_id_to_row_id_to_linked_passage_ids[table_id][row_id]))
                }
            )

            if len(bipartite_subgraph_candidate_list) >= 10:
                break
        table_id_list = list(table_id_to_row_id_to_linked_passage_ids.keys())
        
        for table_id in table_id_list:
            row_id_to_linked_passage_ids = table_id_to_row_id_to_linked_passage_ids[table_id]
            table_content = self.table_key_to_content[table_id]
            rows = table_content['text'].split('\n')[1:]
            for row_id, row in enumerate(rows):
                if row == "":
                    continue
                if str(row_id) not in row_id_to_linked_passage_ids:
                    row_id_to_linked_passage_ids[str(row_id)] = []
                    
        
        return bipartite_subgraph_candidate_list, table_id_to_row_id_to_linked_passage_ids
    @profile
    def add_node(self, graph, source_node_type, source_node_id, target_node_id, score, retrieval_type, source_rank = 0, target_rank = 0):
        if source_node_id not in graph:
            graph[source_node_id] = {'type': source_node_type, 'linked_nodes': [[target_node_id, score, retrieval_type, source_rank, target_rank]]}
        else:
            graph[source_node_id]['linked_nodes'].append([target_node_id, score, retrieval_type, source_rank, target_rank])

    def delete_node(self, graph, source_node_type, source_node_id, target_node_id):
        if source_node_id in graph:
            linked_nodes = graph[source_node_id]['linked_nodes']
            graph[source_node_id]['linked_nodes'] = [node for node in linked_nodes if node[0] != target_node_id]
            if len(graph[source_node_id]['linked_nodes']) == 0:
                del graph[source_node_id]

    @profile
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
        # self.assign_scores(nl_question, reduced_search_space, retrieval_type)
        

    def get_table(self, retrieved_graph):
        table_key_to_augmented_nodes = {}
        table_segment_key_to_augmented_nodes = {}
        
        sorted_retrieved_graph = sorted(retrieved_graph.items(), key=lambda x: max([node_info[1] for node_info in x[1]['linked_nodes']]), reverse=True)
        
        for node_id, node_info in sorted_retrieved_graph:
            
            if node_info['type'] == 'table segment' and (len(table_segment_key_to_augmented_nodes) + len(table_key_to_augmented_nodes)) < self.cfg.top_k_of_table_segment_select_w_llm:
                table_segment_key_to_augmented_nodes[node_id] = list(set([node_info[0] for node_info in node_info['linked_nodes'] if node_info[2] not in ['table_to_table_segment', 'table_reranking']]))

        return table_segment_key_to_augmented_nodes, table_key_to_augmented_nodes


    def combine_linked_passages(self, table_id_to_linked_nodes):

        table_id_to_row_id_to_linked_passage_ids = {}
        table_id_to_table_info = {}
        
        for table_id, table_segment_to_linked_nodes in table_id_to_linked_nodes.items():
            if table_id not in table_id_to_row_id_to_linked_passage_ids:
                table_id_to_row_id_to_linked_passage_ids[table_id] = {}
            
            # 1. Calculate table info and put it in `table_id_to_table_info`
            if self.table_key_to_content[table_id]['chunk_id'] not in self.table_key_to_content:
                continue
            
            linked_passage_info = self.table_key_to_content[self.table_key_to_content[table_id]['chunk_id']]
            table_title = linked_passage_info['question'].split(' [SEP] ')[0]
            table_column_name = linked_passage_info['question'].split(' [SEP] ')[-1].split('\n')[0]
            table_rows = linked_passage_info['question'].split(' [SEP] ')[-1].split('\n')[1:]
            table_rows = [row for row in table_rows if row != ""]
            table_info = {"title": table_title, "column_name": table_column_name, "rows": table_rows}
            
            table_id_to_table_info[table_id] = table_info
                
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
    
    @profile
    def select_table_segments(self, nl_question, table_id_to_row_id_to_linked_passage_ids, table_id_to_table_info):
        
        prompt_list = []
        table_id_list = []
        
        for table_id in table_id_to_table_info.keys():
            table_and_linked_passages = self.stringify_table_and_linked_passages(
                                                table_id_to_table_info[table_id],
                                                table_id_to_row_id_to_linked_passage_ids[table_id]
                                            )
            contents_for_prompt = {'question': nl_question, 'table_and_linked_passages': table_and_linked_passages}
            prompt = self.get_prompt(contents_for_prompt)
            prompt_list.append(prompt)
            table_id_list.append(table_id)
            
        response_list = requests.post(
                self.llm_addr,
                json={
                    "prompt_list": prompt_list,
                    "max_tokens": 64
                },
                timeout=None,
            ).json()["response_list"]
        
        selected_table_segment_list = []
        for table_id, response in zip(table_id_list, response_list):
            selected_rows = response
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
                    response = requests.post(
                        self.trim_addr,
                        json={
                            "raw_text": linked_passage_content,
                            "trim_length": self.cfg.table_and_linked_passages_trim_length
                        },
                        timeout=None,
                    ).json()
                    trimmed_text = response["trimmed_text"]
                    table_and_linked_passages += f"- {trimmed_text}\n"
                    
            table_and_linked_passages += "\n\n"

        return table_and_linked_passages


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
                response = requests.post(
                    self.trim_addr,
                    json={
                        "raw_text": content,
                        "trim_length": self.cfg.passage_trim_length
                    },
                    timeout=None,
                ).json()
                trimmed_text = response["trimmed_text"]
                linked_passage_contents += f"Title: {title}. Content: {trimmed_text}\n\n"

            contents_for_prompt = {"question": nl_question, "table_segment": table_segment_content, "linked_passages": linked_passage_contents}
            prompt = self.get_prompt(contents_for_prompt)
            prompt_list.append(prompt)
            table_segment_node_id_list.append(table_segment_node_id)
        
        # 2. Run LLM
        response_list = requests.post(
                self.llm_addr,
                json={
                    "prompt_list": prompt_list,
                    "max_tokens": 128
                },
                timeout=None,
            ).json()["response_list"]
        
        # 3. Parse LLM results and add the top 
        for table_segment_node_id, response in zip(table_segment_node_id_list, response_list):
            selected_passage_id_list = response
            try:    selected_passage_id_list = ast.literal_eval(selected_passage_id_list)
            except: selected_passage_id_list = [selected_passage_id_list]
            
            
            if selected_passage_id_list is None:
                continue
            
            for selected_passage_id in selected_passage_id_list:
                if selected_passage_id not in self.passage_key_to_content: continue
                self.add_node(integrated_graph, 'table segment', table_segment_node_id, selected_passage_id,   self.cfg.max_edge_score, 'llm_selected')
                self.add_node(integrated_graph, 'passage',       selected_passage_id,   table_segment_node_id, self.cfg.max_edge_score, 'llm_selected')

    def get_prompt(self, contents_for_prompt):
        if 'linked_passages' in contents_for_prompt:
            prompt = self.select_passage_prompt.format(**contents_for_prompt)
        else:
            prompt = self.select_table_segment_prompt.format(**contents_for_prompt)
        
        return prompt
    
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def evaluate(retrieved_graph, qa_data, table_key_to_content, passage_key_to_content, edge_limit):
    filtered_retrieval_type = ['edge_reranking', "node_augmentation", 'retrieval', 'llm_selected']
    filtered_retrieval_type_1 = ['edge_reranking',  'retrieval', 'llm_selected']
    filtered_retrieval_type_2 = ["node_augmentation", 'retrieval']
    # 1. Revise retrieved graph
    revised_retrieved_graph = {}
    for node_id, node_info in retrieved_graph.items():                  

        if node_info['type'] == 'table segment':
            linked_nodes = [x for x in node_info['linked_nodes'] 
                                if x[2] in filtered_retrieval_type and (x[2] in filtered_retrieval_type_2 and (x[3] < 10) and (x[4] < 10000)) 
                                    or x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[4] < 0) and (x[3] < 0)) 
                                    or x[2] in filtered_retrieval_type_1
                            ]
        elif node_info['type'] == 'passage':
            linked_nodes = [x for x in node_info['linked_nodes'] 
                                if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < 0) and (x[4] < 0)) 
                                or x[2] in filtered_retrieval_type and (x[2] in filtered_retrieval_type_2 and (x[4] < 10) and (x[3] < 10000)) 
                                or x[2] in filtered_retrieval_type_1
                            ]
        else:
            continue
        
        if len(linked_nodes) == 0: continue
        
        revised_retrieved_graph[node_id] = copy.deepcopy(node_info)
        revised_retrieved_graph[node_id]['linked_nodes'] = linked_nodes
        
        linked_scores = [linked_node[1] for linked_node in linked_nodes]

        node_score = max(linked_scores)
        
        if node_score == 1000000:
            additional_score_list = [linked_node[1] for linked_node in linked_nodes if linked_node[2] != 'llm_selected']
            if len(additional_score_list) > 0:
                node_score += max(additional_score_list)

        revised_retrieved_graph[node_id]['score'] = node_score


    # 2. Evaluate with revised retrieved graph
    node_count = 0
    edge_count = 0
    answers = qa_data['answers']
    context = ""
    retrieved_table_set = set()
    retrieved_passage_set = set()
    sorted_retrieved_graph = sorted(revised_retrieved_graph.items(), key = lambda x: x[1]['score'], reverse = True)
    
    for node_id, node_info in sorted_retrieved_graph:
        if edge_count < edge_limit:
            node_count += 1
        if node_info['type'] == 'table segment':
            
            table_id = node_id.split('_')[0]
            table = table_key_to_content[table_id]
            chunk_id = table['chunk_id']
            node_info['chunk_id'] = chunk_id
            
            if table_id not in retrieved_table_set:
                retrieved_table_set.add(table_id)
                
                if edge_count == edge_limit:
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
            
            if edge_count == edge_limit:
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
                if edge_count == edge_limit:
                    continue
                context += table['text']
                edge_count += 1

            row_id = int(max_linked_node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            table_segment_text = column_name + '\n' + row_values
            
            if edge_count == edge_limit:
                continue

            retrieved_passage_set.add(node_id)
            passage_content = passage_key_to_content[node_id]
            passage_text = passage_content['title'] + ' ' + passage_content['text']
            
            edge_text = table_segment_text + '\n' + passage_text
            context += edge_text
            edge_count += 1

    normalized_context = remove_accents_and_non_ascii(context)
    normalized_answers = [remove_accents_and_non_ascii(answer) for answer in answers]
    is_has_answer = has_answer(normalized_answers, normalized_context, SimpleTokenizer(), 'string')
    
    if is_has_answer:
        recall = 1
        error_analysis = copy.deepcopy(qa_data)
        error_analysis['retrieved_graph'] = retrieved_graph
        error_analysis['sorted_retrieved_graph'] = sorted_retrieved_graph[:node_count]
        if  "hard_negative_ctxs" in error_analysis:
            del error_analysis["hard_negative_ctxs"]
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

if __name__ == "__main__":
    main()
# 'f77c5527ab108782'
# +2