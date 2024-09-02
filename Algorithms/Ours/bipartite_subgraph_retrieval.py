import ast
import json
import time
import hydra
import torch
import requests
from tqdm import tqdm
from omegaconf import DictConfig
from transformers import set_seed
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
        # query graph for each question
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
        
        self.edge_retriever_addr = "http://localhost:5000/edge_retrieve"
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
        print("2. Loaded " + str(PASSAGES_NUM) + " passages!", end = "\n\n")
        
        
        self.select_passage_prompt = select_passage_prompt
        self.select_table_segment_prompt = select_table_segment_prompt


    def query(self, question):
        # 1. Search bipartite subgraph candidates
        bipartite_subgraph_candidates = self.search_bipartite_subgraph_candidates(question)
        
        # # 2. Find bipartite subgraph
        # self.find_bipartite_subgraph(question, bipartite_subgraph_candidates)
        
        self.find_relevant_nodes(question, bipartite_subgraph_candidates)
        
        return bipartite_subgraph_candidates
        # return bipartite_subgraph


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
        
    def retrieve_edges(self, question):
        response = requests.post(
            self.edge_retriever_addr,
            json={
                "query": question,
                "k": 10000
            },
            timeout=None,
        ).json()
        
        edge_content_list = response['edge_content_list']
        edge_score_list = response['retrieved_score_list']
        
        retrieved_edges = []
        for edge_content, edge_score in zip(edge_content_list, edge_score_list):
            # pass single node graph
            if 'linked_entity_id' not in edge_content:
                continue
            
            edge_content['retrieval_score'] = edge_score
            retrieved_edges.append(edge_content)
            
            if len(retrieved_edges) == self.cfg.top_k_of_retrieved_edges:
                break
        
        return retrieved_edges

    def rerank_edges(self, question, retrieved_edges):
        model_input = [[question, element['title'] + ' [SEP] ' + element['text']] for element in retrieved_edges]
        
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

    def augment_nodes(self, nl_question, integrated_graph):
        node_list = []
        for node_id, node_info in integrated_graph.items():
            node_list.append((node_id, node_info['score'], node_info['type']))
        
        topk_query_nodes = sorted(node_list, key=lambda x: x[1], reverse=True)[:self.cfg.top_k_of_query]
        
        self.expanded_query_retrieval(integrated_graph, nl_question, topk_query_nodes)

    def expanded_query_retrieval(self, graph, nl_question, topk_query_nodes):
        edge_count_list = []
        edge_total_list = []
        target_node_id_list = []
        
        for source_rank, (query_node_id, query_node_score, query_node_type) in enumerate(topk_query_nodes):
            
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
                        "k": self.cfg.top_k_of_target
                    },
                    timeout=None,
                ).json()
                
                retrieved_node_id_list = response['retrieved_key_list']
            else:
                target_node_type = 'table segment'
                passage = self.passage_key_to_content[query_node_id]
                query_node_text = f"{passage['title']} [SEP] {passage['text']}"
                response = requests.post(
                    self.table_segment_retriever_addr,
                    json={
                        "query": expanded_query,
                        "k": self.cfg.top_k_of_target
                    },
                    timeout=None,
                ).json()
                
                retrieved_node_id_list = response['retrieved_key_list']
            
            edge_text_list = []
            for retrieved_node_id in retrieved_node_id_list:
                if target_node_type == 'passage':
                    passage_text = self.passage_key_to_content[retrieved_node_id]['text']
                    target_text = passage_text
                else:
                    table_key = retrieved_node_id.split('_')[0]
                    row_id = int(retrieved_node_id.split('_')[1])
                    table_content = self.table_key_to_content[table_key]
                    table_title = table_content['title']
                    table_rows = table_content['text'].split('\n')
                    column_names = table_rows[0]
                    row_values = table_rows[1:][row_id]
                    target_text = table_title + ' [SEP] ' + column_names + ' [SEP] ' + row_values
                
                edge_text = f"{query_node_text} [SEP] {target_text}"
                edge_text_list.append(edge_text)
            
            edge_count_list.append(len(edge_text_list))
            edge_total_list.extend(edge_text_list)
            target_node_id_list.extend(retrieved_node_id_list)
            
        # predict scores with reranker
        model_input = [[nl_question, edge] for edge in edge_total_list]
        
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
            
        pred_scores = torch.tensor(reranking_scores)

        # decompose pred_scores by using edge_count_list
        start_idx = 0
        for source_rank, (query_node_id, query_node_score, query_node_type) in enumerate(topk_query_nodes):
            if query_node_type == 'table segment':
                target_node_type = 'passage'
            else:
                target_node_type = 'table segment'
            
            end_idx = start_idx + edge_count_list[source_rank]
            sorted_idx = torch.argsort(pred_scores[start_idx:end_idx], descending=True)
            for target_rank, idx in enumerate(sorted_idx):
                target_node_id = target_node_id_list[start_idx + idx]
                augment_type = f'node_augmentation'
                query_node_score = float(pred_scores[start_idx + idx])
                self.add_node(graph, query_node_type, query_node_id, target_node_id, query_node_score, augment_type, source_rank, target_rank)
                self.add_node(graph, target_node_type, target_node_id, query_node_id, query_node_score, augment_type, target_rank, source_rank)
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


    def find_bipartite_subgraph(self, question, bipartite_subgraph_candidates):
        bipartite_subgraph_candidate_list, table_id_to_row_id_to_linked_passage_ids = self.get_bipartite_subgraph_candidate_list(bipartite_subgraph_candidates)
        
        is_aggregate = self.llm_node_selector.detect_aggregation_query(question)
        if is_aggregate:
            selected_rows = self.llm_node_selector.select_row_wise(question, table_id_to_row_id_to_linked_passage_ids)
            if len(selected_rows) != 0:
                bipartite_subgraph_candidate_list = [{"table_segment_id":f"{table_id}_{row_id}", "linked_passage_ids": linked_passage_ids} for table_id, row_id, linked_passage_ids in selected_rows]

        table_segment_id_to_passage_id_list = self.llm_node_selector.select_passage_wise(question, bipartite_subgraph_candidate_list)
        
        for table_segment_id, passage_id_list in table_segment_id_to_passage_id_list.items():
            for passage_id in passage_id_list:
                self.add_node(bipartite_subgraph_candidates, 'table segment', table_segment_id, passage_id, self.cfg.max_edge_score, 'llm_selected')
                self.add_node(bipartite_subgraph_candidates, 'passage', passage_id, table_segment_id, self.cfg.max_edge_score, 'llm_selected')

    
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
            
        return bipartite_subgraph_candidate_list, table_id_to_row_id_to_linked_passage_ids

    def add_node(self, graph, source_node_type, source_node_id, target_node_id, score, retrieval_type, source_rank = 0, target_rank = 0):
        if source_node_id not in graph:
            graph[source_node_id] = {'type': source_node_type, 'linked_nodes': [[target_node_id, score, retrieval_type, source_rank, target_rank]]}
        else:
            graph[source_node_id]['linked_nodes'].append([target_node_id, score, retrieval_type, source_rank, target_rank])


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
    
    #@profile
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

if __name__ == "__main__":
    main()