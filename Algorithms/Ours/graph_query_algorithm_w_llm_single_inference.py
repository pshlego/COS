import ast
import json
import time
import vllm
import hydra
import torch
from tqdm import tqdm
from pymongo import MongoClient
from omegaconf import DictConfig
from transformers import set_seed
from ColBERT.colbert import Searcher
from ColBERT.colbert.infra import ColBERTConfig
from FlagEmbedding import LayerWiseFlagLLMReranker
from Ours.dpr.data.qa_validation import has_answer
from Ours.dpr.utils.tokenizers import SimpleTokenizer
from prompts import select_nodes_prompt
# VLLM Parameters
COK_VLLM_TENSOR_PARALLEL_SIZE = 2 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
COK_VLLM_GPU_MEMORY_UTILIZATION = 0.5 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
set_seed(0)

class GraphQueryEngine:
    def __init__(self, cfg):
        # mongodb setup
        client = MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password))
        mongodb = client[cfg.dbname]

        # load dataset
        ## two node graphs
        edge_contents = mongodb[cfg.edge_name]
        num_of_edges = edge_contents.count_documents({})
        self.edge_key_to_content = {}
        self.table_key_to_edge_keys = {}
        print(f"Loading {num_of_edges} graphs...")
        for id, edge_content in tqdm(enumerate(edge_contents.find()), total=num_of_edges):
            self.edge_key_to_content[edge_content['chunk_id']] = edge_content
            
            if str(edge_content['table_id']) not in self.table_key_to_edge_keys:
                self.table_key_to_edge_keys[str(edge_content['table_id'])] = []

            self.table_key_to_edge_keys[str(edge_content['table_id'])].append(id)
        
        entity_linking_results = mongodb[cfg.entity_linking_result_name]
        num_of_entity_linking_results = entity_linking_results.count_documents({})
        self.table_chunk_id_to_linked_passages = {}
        print(f"Loading {num_of_entity_linking_results} entity linking results...")
        for entity_linking_content in tqdm(entity_linking_results.find(), total=num_of_entity_linking_results):
            self.table_chunk_id_to_linked_passages[entity_linking_content['table_chunk_id']] = entity_linking_content
        
        ## corpus
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
        self.colbert_edge_retriever = Searcher(index=f"{cfg.edge_index_name}.nbits{cfg.nbits}", config=ColBERTConfig(), collection=cfg.collection_edge_path, index_root=cfg.edge_index_root_path)
        self.colbert_table_retriever = Searcher(index=f"{cfg.table_index_name}.nbits{cfg.nbits}", config=ColBERTConfig(), collection=cfg.collection_table_path, index_root=cfg.table_index_root_path)
        self.colbert_passage_retriever = Searcher(index=f"{cfg.passage_index_name}.nbits{cfg.nbits}", config=ColBERTConfig(), collection=cfg.collection_passage_path, index_root=cfg.passage_index_root_path)
        self.cross_encoder_edge_retriever = LayerWiseFlagLLMReranker("/mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/Merged_BAAI_RERANKER_15_96_ckpt_400", use_fp16=True)
        
        ## large language model
        self.llm = vllm.LLM(
            cfg.llm_path,
            worker_use_ray=True,
            tensor_parallel_size=COK_VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=COK_VLLM_GPU_MEMORY_UTILIZATION, 
            trust_remote_code=True,
            dtype="half", # note: bfloat16 is not supported on nvidia-T4 GPUs
            max_model_len=5000, # input length + output length
            enforce_eager=True,
        )
        
        self.tokenizer = self.llm.get_tokenizer()
        
        ## prompt templates
        # self.select_table_segment_prompt = select_table_segment_prompt
        # self.select_passage_prompt = select_passage_prompt
        self.select_nodes_prompt = select_nodes_prompt
        
        # load experimental settings
        self.top_k_of_edge = cfg.top_k_of_edge
        self.top_k_of_table_segment_augmentation = cfg.top_k_of_table_segment_augmentation
        self.top_k_of_passage_augmentation = cfg.top_k_of_passage_augmentation
        self.top_k_of_table_segment = cfg.top_k_of_table_segment
        self.top_k_of_passage = cfg.top_k_of_passage
        self.top_k_of_table_select_w_llm = cfg.top_k_of_table_select_w_llm

        self.node_scoring_method = cfg.node_scoring_method
        self.batch_size = cfg.batch_size
    
    # @profile
    def query(self, nl_question, retrieval_time = 2):
        
        # 1. Edge Retrieval
        retrieved_edges = self.retrieve_edges(nl_question)
        
        # 2. Graph Integration
        integrated_graph = self.integrate_graphs(retrieved_edges)
        retrieval_type = None
        
        for i in range(retrieval_time):
            if i >= 1:
                self.reranking_edges(nl_question, integrated_graph)
                retrieval_type = 'edge_reranking'
                self.assign_scores(integrated_graph, retrieval_type)
            
            if i < retrieval_time:
                topk_table_segment_nodes = []
                topk_passage_nodes = []
                for node_id, node_info in integrated_graph.items():
                    if node_info['type'] == 'table segment':
                        topk_table_segment_nodes.append([node_id, node_info['score']])
                    elif node_info['type'] == 'passage':
                        topk_passage_nodes.append([node_id, node_info['score']])

                topk_table_segment_nodes = sorted(topk_table_segment_nodes, key=lambda x: x[1], reverse=True)[:self.top_k_of_table_segment_augmentation]
                topk_passage_nodes = sorted(topk_passage_nodes, key=lambda x: x[1], reverse=True)[:self.top_k_of_passage_augmentation]
                
                # 3.1 Passage Node Augmentation
                self.augment_node(integrated_graph, nl_question, topk_table_segment_nodes, 'table segment', 'passage', i)
                
                # 3.2 Table Segment Node Augmentation
                self.augment_node(integrated_graph, nl_question, topk_passage_nodes, 'passage', 'table segment', i)
                
                self.assign_scores(integrated_graph, retrieval_type)

            if i == retrieval_time - 1:
                table_key_list, deleted_node_id_list, table_key_to_augmented_nodes = self.get_table(integrated_graph)
                
                selected_node_info_list = self.select_nodes(nl_question, table_key_to_augmented_nodes)
                
                for selected_node_info in selected_node_info_list:
                    table_segment_node_id = selected_node_info['table_segment']
                    passage_id = selected_node_info['passage']
                    reranking_score = 10000
                    self.add_node(integrated_graph, 'table segment', table_segment_node_id, passage_id, reranking_score, 'llm_inference')
                    self.add_node(integrated_graph, 'passage', passage_id, table_segment_node_id, reranking_score, 'llm_inference')        
                
                # self.select_passages(nl_question, selected_table_segment_list, integrated_graph)
            
            self.assign_scores(integrated_graph, retrieval_type)
            retrieved_graphs = integrated_graph
        
        return retrieved_graphs
    
    # @profile
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
    
    # @profile
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
    
    # @profile
    def select_nodes(self, nl_question, table_key_to_augmented_nodes):
        prompt_list = []
        table_key_list = []
        table_id_to_linked_passage_ids = {}
        for table_key, table_segment_to_augmented_nodes in table_key_to_augmented_nodes.items():
            if table_key not in table_id_to_linked_passage_ids:
                table_id_to_linked_passage_ids[table_key] = {}
            
            linked_passage_info = self.table_chunk_id_to_linked_passages[self.table_key_to_content[table_key]['chunk_id']]
            table_title = linked_passage_info['question'].split(' [SEP] ')[0]
            table_column_name = linked_passage_info['question'].split(' [SEP] ')[-1].split('\n')[0]
            table_rows = linked_passage_info['question'].split(' [SEP] ')[-1].split('\n')[1:]
            table_rows = [row for row in table_rows if row != ""]
            table_info = {"title": table_title, "column_name": table_column_name, "rows": table_rows}
            linked_passages = linked_passage_info['results']
            
            row_id_to_linked_passage_contents = {}
            for linked_passage_info in linked_passages:
                row_id = str(linked_passage_info['row'])
                if row_id not in row_id_to_linked_passage_contents:
                    row_id_to_linked_passage_contents[row_id] = []
                
                if row_id not in table_id_to_linked_passage_ids[table_key]:
                    table_id_to_linked_passage_ids[table_key][row_id] = []
                
                try:
                    row_id_to_linked_passage_contents[row_id].append(self.passage_key_to_content[linked_passage_info['retrieved'][0]])
                    table_id_to_linked_passage_ids[table_key][row_id].append([linked_passage_info['retrieved'][0], 'entity_linking'])
                except:
                    continue
            
            for row_id, augmented_nodes in table_segment_to_augmented_nodes.items():
                if row_id not in row_id_to_linked_passage_contents:
                    row_id_to_linked_passage_contents[row_id] = []
                
                if row_id not in table_id_to_linked_passage_ids[table_key]:
                    table_id_to_linked_passage_ids[table_key][row_id] = []
                
                for augmented_node in list(set(augmented_nodes)):
                    try:
                        row_id_to_linked_passage_contents[row_id].append(self.passage_key_to_content[augmented_node])
                        table_id_to_linked_passage_ids[table_key][row_id].append([augmented_node, 'passage_node_augmentation'])
                    except:
                        continue
            
            table_and_linked_passages = self.get_table_and_linked_passages(table_info, row_id_to_linked_passage_contents)
            contents_for_prompt = {'question':nl_question, 'table_and_linked_passages': table_and_linked_passages}
            prompt = self.get_prompt(contents_for_prompt)
            prompt_list.append(prompt)
            table_key_list.append(table_key)
            
        responses = self.llm.generate(
                prompt_list,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.5,  # randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=64,  # Maximum number of tokens to generate per output sequence.
                    logprobs=1
                ),
                use_tqdm = False
            )
        
        selected_node_list = []
        for table_key, response in zip(table_key_list, responses):
            selected_nodes = response.outputs[0].text
            try:
                selected_nodes = ast.literal_eval(selected_nodes)
                for row_id, linked_passage_list in selected_nodes.items():
                    try:
                        row_id = row_id.split('_')[1]
                        row_id = str(int(row_id) - 1)
                    except:
                        continue
                    
                    if row_id not in table_id_to_linked_passage_ids[table_key]:
                        continue
                    
                    for passage_title in linked_passage_list:
                        if passage_title not in self.passage_key_to_content:
                            continue
                        selected_node_list.append({"table_segment": f"{table_key}_{row_id}", "passage": passage_title})
            except:
                continue
        
        return selected_node_list
    
    # @profile
    def select_passages(self, nl_question, selected_table_segment_list, integrated_graph):
        prompt_list = []
        table_segment_node_id_list = []
        for selected_table_segment in selected_table_segment_list:
            table_segment_node_id = selected_table_segment['table_segment_node_id']
            linked_passages = selected_table_segment['linked_passages']
            table_key = table_segment_node_id.split('_')[0]
            table = self.table_key_to_content[table_key]
            table_title = table['title']
            row_id = int(table_segment_node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            table_segment_text = column_name + '\n' + row_values
            table_segment_content = {"title": table_title, "content": table_segment_text}
            linked_passage_contents = []
            linked_passage_id_set = set()
            for linked_passage_info in linked_passages:
                linked_passage_id = linked_passage_info[0]
                if linked_passage_id in linked_passage_id_set:
                    continue
                linked_passage_contents.append({"title":linked_passage_id, "content": self.passage_key_to_content[linked_passage_id]['text']})
                linked_passage_id_set.add(linked_passage_id)
                
            graph = {"table_segment": table_segment_content, "linked_passages": linked_passage_contents}

            table_segment = graph['table_segment']
            table_segment_content = f"Table Title: {table_segment['title']}" + "\n" + table_segment['content'].replace(' , ', '[special tag]').replace(', ', ' | ').replace('[special tag]', ' , ')
            
            linked_passages = graph['linked_passages']
            linked_passage_contents = ""
            for linked_passage in linked_passages:
                title = linked_passage['title']
                content = linked_passage['content']
                tokenized_content = self.tokenizer.encode(content)
                trimmed_tokenized_content = tokenized_content[:128]
                trimmed_content = self.tokenizer.decode(trimmed_tokenized_content)
                linked_passage_contents += f"Title: {title}. Content: {trimmed_content}\n\n"
                
            contents_for_prompt = {"question": nl_question, "table_segment": table_segment_content, "linked_passages": linked_passage_contents}
            prompt = self.get_prompt(contents_for_prompt)
            prompt_list.append(prompt)
            table_segment_node_id_list.append(table_segment_node_id)
        
        responses = self.llm.generate(
                prompt_list,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.5,  # randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=128,  # Maximum number of tokens to generate per output sequence.
                    logprobs=1
                ),
                use_tqdm = False
            )
        
        for table_segment_node_id, response in zip(table_segment_node_id_list, responses):
            selected_passage_id_list = response.outputs[0].text
            try:
                selected_passage_id_list = ast.literal_eval(selected_passage_id_list)
                selected_passage_id_list = list(set(selected_passage_id_list))
            except:
                selected_passage_id_list = [selected_passage_id_list]
            
            for selected_passage_id in selected_passage_id_list:
                if selected_passage_id not in self.passage_key_to_content:
                    continue
                
                self.add_node(integrated_graph, 'table segment', table_segment_node_id, selected_passage_id, 1000000, 'llm_based_selection')
                self.add_node(integrated_graph, 'passage', selected_passage_id, table_segment_node_id, 1000000, 'llm_based_selection')
    
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

    # @profile
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

    def delete_nodes(self, graph, node_list):
        # Convert node_list to a set for O(1) average time complexity lookups
        node_set = set(node_list)

        # Remove nodes and update linked nodes in one pass
        nodes_to_delete = []

        for node_id in list(graph.keys()):  # list(graph.keys()) to avoid runtime errors during deletion
            if node_id in node_set:
                del graph[node_id]
            else:
                graph[node_id]['linked_nodes'] = [linked_node for linked_node in graph[node_id]['linked_nodes'] if linked_node[0] not in node_set]
                if not graph[node_id]['linked_nodes']:
                    nodes_to_delete.append(node_id)

        # Delete nodes that are not linked to any other node
        for node_id in nodes_to_delete:
            del graph[node_id]
        
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

    def get_table(self, retrieved_graph):
        sorted_retrieved_graph = sorted(retrieved_graph.items(), key=lambda x: x[1]['score'], reverse=True)
        
        table_key_list = []
        node_id_list = []
        table_key_to_augmented_nodes = {}
        for node_id, node_info in sorted_retrieved_graph:
            if node_info['type'] == 'table segment' and len(table_key_list) <= self.top_k_of_table_select_w_llm:
                table_key = node_id.split('_')[0]
                if table_key not in table_key_list:
                    table_key_list.append(table_key)
                    table_key_to_augmented_nodes[table_key] = {}
                
                table_key_to_augmented_nodes[table_key][node_id.split('_')[-1]] = [node_info[0] for node_info in node_info['linked_nodes'] if 'augmentation' in node_info[2]]
                
            if node_info['type'] == 'table segment' and node_id.split('_')[0] in table_key_list:
                node_id_list.append(node_id)

        return table_key_list, node_id_list, table_key_to_augmented_nodes
    
    def get_table_and_linked_passages(self, table_info, row_id_to_linked_passage_contents):
        table_and_linked_passages = ""
        table_and_linked_passages += f"Table Name: {table_info['title']}\n"
        table_and_linked_passages += f"Column Name: {table_info['column_name'].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ')}\n\n"
        for row_id, row_content in enumerate(table_info['rows']):
            table_and_linked_passages += f"Row_{row_id + 1}: {row_content.replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ')}\n"
            if str(row_id) in row_id_to_linked_passage_contents:
                table_and_linked_passages += f"Passages linked to Row_{row_id + 1}:\n"
                passage_set = set()
                for linked_passage in row_id_to_linked_passage_contents[str(row_id)]:
                    passage_title = linked_passage['title']
                    
                    if passage_title not in passage_set:
                        passage_set.add(passage_title)
                    else:
                        continue
                    
                    tokenized_content = self.tokenizer.encode(linked_passage['text'])
                    trimmed_tokenized_content = tokenized_content[:64]
                    trimmed_content = self.tokenizer.decode(trimmed_tokenized_content)
                    table_and_linked_passages += f"- {passage_title}: {trimmed_content}\n"
                    
            table_and_linked_passages += "\n\n"

        return table_and_linked_passages

    def get_prompt(self, contents_for_prompt):
        # if 'linked_passages' in contents_for_prompt:
        #     prompt = self.select_passage_prompt.format(**contents_for_prompt)
        # else:
        prompt = self.select_nodes_prompt.format(**contents_for_prompt)
        
        return prompt
    
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
    
@hydra.main(config_path="conf", config_name="graph_query_algorithm")
def main(cfg: DictConfig):
    # load qa dataset
    print(f"Loading qa dataset...")
    qa_dataset = json.load(open(cfg.qa_dataset_path))
    graph_query_engine = GraphQueryEngine(cfg)
    tokenizer = SimpleTokenizer()
    
    # query
    print(f"Start querying...")
    recall_list = []
    error_cases = []
    query_time_list = []
    retrieved_query_list = []
    for qidx, qa_datum in tqdm(enumerate(qa_dataset), total=len(qa_dataset)):
        
        nl_question = qa_datum['question']
        answers = qa_datum['answers']
        
        init_time = time.time()
        retrieved_graph = graph_query_engine.query(nl_question, retrieval_time = 2)
        end_time = time.time()
        query_time_list.append(end_time - init_time)
        
        # context = get_context(retrieved_graph, graph_query_engine)
        # is_has_answer = has_answer(answers, context, tokenizer, 'string', max_length=4096)

        # if is_has_answer:
        #     recall_list.append(1)
        # else:
        #     qa_datum['retrieved_graph'] = retrieved_graph
        #     if  "hard_negative_ctxs" in qa_datum:
        #         del qa_datum["hard_negative_ctxs"]
        #     error_cases.append(qa_datum)
        #     recall_list.append(0)
        
        retrieved_query_list.append(retrieved_graph)

    # print(f"HITS4K: {sum(recall_list) / len(recall_list)}")
    # print(f"Average query time: {sum(query_time_list) / len(query_time_list)}")
    
    # save integrated graph
    print(f"Saving integrated graph...")
    json.dump(retrieved_query_list, open(cfg.integrated_graph_save_path, 'w'))
    json.dump(query_time_list, open(cfg.query_time_save_path, 'w'))
    # json.dump(error_cases, open(cfg.error_cases_save_path, 'w'))

if __name__ == "__main__":
    main()