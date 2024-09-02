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
        edge_contents = []
        with open(cfg.edge_dataset_path, "r") as file:
            for line in file:
                edge_contents.append(json.loads(line))
        num_of_edges = len(edge_contents)
            
        self.edge_key_to_content = {}
        print(f"Loading {num_of_edges} graphs...")

        for id, edge_content in enumerate(edge_contents):
            self.edge_key_to_content[edge_content['chunk_id']] = edge_content

        # load retrievers
        ## id mappings
        self.id_to_edge_key = json.load(open(cfg.edge_ids_path))
        
        ## colbert retrievers
        print(f"Loading index...")
        self.colbert_edge_retriever = Searcher(index=f"{cfg.edge_index_name}.nbits{cfg.nbits}", config=ColBERTConfig(), collection=cfg.collection_edge_path, index_root=cfg.edge_index_root_path)
        
        # load experimental settings
        self.top_k_of_edge = 5000 #cfg.top_k_of_edge

        self.node_scoring_method = cfg.node_scoring_method
        self.batch_size = cfg.batch_size

    def query(self, nl_question, retrieval_time = 2):
        
        # 1. Edge Retrieval
        top_k_to_retrieved_graph = {}
        retrieved_edges = self.retrieve_edges(nl_question)
        
        # for top_k in [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]:
        #     # 2. Graph Integration
        #     integrated_graph = self.integrate_graphs(retrieved_edges[:top_k])
        #     retrieval_type = None
        #     self.assign_scores(integrated_graph, retrieval_type)
        #     retrieved_graph = integrated_graph
        #     top_k_to_retrieved_graph[str(top_k)] = retrieved_graph

        return retrieved_edges
    
    def retrieve_edges(self, nl_question):
        
        retrieved_edges_info = self.colbert_edge_retriever.search(nl_question, 1000)
        
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
    
    retrieved_graph_path = "/mnt/sdd/shpark/edge_retrieval_v2_fine_tuned_bsize_512_1000.json"
    
    # load qa dataset
    print(f"Loading qa dataset...")
    ALL_TABLES_PATH = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    with open(ALL_TABLES_PATH) as f:
        all_tables = json.load(f)
    
    qa_dataset = json.load(open(cfg.qa_dataset_path))
    graph_query_engine = GraphQueryEngine(cfg)
    tokenizer = SimpleTokenizer()
    
    # query
    print(f"Start querying...")
    # Constants
    LIMITS = [1, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 1000]
    NUM_LIMITS = len(LIMITS)

    # Initialize recall arrays
    table_segment_answer_recall = [0] * NUM_LIMITS
    passage_answer_recall = [0] * NUM_LIMITS
    edge_answer_recall = [0] * NUM_LIMITS
    
    if os.path.exists(retrieved_graph_path):
        retrieved_graphs = json.load(open(retrieved_graph_path))
    else:
        retrieved_graphs = []
        for qidx, qa_datum in tqdm(enumerate(qa_dataset), total=len(qa_dataset)):
            
            nl_question = qa_datum['question']
            answers = qa_datum['answers']
            top_k_to_retrieved_graph = graph_query_engine.query(nl_question, retrieval_time = 2)
            retrieved_graphs.append(top_k_to_retrieved_graph)

        # save integrated graph
        print(f"Saving integrated graph...")
        json.dump(retrieved_graphs, open(retrieved_graph_path, 'w'))

    print(f"Start evaluating...")
    print(retrieved_graph_path)

    # Counters
    table_segment_count = 0
    passage_count = 0
    edge_count = 0

    # Preprocess qa_data to extract gold_table_segments and gold_passages
    qa_data_processed = []
    for qa_datum in qa_dataset:
        gt_chunk_id_to_row_info = {ctx['chunk_id']: ctx['rows'] for ctx in qa_datum['positive_ctxs'] if isinstance(ctx, dict) and 'chunk_id' in ctx and 'rows' in ctx}
        gold_table_segments = {
            f"{ctx['chunk_id']}_{node[1][0]}"
            for ctx in qa_datum['positive_ctxs']
            for node in ctx['answer_node']
            if isinstance(ctx, dict) and 'chunk_id' in ctx and 'answer_node' in ctx
        }
        gold_passages = [
            ans[2].replace('/wiki/', '').replace('_', ' ')
            for ctx in qa_datum['positive_ctxs']
            for ans in ctx['answer_node']
            if isinstance(ctx, dict) and 'answer_node' in ctx and ans[3] == 'passage'
        ]
        qa_data_processed.append((gt_chunk_id_to_row_info, gold_table_segments, gold_passages))
    # Main loop
    for (retrieved_graph, (gt_chunk_id_to_row_info, gold_table_segments, gold_passages)) in tqdm(zip(retrieved_graphs, qa_data_processed), total=len(retrieved_graphs)):
        if gold_passages:
            passage_count += 1

        exist_table = set()
        exist_passage = set()
        all_included_table_segment = []
        all_included_passage = []
        all_included_edge = []

        for edge in retrieved_graph[:1000]:
            if 'linked_entity_id' not in edge:
                continue

            table_id = edge['table_id']
            table_chunk_id = all_tables[table_id]['chunk_id']
            doc_id = edge['chunk_id']
            row_id = int(doc_id.split('_')[1])
            gt_row_info = gt_chunk_id_to_row_info.get(table_chunk_id, {})
            gt_row_id = gt_row_info[row_id] if isinstance(gt_row_info, list) and row_id < len(gt_row_info) else row_id
            retrieved_table_segment = f"{table_chunk_id}_{gt_row_id}"
            retrieved_passage = edge['linked_entity_id']

            if retrieved_table_segment not in exist_table:
                exist_table.add(retrieved_table_segment)
                has_answer = retrieved_table_segment in gold_table_segments
                all_included_table_segment.append(has_answer)

            if retrieved_passage not in exist_passage:
                exist_passage.add(retrieved_passage)
                has_answer = retrieved_passage in gold_passages
                all_included_passage.append(has_answer)

            if len(gold_passages) != 0:
                has_answer = retrieved_passage in gold_passages and retrieved_table_segment in gold_table_segments
            else:
                has_answer = retrieved_table_segment in gold_table_segments
            all_included_edge.append(has_answer)

        table_segment_count += 1
        edge_count += 1

        for l, limit in enumerate(LIMITS):
            if any(all_included_table_segment[:limit]):
                table_segment_answer_recall[l] += 1
            if gold_passages and any(all_included_passage[:limit]):
                passage_answer_recall[l] += 1
            if any(all_included_edge[:limit]):
                edge_answer_recall[l] += 1

    # Calculate and print recalls
    def print_recall(name, recall_array, count):
        for l, limit in enumerate(LIMITS):
            recall = recall_array[l] / count if count != 0 else 0
            print(f'{name} answer recall @ {limit}: {recall:.4f}')

    print_recall('Table segment', table_segment_answer_recall, table_segment_count)
    print_recall('Passage', passage_answer_recall, passage_count)
    print_recall('Edge', edge_answer_recall, edge_count)

if __name__ == "__main__":
    main()