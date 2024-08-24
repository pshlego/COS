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
        edge_contents = mongodb["table_chunks_to_passages_cos_table_passage"]
        num_of_edges = edge_contents.count_documents({})
        self.edge_key_to_content = {}
        self.table_key_to_edge_keys = {}
        print(f"Loading {num_of_edges} graphs...")
        for id, edge_content in tqdm(enumerate(edge_contents.find()), total=num_of_edges):
            self.edge_key_to_content[edge_content['table_chunk_id']] = edge_content
            
            # if str(edge_content['table_id']) not in self.table_key_to_edge_keys:
            #     self.table_key_to_edge_keys[str(edge_content['table_id'])] = []

            # self.table_key_to_edge_keys[str(edge_content['table_id'])].append(id)

        print("4. Loading passages...")
        self.passage_key_to_content = {}
        passage_contents = json.load(open(cfg.passage_data_path))
        print("4. Loaded " + str(len(passage_contents)) + " passages!")
        print("4. Processing passages...")
        for passage_content in tqdm(passage_contents):
            self.passage_key_to_content[passage_content['title']] = passage_content
        print("4. Processing passages complete!", end = "\n\n")
        
        # load retrievers
        ## id mappings
        self.id_to_edge_key = json.load(open(cfg.edge_ids_path))
        self.id_to_passage_key = json.load(open(cfg.passage_ids_path))
        # self.passage_key_to_id = json.load(open(cfg.passage_key_to_ids_path))
        
        ## colbert retrievers
        print(f"Loading index...")
        #self.colbert_edge_retriever = Searcher(index=f"{cfg.edge_index_name}.nbits{cfg.nbits}", config=ColBERTConfig(), collection=cfg.collection_edge_path, index_root=cfg.edge_index_root_path, checkpoint=cfg.edge_checkpoint_path)
        self.colbert_passage_retriever = Searcher(index=f"{cfg.passage_index_name}.nbits{cfg.nbits}", config=ColBERTConfig(), collection=cfg.collection_passage_path, index_root=cfg.passage_index_root_path, checkpoint=cfg.passage_checkpoint_path)
        
        # load experimental settings
        self.top_k_of_passage = 5 #cfg.top_k_of_edge
            
        self.node_scoring_method = cfg.node_scoring_method
        # self.batch_size = cfg.batch_size
    def query(self, nl_question, table, row_id, topk, positive_passages, retrieval_time = 2):
        chunk_id = f"{table['chunk_id']}"
        # entity_linking_result = self.edge_key_to_content[chunk_id]
        linked_passage_list = []
        # for mention_info in entity_linking_result['results']:
        #     row = mention_info['row']
        #     if str(row_id) == str(row):
        #         linked_passage_list.extend(mention_info['retrieved'][1:topk])
        # linked_passage_list = []
        # 1. Edge Retrieval
        expanded_query = self.get_expanded_query(nl_question, table, row_id)

        retrieved_node_info = self.colbert_passage_retriever.search(expanded_query, self.top_k_of_passage)#, filter_fn=filter_fn, pid_deleted_list=pid_deleted_list)
        retrieved_id_list = retrieved_node_info[0]
        retrieved_score_list = retrieved_node_info[2]
        top_k = self.top_k_of_passage

        retrieved_passage_list = []
        
        for target_rank, retrieved_id in enumerate(retrieved_id_list):
            retrieved_node_id = self.id_to_passage_key[str(retrieved_id)]
            retrieved_passage_list.append(retrieved_node_id)
    
        retrieved_passage_list = retrieved_passage_list + linked_passage_list
        retrieved_passage_list = list(set(retrieved_passage_list))
        
        # table_title = table['title']
        # table_column_names = table['text'].split('\n')[0]
        # table_row_values = table['text'].split('\n')[row_id+1]
        # table_text = f"{table_title} [SEP] {table_column_names} [SEP] {table_row_values}"
        
        # edge_text_list = []
        # for retrieved_passage_id in retrieved_passage_list:
        #     passage_content = self.passage_key_to_content[retrieved_passage_id]
        #     passage_text = f"{passage_content['title']} [SEP] {passage_content['text']}"
        #     edge_text = f"{table_text} [SEP] {passage_text}"
        #     edge_text_list.append(edge_text)
        # if edge_text_list == []:
        #     return []
        # edges = self.colbert_edge_retriever.checkpoint.doc_tokenizer.tensorize(edge_text_list)
        # queries = self.colbert_edge_retriever.checkpoint.query_tokenizer.tensorize([nl_question])
        # encoded_Q = self.colbert_edge_retriever.checkpoint.query(*queries)
        # Q_duplicated = encoded_Q.repeat_interleave(len(edge_text_list), dim=0).contiguous()
        # encoded_D, encoded_D_mask = self.colbert_edge_retriever.checkpoint.doc(*edges, keep_dims='return_mask')
        # pred_scores = self.colbert_edge_retriever.checkpoint.score(Q_duplicated, encoded_D, encoded_D_mask)
        # retrieved_passage_list = [retrieved_passage_list[i] for i in torch.argsort(pred_scores, descending=True)]#[:top_k]
        # entity
        return retrieved_passage_list
    
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

def filter_fn(pid, values_to_remove):
    return pid[~torch.isin(pid, values_to_remove)].to("cuda")

@hydra.main(config_path="conf", config_name="graph_query_algorithm")
def main(cfg: DictConfig):
    # load qa dataset
    print(f"Loading qa dataset...")
    ALL_TABLES_PATH = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    with open(ALL_TABLES_PATH) as f:
        all_tables = json.load(f)
    
    # with open(f"2_5.json") as f:
    #     qid_list = json.load(f)
    
    table_chunk_to_table = {}
    for table_info in all_tables:
        table_chunk_to_table[table_info['chunk_id']] = table_info
    
    qa_dataset = json.load(open("/home/shpark/OTT_QA_Workspace/data_graph_error_case.json"))
    graph_query_engine = GraphQueryEngine(cfg)
    
    path = "/home/shpark/OTT_QA_Workspace/error_cases.json"
    # positive_id_list = json.load(open("/home/shpark/OTT_QA_Workspace/positive_id_list.json"))
    # qid_list = json.load(open(path))
    qid_list = []
    print(f"Start querying...")
    for topk in [5]:
        for top_k in [5]:
            # for top_k_linked_passage in [5]:
            # Initialize recall arrays
            graph_query_engine.top_k_of_passage = top_k
            answer_recall_list = []
            linked_passage_len_list = []
            missing_link_predict_len_list = []
            for qidx, qa_datum in tqdm(enumerate(qa_dataset), total=len(qa_dataset)):
                # if qa_datum['id'] not in positive_id_list:
                #     continue
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
                        # if "Whangarei" in nl_question:
                        retrieved_nodes = graph_query_engine.query(nl_question, table, row_id, topk, positive_passages, retrieval_time = 2)
                        # linked_passage_len_list.append(len(linked_passage_list))
                        missing_link_predict_len_list.append(len(set(retrieved_nodes)))
                        retrieved_node_list.extend(retrieved_nodes)
                        # else:
                        #     continue

                if set(positive_passages).intersection(set(retrieved_node_list)) == set():
                    answer_recall_list.append(0)
                else:
                    answer_recall_list.append(1)
                    qid_list.append(qa_datum['id'])
                    
            # save integrated graph
            # print(f"Saving integrated graph...")
            json.dump(qid_list, open(f"/home/shpark/OTT_QA_Workspace/expanded_query_retrieval_id_list.json", 'w'))

            print(f"Start evaluating...")
            print(f"missing_link_predict_len_list: {sum(missing_link_predict_len_list) / len(missing_link_predict_len_list)}")
            print(f"Entity Linking Top k: {topk}, Passage Augmentation Top K: {top_k}, Answer Recall: {sum(answer_recall_list) / len(answer_recall_list)}")
    # print(f"Topk: {5}, Linked Passage Len: {sum(linked_passage_len_list) / len(linked_passage_len_list)}")

if __name__ == "__main__":
    main()
# 1. dev에서 table segment와 passage가 graph상에 연결되지 않은 경우를 확인
# 2. 해당 경우에 대해 gold table segment와 질의를 합쳐 passage를 검색했을 때 검색되는 경우를 확인
# 3. 검색되는 경우가 전체에서 얼마를 차지하는지 파악

#1: 0.33852140077821014
#2: 0.5525291828793775
#3: 0.6653696498054474
#4: 0.7159533073929961
#5: 0.754863813229572
#10: 0.8015564202334631
#100: 

# 2

# entity linking으로 대체하기 어려운 짏의
#'0eb492f852610b99'



#Entity Linking:'f6efe2fdaeea542c' (4), '8c10f7945508d534' (1), 'e96cc27e674c7730' (3)
#expanded query retrieval: '89fbb12ab4b204a7' (4), '4f932f300d308528' (4), '4f932f300d308528' (5), '476b24788b353888' (4), 'ac65c17e2b408a53' (3), 'e96cc27e674c7730' (4)