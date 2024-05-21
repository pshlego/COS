import json
import time
import hydra
from tqdm import tqdm
from pymongo import MongoClient
from omegaconf import DictConfig
from ColBERT.colbert import Searcher
from ColBERT.colbert.infra import ColBERTConfig
from Ours.dpr.data.qa_validation import has_answer
from Ours.dpr.utils.tokenizers import SimpleTokenizer

class GraphQueryEngine:
    def __init__(self, cfg):
        # mongodb setup
        client = MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password))
        mongodb = client[cfg.dbname]

        # load dataset
        ## two node graphs
        two_node_graph_contents = mongodb[cfg.two_node_graph_name]
        num_of_two_node_graphs = two_node_graph_contents.count_documents({})
        self.two_node_graph_key_to_content = {}
        print(f"Loading {num_of_two_node_graphs} graphs...")
        for two_node_graph_content in tqdm(two_node_graph_contents.find(), total=num_of_two_node_graphs):
            self.two_node_graph_key_to_content[two_node_graph_content['chunk_id']] = two_node_graph_content

        ## corpus
        print(f"Loading corpus...")
        self.table_key_to_content = {}
        table_contents = json.load(open(cfg.table_data_path))
        for table_key, table_content in enumerate(table_contents):
            self.table_key_to_content[str(table_key)] = table_content
        
        self.passage_key_to_content = {}
        passage_contents = json.load(open(cfg.passage_data_path))
        for passage_content in passage_contents:
            self.passage_key_to_content[passage_content['title']] = passage_content

        # load retrievers
        ## id mappings
        self.id_to_two_node_graph_key = json.load(open(cfg.two_node_graph_ids_path))
        self.id_to_table_key = json.load(open(cfg.table_ids_path))
        self.id_to_passage_key = json.load(open(cfg.passage_ids_path))
        
        ## colbert retrievers
        two_node_graph_config = ColBERTConfig(root=cfg.collection_two_node_graph_root_dir_path)
        table_config = ColBERTConfig(root=cfg.collection_table_root_dir_path)
        passage_config = ColBERTConfig(root=cfg.collection_passage_root_dir_path)

        two_node_graph_index_name = cfg.two_node_graph_index_name
        table_index_name = cfg.table_index_name
        passage_index_name = cfg.passage_index_name

        print(f"Loading index...")
        self.colbert_two_node_graph_retriever = Searcher(index=f"{two_node_graph_index_name}.nbits{cfg.nbits}", config=two_node_graph_config, index_root=cfg.two_node_graph_index_root_path)
        self.colbert_table_retriever = Searcher(index=f"{table_index_name}.nbits{cfg.nbits}", config=table_config, index_root=cfg.table_index_root_path)
        self.colbert_passage_retriever = Searcher(index=f"{passage_index_name}.nbits{cfg.nbits}", config=passage_config, index_root=cfg.passage_index_root_path)
        
        # load experimental settings
        self.top_k_of_two_node_graph = cfg.top_k_of_two_node_graph
        self.top_k_of_table_augmentation = cfg.top_k_of_table_augmentation
        self.top_k_of_passage_augmentation = cfg.top_k_of_passage_augmentation
        self.top_k_of_table = cfg.top_k_of_table
        self.top_k_of_passage = cfg.top_k_of_passage

        self.node_scoring_method = cfg.node_scoring_method

    def query(self, nl_question):
        # 1. two-node graph retrieval
        retrieved_two_node_graphs = self.retrieve_two_node_graphs(nl_question)
        
        # 2. Graph Integration
        integrated_graph = self.integrate_graphs(retrieved_two_node_graphs)

        topk_table_segment_nodes = []
        topk_passage_nodes = []
        for node_id, node_info in integrated_graph.items():
            if node_info['type'] == 'table segment':
                topk_table_segment_nodes.append([node_id, node_info['score']])
            elif node_info['type'] == 'passage':
                topk_passage_nodes.append([node_id, node_info['score']])

        topk_table_segment_nodes = sorted(topk_table_segment_nodes, key=lambda x: x[1], reverse=True)[:self.top_k_of_table_augmentation]
        topk_passage_nodes = sorted(topk_passage_nodes, key=lambda x: x[1], reverse=True)[:self.top_k_of_passage_augmentation]
        
        # 3.1 Passage Node Augmentation
        self.augment_node(integrated_graph, nl_question, topk_table_segment_nodes, 'table segment', 'passage')
        
        # 3.2 Table Segment Node Augmentation
        self.augment_node(integrated_graph, nl_question, topk_passage_nodes, 'passage', 'table segment')
        
        self.assign_scores(integrated_graph)
        retrieved_graphs = integrated_graph
        
        return retrieved_graphs
    
    def retrieve_two_node_graphs(self, nl_question):
        retrieved_two_node_graphs_info = self.colbert_two_node_graph_retriever.search(nl_question, 10000)
        retrieved_two_node_graph_id_list = retrieved_two_node_graphs_info[0]
        retrieved_two_node_graph_score_list = retrieved_two_node_graphs_info[2]
        
        retrieved_two_node_graph_contents = []
        for graphidx, retrieved_id in enumerate(retrieved_two_node_graph_id_list[:self.top_k_of_two_node_graph]):
            retrieved_two_node_graph_content = self.two_node_graph_key_to_content[self.id_to_two_node_graph_key[str(retrieved_id)]]

            # pass single node graph
            if 'linked_entity_id' not in retrieved_two_node_graph_content:
                continue
            
            retrieved_two_node_graph_content['two_node_graph_score'] = retrieved_two_node_graph_score_list[graphidx]
            retrieved_two_node_graph_contents.append(retrieved_two_node_graph_content)

        return retrieved_two_node_graph_contents
    
    def integrate_graphs(self, retrieved_graphs):
        integrated_graph = {}
        retrieval_type='two_node_graph_retrieval'
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
            two_node_graph_score = retrieved_graph_content['two_node_graph_score']
            
            # add nodes
            self.add_node(integrated_graph, 'table segment', table_segment_node_id, passage_id, two_node_graph_score, retrieval_type)
            self.add_node(integrated_graph, 'passage', passage_id, table_segment_node_id, two_node_graph_score, retrieval_type)

            # node scoring
            self.assign_scores(integrated_graph)

        return integrated_graph

    def augment_node(self, graph, nl_question, topk_query_nodes, query_node_type, retrieved_node_type):
        for query_node_id, query_node_score in topk_query_nodes:
            expanded_query = self.get_expanded_query(nl_question, query_node_id, query_node_type)
            
            if query_node_type == 'table segment':
                retrieved_node_info = self.colbert_passage_retriever.search(expanded_query, self.top_k_of_passage)
            else:
                retrieved_node_info = self.colbert_table_retriever.search(expanded_query, self.top_k_of_passage)
            
            retrieved_id_list = retrieved_node_info[0]
            retrieved_score_list = retrieved_node_info[2]
            
            for pid, retrieved_id in enumerate(retrieved_id_list):
                if query_node_type == 'table segment':
                    retrieved_node_id = self.id_to_passage_key[str(retrieved_id)]
                    retrieval_type = 'passage_node_augmentation'
                else:
                    retrieved_node_id = self.id_to_table_key[str(retrieved_id)]
                    retrieval_type = 'table_segment_node_augmentation'
                
                self.add_node(graph, query_node_type, query_node_id, retrieved_node_id, query_node_score, retrieval_type)
                self.add_node(graph, retrieved_node_type, retrieved_node_id, query_node_id, query_node_score, retrieval_type)

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

    def add_node(self, graph, source_node_type, source_node_id, target_node_id, score, retrieval_type):
        # add source and target node
        if source_node_id not in graph:
            graph[source_node_id] = {'type': source_node_type, 'linked_nodes': [[target_node_id, score, retrieval_type]]}
        # add target node
        else:
            graph[source_node_id]['linked_nodes'].append([target_node_id, score, retrieval_type])
    
    def assign_scores(self, graph):
        for node_id, node_info in graph.items():

            if 'score' in node_info:
                continue

            linked_scores = [linked_node[1] for linked_node in node_info['linked_nodes']]
            
            if self.node_scoring_method == 'min':
                node_score = min(linked_scores)
            elif self.node_scoring_method == 'max':
                node_score = max(linked_scores)
            elif self.node_scoring_method == 'mean':
                node_score = sum(linked_scores) / len(linked_scores)
            
            graph[node_id]['score'] = node_score
            
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
        
            max_linked_node_id, max_score, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default=(None, 0, 'two_node_graph_retrieval'))

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
            
            two_node_graph_text = table_segment_text + '\n' + passage_text
            context += two_node_graph_text
            
        elif node_info['type'] == 'passage':
            if node_id in retrieved_passage_set:
                continue
            
            retrieved_passage_set.add(node_id)
            passage_content = graph_query_engine.passage_key_to_content[node_id]
            passage_text = passage_content['title'] + ' ' + passage_content['text']
            
            max_linked_node_id, max_score, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default=(None, 0, 'two_node_graph_retrieval'))
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
            
            two_node_graph_text = table_segment_text + '\n' + passage_text
            context += two_node_graph_text

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
        retrieved_graph = graph_query_engine.query(nl_question)
        end_time = time.time()
        query_time_list.append(end_time - init_time)
        
        context = get_context(retrieved_graph, graph_query_engine)
        is_has_answer = has_answer(answers, context, tokenizer, 'string', max_length=4096)

        if is_has_answer:
            recall_list.append(1)
        else:
            qa_datum['retrieved_graph'] = retrieved_graph
            error_cases.append(qa_datum)
            recall_list.append(0)
        
        retrieved_query_list.append(retrieved_graph)

    print(f"HITS4K: {sum(recall_list) / len(recall_list)}")
    print(f"Average query time: {sum(query_time_list) / len(query_time_list)}")
    
    # save integrated graph
    print(f"Saving integrated graph...")
    json.dump(retrieved_query_list, open(cfg.integrated_graph_save_path, 'w'))
    json.dump(query_time_list, open(cfg.query_time_save_path, 'w'))
    json.dump(error_cases, open(cfg.error_cases_save_path, 'w'))
if __name__ == "__main__":
    main()