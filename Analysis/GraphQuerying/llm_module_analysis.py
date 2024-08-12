import json
import copy
from tqdm import tqdm

generated_data_graphs_path = "/mnt/sdf/OTT-QAMountSpace/AnalysisResults/COS/DataGraphConstructor/table_chunks_to_passages_cos_table_passage.json"

with open(generated_data_graphs_path, 'r') as f:
    generated_data_graphs = json.load(f)

table_key_to_linked_nodes = {}
for generated_data_graph in tqdm(generated_data_graphs):
    table_key = generated_data_graph['table_chunk_id']
    if table_key not in table_key_to_linked_nodes:
        table_key_to_linked_nodes[table_key] = []
    
    for result in generated_data_graph['results']:
        table_key_to_linked_nodes[table_key].extend(result['retrieved'][:1])

llm_error_cases_path = "/mnt/sdf/OTT-QAMountSpace/AnalysisResults/Ours/GraphQuerier/error_cases/final_20_5.json"

with open(llm_error_cases_path, 'r') as f:
    llm_error_cases = json.load(f)

error_case = {"both": 0, "table": 0, "passage": 0, "none": 0}

top_k = 10
qid_list = []

for llm_error_case in llm_error_cases:
    positive_ctxs = llm_error_case['positive_ctxs']
    positive_table_segments = set()
    raw_positive_table_segments = set()
    positive_passages = set()
    positive_tables = set()
    for positive_ctx in positive_ctxs:
        chunk_id = positive_ctx['chunk_id']
        chunk_rows = positive_ctx['rows']
        positive_tables.add(chunk_id)
        for answer_node in positive_ctx['answer_node']:
            row_id = answer_node[1][0]
            chunk_row_id = chunk_rows.index(row_id)
            table_segment_id = f"{chunk_id}_{chunk_row_id}"
            raw_table_segment_id = f"{chunk_id}_{row_id}"
            positive_table_segments.add(table_segment_id)
            raw_positive_table_segments.add(raw_table_segment_id)
            if answer_node[3] == 'passage':
                passage_id = answer_node[2].replace('/wiki/','').replace('_', ' ')
                positive_passages.add(passage_id)
    
    
    filtered_retrieval_type = ['edge_reranking', "passage_node_augmentation_1"]
    filtered_retrieval_type_1 = ['edge_reranking']
    filtered_retrieval_type_2 = ["passage_node_augmentation_1"]
    retrieved_graph = llm_error_case['retrieved_graph']
    revised_retrieved_graph = {}
    for node_id, node_info in retrieved_graph.items():                  

        if node_info['type'] == 'table segment':
            linked_nodes = [x for x in node_info['linked_nodes'] 
                                if x[2] in filtered_retrieval_type and (x[2] in filtered_retrieval_type_2 and (x[3] < 10) and (x[4] < 2)) 
                                    or x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[4] < 1) and (x[3] < 1)) 
                                    or x[2] in filtered_retrieval_type_1#['edge_reranking', "llm_selected"]
                            ]
        elif node_info['type'] == 'passage':
            linked_nodes = [x for x in node_info['linked_nodes'] 
                                if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < 1) and (x[4] < 1)) 
                                or x[2] in filtered_retrieval_type and (x[2] in filtered_retrieval_type_2 and (x[4] < 10) and (x[3] < 2)) 
                                or x[2] in filtered_retrieval_type_1#['edge_reranking', "llm_selected"]
                            ]
        if len(linked_nodes) == 0: continue
        
        revised_retrieved_graph[node_id] = copy.deepcopy(node_info)
        revised_retrieved_graph[node_id]['linked_nodes'] = linked_nodes
        
        linked_scores = [linked_node[1] for linked_node in linked_nodes]
        node_score = max(linked_scores)
        revised_retrieved_graph[node_id]['score'] = node_score

    sorted_retrieved_graph = sorted(retrieved_graph.items(), key = lambda x: x[1]['score'], reverse = True)
    table_key_list = []
    passage_key_list = []
    table_key_to_augmented_nodes = {}
    
    for node_id, node_info in sorted_retrieved_graph:
        if node_info['type'] == 'table segment' and len(table_key_list) < top_k:
            table_key = node_info['chunk_id']
        else:
            continue
        if table_key not in table_key_list:
            table_key_list.append(table_key)
            table_key_to_augmented_nodes[table_key] = {}
        
        linked_passages = list(set([node_info[0] for node_info in node_info['linked_nodes']]))
        passage_key_list.extend(linked_passages)
        passage_key_list.extend(table_key_to_linked_nodes[table_key])
        table_key_to_augmented_nodes[table_key][node_id.split('_')[-1]] = linked_passages
    
    if positive_tables.intersection(set(table_key_list)) and positive_passages.intersection(set(passage_key_list)):
        error_case['both'] += 1
        qid_list.append(llm_error_case['id'])
    elif positive_tables.intersection(set(table_key_list)):
        error_case['table'] += 1
        # qid_list.append(llm_error_case['id'])
    elif positive_passages.intersection(set(passage_key_list)):
        error_case['passage'] += 1
        qid_list.append(llm_error_case['id'])
    else:
        error_case['none'] += 1

print(error_case)
json.dump(qid_list, open("qid_list.json", 'w'))
    