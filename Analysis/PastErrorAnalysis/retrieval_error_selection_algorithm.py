import json
from pymongo import MongoClient
from tqdm import tqdm
import numpy as np
def min_max_normalize(scores):
    min_val = np.min(scores)
    max_val = np.max(scores)
    if min_val == max_val:
        return np.ones(len(scores))
    normalized = (scores - min_val) / (max_val - min_val)
    return normalized.squeeze()
def get_linked_passages(raw_graph, selection_method, parameter):
    linked_passages = []
    for linked_passage in raw_graph['results']:
        if selection_method == 'topk':
            linked_passages.extend(linked_passage['retrieved'][:parameter])
        elif selection_method == 'topp':
            normalizes_score_list = min_max_normalize(linked_passage['scores'])
            acc_score = 0
            temperature = 1  # Consider removing if temperature is always 1, as it doesn't change the output
            exp_scores = np.exp(normalizes_score_list * temperature)
            softmax_scores = exp_scores / exp_scores.sum()  # More efficient softmax calculation
            for row_k, mention_linked_entity_id in enumerate(linked_passage['retrieved']):
                if acc_score <= parameter:
                    linked_passages.append(mention_linked_entity_id)
                else:
                    linked_passages.append(mention_linked_entity_id)
                    break
                acc_score += softmax_scores[row_k]
        elif selection_method == 'thr':
            normalizes_score_list = min_max_normalize(linked_passage['scores'])
            for row_k, mention_linked_entity_id in enumerate(linked_passage['retrieved']):
                if normalizes_score_list[row_k] >= parameter:
                    linked_passages.append(mention_linked_entity_id)
    return linked_passages
# MongoDB Connection Setup
username = "root"
password = "1234"
client = MongoClient(f"mongodb://{username}:{password}@localhost:27017/")
db = client["mydatabase"]
print("MongoDB Connected")

# Load Development Instances
dev_collection = db["ott_dev_q_to_tables_with_bm25neg"]
dev_instances = list(dev_collection.find())
print(f"Loaded {len(dev_instances)} development instances.")

# Load Graph Data
graph_collection = db["table_chunks_to_passages_cos_table_passage"]
raw_graphs = [graph for graph in graph_collection.find()]
print(f"Loaded {len(raw_graphs)} graphs.")

# Process Graphs
# TODO 각각의 selection method, parameter에 따라 어떤 passage를 link시킬 것인지가 달라져야 함
graphs = {}
# selection_method = 'thr'
# parameter = 1
# for raw_graph in tqdm(raw_graphs, desc="Processing graphs"):
#     chunk_id = '_'.join(raw_graph['table_chunk_id'].split('_')[:-1])
#     linked_passages = []
#     for linked_passage in raw_graph['results']:
#         if selection_method == 'topk':
#             linked_passages.extend(linked_passage['retrieved'][:parameter])
#         elif selection_method == 'topp':
#             normalizes_score_list = min_max_normalize(linked_passage['scores'])
#             acc_score = 0
#             temperature = 1  # Consider removing if temperature is always 1, as it doesn't change the output
#             exp_scores = np.exp(normalizes_score_list * temperature)
#             softmax_scores = exp_scores / exp_scores.sum()  # More efficient softmax calculation
#             for row_k, mention_linked_entity_id in enumerate(linked_passage['retrieved']):
#                 if acc_score <= parameter:
#                     linked_passages.append(mention_linked_entity_id)
#                 else:
#                     linked_passages.append(mention_linked_entity_id)
#                     break
#                 acc_score += softmax_scores[row_k]
#             graphs.setdefault(chunk_id, set()).update(tuple(linked_passages))
#         elif selection_method == 'thr':
#             normalizes_score_list = min_max_normalize(linked_passage['scores'])
#             for row_k, mention_linked_entity_id in enumerate(linked_passage['retrieved']):
#                 if normalizes_score_list[row_k] >= parameter:
#                     linked_passages.append(mention_linked_entity_id)
# Analysis Settings
hierarchical_levels = ['star', 'edge', 'both'] # ['star', 'edge']
is_colbert, search_space = True, None # True, larger
for is_colbert in [True]:
    if is_colbert:
        search_space_list = ['larger']
    else:
        search_space_list = [None]
    # Analyze Error Instances for Each Hierarchical Level
    for selection_method in ['thr', 'topp', 'topk']:#, 'larger'
        if selection_method == 'thr':
            parameters = [0.875, 0.85, 0.825, 0.85, 0.7, 0.5, 0.3, 0.1] # 0.975, 0.95, 0.925, 0.9
            # parameters = [60.94, 56.85, 54.61, 52.75]
            # parameters = [60.94, 56.85, 54.61, ]
        elif selection_method == 'topp':
            parameters = [0.9, 0.7, 0.5, 0.3, 0.1]
            # parameters = [0.9]
        elif selection_method == 'topk':
            parameters = [1, 2, 3, 4, 5]
            # parameters = [1]
        for parameter in parameters:
            graphs = {}
            for raw_graph in tqdm(raw_graphs, desc="Processing graphs"):
                chunk_id = '_'.join(raw_graph['table_chunk_id'].split('_')[:-1])
                linked_passages = get_linked_passages(raw_graph, selection_method, parameter)
                graphs.setdefault(chunk_id, set()).update(tuple(linked_passages))
            parameter_name = str(parameter).replace('.', '_')
            print(f"Analyzing {selection_method} selection method and {parameter} parameter")
            prefix = "colbert_" if is_colbert else ""
            
            if selection_method == 'topk':
                query_results_path = f"/home/shpark/mnt_sdc/shpark/query_result/colbert/selection_algorithms/topk/colbert_graph_query_results_fix_table_error_k_700_length_512_filtered_both_larger_topk_{parameter_name}.json"
            else:
                query_results_path = f"/home/shpark/mnt_sdc/shpark/query_result/colbert/normalize/colbert_graph_query_results_fix_table_error_k_700_length_512_filtered_both_larger_{selection_method}_{parameter_name}.json"
            base_path = f'/home/shpark/COS/error_analysis/results_all/selection_algorithms/{prefix}{selection_method}_{parameter_name}'
            
            with open(query_results_path, 'r') as file:
                query_results = json.load(file)

            error_instances, error_instances_ids, data_error_instances, table_instances, passage_instances, table_passage_instances = [], [], [], [], [], []
            
            for idx, instance in enumerate(tqdm(dev_instances, desc=f"Analyzing {parameter} level")):
                if not any(ctx["has_answer"] for ctx in query_results[idx]["ctxs"][:100]):
                    error_instances.append(query_results[idx])
                    error_instances_ids.append(idx)
            print(f"Total Error for {parameter}: {len(error_instances)}")

            for idx in error_instances_ids:
                instance = dev_instances[idx]
                gold_passage_set = {gold_passage for positive_ctx in instance['positive_ctxs'] for gold_passage in positive_ctx['target_pasg_titles'] if gold_passage is not None}
                # linked_passage_set = get_linked_passages(raw_graphs[instance['positive_table']], selection_method, parameter)
                linked_passage_set = graphs[instance['positive_table']]

                if gold_passage_set.isdisjoint(linked_passage_set) and len(list(gold_passage_set)) > 0:
                    data_error_instances.append(query_results[idx])
                    error_instance = query_results[idx]
                else:
                    if len(list(gold_passage_set)) == 0:
                        table_instances.append(query_results[idx])
                    else:
                        gold_table = any('_'.join(ctx['table_name'].split('_')[:-1]) == instance['positive_table'] for ctx in query_results[idx]["ctxs"][:100])
                        if gold_table:
                            passage_instances.append(query_results[idx])
                        else:
                            table_passage_instances.append(query_results[idx])

            print(f"Pseudo Data Graph Construction Error for {parameter}: {len(data_error_instances)}")

            # Save Results
            for name, data in [("error_instances", error_instances), ("data_error_instances", data_error_instances),
                            ("table_error_instances", table_instances), ("passage_error_instances", passage_instances),
                            ("table_passage_error_instances", table_passage_instances)]:
                with open(f"{base_path}_{name}.json", 'w') as file:
                    json.dump(data, file, indent=4)

            print(f"{parameter}: Tables - {len(table_instances)}, Passages - {len(passage_instances)}, Table/Passages - {len(table_passage_instances)}")
