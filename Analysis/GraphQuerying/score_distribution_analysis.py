import json
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# Load table data
table_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
table_contents = json.load(open(table_data_path))
table_key_to_content = {}
for table_key, table_content in tqdm(enumerate(table_contents)):
    table_key_to_content[str(table_key)] = table_content

# Load query results
data_graph_error_case = json.load(open("/home/shpark/OTT_QA_Workspace/data_graph_error_case.json"))
current_error_case = json.load(open("/mnt/sdf/OTT-QAMountSpace/AnalysisResults/Ours/GraphQuerier/error_cases/missing_link_3_3.json"))
# corrected_qid_list = json.load(open("/home/shpark/OTT_QA_Workspace/2_5.json"))
current_error_case_id_list = [x['id'] for x in current_error_case]
data_graph_error_case_id_list = [x['id'] for x in data_graph_error_case]
query_results_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/missing_link_3_3.jsonl"#"/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/expanded_query_retrieval_v2.jsonl"
data = read_jsonl(query_results_path)

positive_score_dist_of_gold_tables = []
total_score_dist_of_gold_tables = []
qid_list = []
for datum in tqdm(data):
    qa_data = datum["qa data"]
    if qa_data['id'] not in data_graph_error_case_id_list:
        continue
    if qa_data['id'] not in current_error_case_id_list:
        continue
    # if qa_data['id'] not in corrected_qid_list:
    #     continue
    retrieved_graph = datum["retrieved graph"]
    positive_ctxs = qa_data['positive_ctxs']
    positive_table_segments = set()
    positive_passages = set()

    for positive_ctx in positive_ctxs:
        chunk_id = positive_ctx['chunk_id']
        chunk_rows = positive_ctx['rows']
        for answer_node in positive_ctx['answer_node']:
            row_id = answer_node[1][0]
            chunk_row_id = chunk_rows.index(row_id)
            table_segment_id = f"{chunk_id}_{chunk_row_id}"
            positive_table_segments.add(table_segment_id)
            if answer_node[3] == 'passage':
                passage_id = answer_node[2].replace('/wiki/', '').replace('_', ' ')
                positive_passages.add(passage_id)

    revised_retrieved_graph = {}
    for node_id, node_info in retrieved_graph.items():
        linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in ['edge_and_table_retrieval','table_to_table_segment']]  # Filtering condition here
        if len(linked_nodes) == 0: continue
        revised_retrieved_graph[node_id] = copy.deepcopy(node_info)
        linked_scores = [linked_node[1] for linked_node in linked_nodes]
        node_score = max(linked_scores)
        revised_retrieved_graph[node_id]['score'] = node_score

    if revised_retrieved_graph:
        scores = np.array([info['score'] for info in revised_retrieved_graph.values()])
        min_score = scores.min()
        max_score = scores.max()
        for node_id in revised_retrieved_graph:
            if max_score - min_score > 0:
                revised_retrieved_graph[node_id]['score'] = (revised_retrieved_graph[node_id]['score'] - min_score) / (max_score - min_score)
            else:
                revised_retrieved_graph[node_id]['score'] = 0

    sorted_retrieved_graph = sorted(revised_retrieved_graph.items(), key=lambda x: x[1]['score'], reverse=True)
    rank = 0
    for node_id, node_info in sorted_retrieved_graph:
        if node_info['type'] == 'table segment':
            table_id = node_id.split('_')[0]
            row_id = node_id.split('_')[1]
            chunk_id = table_key_to_content[table_id]['chunk_id']
            if f"{chunk_id}_{row_id}" in positive_table_segments:
                if rank < 5:
                    qid_list.append(qa_data['id'])
                # if rank > 50:
                #     continue
                positive_score_dist_of_gold_tables.append(rank)
                break
            rank += 1
            total_score_dist_of_gold_tables.append(node_info['score'])
json.dump(qid_list, open("/home/shpark/OTT_QA_Workspace/qid_list.json", 'w'), indent=2)
# Plot the score distributions
# plt.figure(figsize=(10, 6))
# # plt.hist(total_score_dist_of_gold_tables, bins=120, label='Total Score Distribution', alpha=0.5, color='red')
# plt.hist(positive_score_dist_of_gold_tables, bins=120, label='Positive Score Distribution', color='blue')
# plt.xlabel('Scaled Score')
# plt.ylabel('Frequency')
# plt.title('Scaled Score Distribution of Positive Table Segments vs Total Score Distribution')
# plt.legend()

# # Save the plot as a PNG file
# output_file = "/home/shpark/OTT_QA_Workspace/scaled_rank_distribution_of_gold_tables.png"
# plt.savefig(output_file)
# plt.close()

# print(f"Plot saved as {output_file}")
