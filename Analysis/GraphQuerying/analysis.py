import json

# with open('/home/shpark/OTT_QA_Workspace/passage_error_cases_passage_10_2_short.json', 'r') as f:
#     passage_error_cases_passage_10_2 = json.load(f)

with open('/home/shpark/OTT_QA_Workspace/passage_error_cases_passage_10_2.json', 'r') as f:
    passage_error_cases_passage_10_2 = json.load(f)

# with open('/home/shpark/OTT_QA_Workspace/both_error_cases_passage_10_2_short.json', 'r') as f:
#     both_error_cases_passage_10_2 = json.load(f)

# with open('/home/shpark/OTT_QA_Workspace/both_error_cases_passage_10_2.json', 'r') as f:
#     both_error_cases_passage_10_2_original = json.load(f)

# with open('/home/shpark/OTT_QA_Workspace/table_error_cases_passage_10_2_short.json', 'r') as f:
#     table_error_cases_passage_10_2 = json.load(f)

with open('/home/shpark/OTT_QA_Workspace/Analysis/GraphQueryResults/data_graph_error_cases.json', 'r') as f:
    data_graph_error_cases = json.load(f)

error_case_wo_data_graph_errors = list(set(passage_error_cases_passage_10_2.keys())-set([data_graph_error_case['id'] for data_graph_error_case in data_graph_error_cases]))
rank_histogram = {}
data_graph_error_cases = []
non_data_graph_error_cases = []
qid_list = []
for qid, error_case in passage_error_cases_passage_10_2.items():
    rank_list = []
    
    if qid in error_case_wo_data_graph_errors:
        non_data_graph_error_cases.append(error_case)
    else:
        data_graph_error_cases.append(error_case)
    
    table_segment_list = [f"{node[1]['chunk_id']}_{node[0].split('_')[-1]}" for node in error_case['sorted_retrieved_graph'] if 'chunk_id' in node[1]]
    for positive_table_segment in error_case['positive_table_segments']:
        try:
            rank = table_segment_list.index(positive_table_segment)
        except:
            continue
        rank_list.append(rank)
        
    min_rank = min(rank_list)
    if min_rank == 0 and qid not in error_case_wo_data_graph_errors:
        qid_list.append(qid)
    if min_rank in rank_histogram:
        rank_histogram[min_rank] += 1
    else:
        rank_histogram[min_rank] = 1
        
sorted_dict = dict(sorted(rank_histogram.items()))
print(qid_list)
print(sorted_dict)
print(len(data_graph_error_cases))
print(len(non_data_graph_error_cases))
# += 1