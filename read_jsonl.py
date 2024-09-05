import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

path = "/home/shpark/OTT_QA_Workspace/data_graph_error_case.json"

data = json.load(open(path, 'r', encoding='utf-8'))

# path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/bipartite_subgraph_retrieval/retrieved_subgraph/300_150_llm_70B_64.jsonl"
# data = read_jsonl(path)

for datum in data:
    question = datum['question']
    row_id = datum['positive_ctxs'][0]['rows'].index(datum['positive_ctxs'][0]['answer_node'][0][1][0])
    gold_row = datum['positive_ctxs'][0]['text'].split('\n')[1+row_id]
    gold_passage = datum['positive_ctxs'][0]['target_pasg_titles']
    print()

# new_error_cases_path = "/mnt/sdf/OTT-QAMountSpace/AnalysisResults/Ours/GraphQuerier/error_analysis/300_150_llm_70B_128_150.jsonl"
# new_error_cases = read_jsonl(new_error_cases_path)
# new_error_cases_id = [case['id'] for case in new_error_cases]

# past_error_cases_path = "/mnt/sdf/OTT-QAMountSpace/AnalysisResults/Ours/GraphQuerier/error_analysis/300_150_llm_70B_128_w_llm_150.jsonl"
# past_error_cases = read_jsonl(past_error_cases_path)
# past_error_cases_id = [case['id'] for case in past_error_cases]

# diff = set(new_error_cases_id) - set(past_error_cases_id)

# gold_path = "/mnt/sdf/OTT-QAMountSpace/AnalysisResults/Ours/GraphQuerier/error_analysis/expanded_query_retrieval_5_5_full_150_full.jsonl"
# gold_data = read_jsonl(gold_path)
# gold_data_dict = {case['id']: case for case in gold_data}

# for new_error_case in new_error_cases:
#     if new_error_case['id'] in diff:
#         gold = gold_data_dict[new_error_case['id']]
#         print(new_error_case)
# '2a1065f9912d3e46', 'b39816369f707b40', 'ba3ac95ed8e9d05b', '538f2e1e985199d4', '2231f0cb78f0e912', '1105e7db018cb365', 'b1407faec8e3917c', 'a035145717053d7d', '731adb9db2bd0f1b', '99ce8f40ac13bb76', 'f985766405c50438', '28a009738a90dbc4', 'dcdf89436e31c4fe'
