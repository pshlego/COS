import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/bipartite_subgraph_retrieval/retrieved_subgraph/300_150_llm_70B_64.jsonl"
data = read_jsonl(path)

for datum in data:
    qa_datum = datum['qa_datum']

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

