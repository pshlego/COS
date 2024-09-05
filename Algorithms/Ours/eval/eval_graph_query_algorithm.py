import re
import json
import copy
import unicodedata
from tqdm import tqdm
from Ours.dpr.data.qa_validation import has_answer
from Ours.dpr.utils.tokenizers import SimpleTokenizer
FINAL_MAX_EDGE_COUNT = 50

def dump_jsonl(data, path):
    """
    Dumps a list of dictionaries to a JSON Lines file.

    :param data: List of dictionaries to be dumped into JSONL.
    :param path: Path where the JSONL file will be saved.
    """
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')
        print(f"Data successfully written to {path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

time_list = read_jsonl("/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/expanded_query_retrieval_5_5_full_time.jsonl")

# def evaluate(retrieved_graph_list, qa_dataset, graph_query_engine):
def evaluate(retrieved_graph, qa_data, table_key_to_content, passage_key_to_content):
    filtered_retrieval_type = ['edge_reranking', "node_augmentation", 'llm_selected']
    filtered_retrieval_type_1 = ['edge_reranking', 'llm_selected']
    filtered_retrieval_type_2 = ["node_augmentation"]
    # filtered_retrieval_type = ['llm_selected']
    # filtered_retrieval_type_1 = ['llm_selected']
    # filtered_retrieval_type_2 = []
    # 1. Revise retrieved graph
    revised_retrieved_graph = {}
    for node_id, node_info in retrieved_graph.items():                  

        if node_info['type'] == 'table segment':
            linked_nodes = [x for x in node_info['linked_nodes'] 
                                if x[2] in filtered_retrieval_type and (x[2] in filtered_retrieval_type_2 and (x[3] < 10) and (x[4] < 10000)) 
                                    or x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[4] < 0) and (x[3] < 0)) 
                                    or x[2] in filtered_retrieval_type_1
                            ]
        elif node_info['type'] == 'passage':
            linked_nodes = [x for x in node_info['linked_nodes'] 
                                if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < 0) and (x[4] < 0)) 
                                or x[2] in filtered_retrieval_type and (x[2] in filtered_retrieval_type_2 and (x[4] < 10) and (x[3] < 10000)) 
                                or x[2] in filtered_retrieval_type_1
                            ]
        else:
            continue
        
        if len(linked_nodes) == 0: continue
        
        revised_retrieved_graph[node_id] = copy.deepcopy(node_info)
        revised_retrieved_graph[node_id]['linked_nodes'] = linked_nodes
        
        linked_scores = [linked_node[1] for linked_node in linked_nodes]

        node_score = max(linked_scores)
        
        if node_score == 1000000:
            additional_score_list = [linked_node[1] for linked_node in linked_nodes if linked_node[2] != 'llm_selected']
            if len(additional_score_list) > 0:
                node_score += max(additional_score_list)

        revised_retrieved_graph[node_id]['score'] = node_score


    # 2. Evaluate with revised retrieved graph
    node_count = 0
    edge_count = 0
    answers = qa_data['answers']
    context = ""
    retrieved_table_set = set()
    retrieved_passage_set = set()
    sorted_retrieved_graph = sorted(revised_retrieved_graph.items(), key = lambda x: x[1]['score'], reverse = True)
    
    for node_id, node_info in sorted_retrieved_graph:
        if edge_count < FINAL_MAX_EDGE_COUNT:
            node_count += 1
        if node_info['type'] == 'table segment':
            
            table_id = node_id.split('_')[0]
            table = table_key_to_content[table_id]
            chunk_id = table['chunk_id']
            node_info['chunk_id'] = chunk_id
            
            if table_id not in retrieved_table_set:
                retrieved_table_set.add(table_id)
                
                if edge_count == FINAL_MAX_EDGE_COUNT:
                    continue
                
                context += table['text']
                edge_count += 1
                
            max_linked_node_id, max_score, _, _, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default=(None, 0, 'edge_reranking', 0, 0))
            
            if max_linked_node_id in retrieved_passage_set:
                continue
            
            retrieved_passage_set.add(max_linked_node_id)
            passage_content = passage_key_to_content[max_linked_node_id]
            passage_text = passage_content['title'] + ' ' + passage_content['text']
            
            row_id = int(node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            table_segment_text = column_name + '\n' + row_values
            
            edge_text = table_segment_text + '\n' + passage_text
            
            if edge_count == FINAL_MAX_EDGE_COUNT:
                continue
            
            edge_count += 1
            context += edge_text
        
        elif node_info['type'] == 'passage':

            if node_id in retrieved_passage_set:
                continue

            max_linked_node_id, _, _, _, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default = (None, 0, 'edge_reranking', 0, 0))
            table_id = max_linked_node_id.split('_')[0]
            table = table_key_to_content[table_id]
            
            if table_id not in retrieved_table_set:
                retrieved_table_set.add(table_id)
                if edge_count == FINAL_MAX_EDGE_COUNT:
                    continue
                context += table['text']
                edge_count += 1

            row_id = int(max_linked_node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            table_segment_text = column_name + '\n' + row_values
            
            if edge_count == FINAL_MAX_EDGE_COUNT:
                continue

            retrieved_passage_set.add(node_id)
            passage_content = passage_key_to_content[node_id]
            passage_text = passage_content['title'] + ' ' + passage_content['text']
            
            edge_text = table_segment_text + '\n' + passage_text
            context += edge_text
            edge_count += 1

    normalized_context = remove_accents_and_non_ascii(context)
    normalized_answers = [remove_accents_and_non_ascii(answer) for answer in answers]
    is_has_answer = has_answer(normalized_answers, normalized_context, SimpleTokenizer(), 'string')
    
    if is_has_answer:
        recall = 1
        error_analysis = copy.deepcopy(qa_data)
        error_analysis['retrieved_graph'] = retrieved_graph
        error_analysis['sorted_retrieved_graph'] = sorted_retrieved_graph[:node_count]
        if  "hard_negative_ctxs" in error_analysis:
            del error_analysis["hard_negative_ctxs"]
    else:
        recall = 0
        error_analysis = copy.deepcopy(qa_data)
        error_analysis['retrieved_graph'] = retrieved_graph
        error_analysis['sorted_retrieved_graph'] = sorted_retrieved_graph[:node_count]
        if  "hard_negative_ctxs" in error_analysis:
            del error_analysis["hard_negative_ctxs"]

    return recall, error_analysis

def remove_accents_and_non_ascii(text):
    # Normalize the text to decompose combined characters
    normalized_text = unicodedata.normalize('NFD', text)
    # Remove all characters that are not ASCII
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining non-letter characters
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,!.?\-]', '', ascii_text)
    return cleaned_text

if __name__ == '__main__':
    results_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/bipartite_subgraph_retrieval/retrieved_subgraph/beam_search_300_150_w_llm_10.jsonl"#"/mnt/sdf/OTT-QAMountSpace/ExperimentResults/bipartite_subgraph_retrieval/retrieved_subgraph/300_150_llm_8B.jsonl"#"/mnt/sdf/OTT-QAMountSpace/ExperimentResults/bipartite_subgraph_retrieval/retrieved_subgraph/beam_search_40.jsonl"#/mnt/sdf/OTT-QAMountSpace/ExperimentResults/bipartite_subgraph_retrieval/retrieved_subgraph/200_100_original_llm_v2.jsonl"
    #results_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/expanded_query_retrieval_5_5_full.jsonl"
    table_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    passage_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json"
    #"/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/expanded_query_retrieval_5_5_full.jsonl"#
    # 3. Load tables
    print("3. Loading tables...")
    table_key_to_content = {}
    table_contents = json.load(open(table_data_path))
    print("3. Loaded " + str(len(table_contents)) + " tables!")
    print("3. Processing tables...")
    for table_key, table_content in tqdm(enumerate(table_contents)):
        table_key_to_content[str(table_key)] = table_content
    print("3. Processing tables complete!", end = "\n\n")
    
    # 4. Load passages
    print("4. Loading passages...")
    passage_key_to_content = {}
    passage_contents = json.load(open(passage_data_path))
    print("4. Loaded " + str(len(passage_contents)) + " passages!")
    print("4. Processing passages...")
    for passage_content in tqdm(passage_contents):
        passage_key_to_content[passage_content['title']] = passage_content
    print("4. Processing passages complete!", end = "\n\n")
    data_graph_error_cases = json.load(open("/home/shpark/OTT_QA_Workspace/data_graph_error_case.json"))
    error_cases_id_list = ['2a1065f9912d3e46', 'b39816369f707b40', 'ba3ac95ed8e9d05b', '538f2e1e985199d4', '2231f0cb78f0e912', '1105e7db018cb365', 'b1407faec8e3917c', 'a035145717053d7d', '731adb9db2bd0f1b', '99ce8f40ac13bb76', 'f985766405c50438', '28a009738a90dbc4', 'dcdf89436e31c4fe'] #[data_graph_error_case['id'] for data_graph_error_case in data_graph_error_cases]
    retrieved_results = read_jsonl(results_path)
    recall_list = []
    error_case_list = [] 

    for retrieved_result in tqdm(retrieved_results):#zip(min_data,max_data)):#[:209]
        qa_data = retrieved_result["qa data"]
        if qa_data['id'] not in error_cases_id_list:
            continue
        row_id = qa_data['positive_ctxs'][0]['rows'].index(qa_data['positive_ctxs'][0]['answer_node'][0][1][0])
        gold_row = qa_data['positive_ctxs'][0]['text'].split('\n')[1+row_id]
        gold_passage = qa_data['positive_ctxs'][0]['target_pasg_titles']
        retrieved_graph = retrieved_result["retrieved graph"]
        recall, error_case  = evaluate(retrieved_graph, qa_data, table_key_to_content, passage_key_to_content)
        recall_list.append(recall)

        # if recall == 0:
        error_case_list.append(error_case)
            
    print("Min: ", sum(recall_list)/len(recall_list))
    print('len: ', len(recall_list))
    print()
    
    # dump_jsonl(error_case_list, "/mnt/sdf/OTT-QAMountSpace/AnalysisResults/Ours/GraphQuerier/error_analysis/300_150_llm_70B_128_w_llm_150_full.jsonl")