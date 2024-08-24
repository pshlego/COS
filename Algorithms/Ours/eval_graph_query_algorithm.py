import re
import json
import copy
import unicodedata
from tqdm import tqdm
from Ours.dpr.data.qa_validation import has_answer
from Ours.dpr.utils.tokenizers import SimpleTokenizer
FINAL_MAX_EDGE_COUNT = 5

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
def evaluate(retrieved_graph, qa_data, table_key_to_content, passage_key_to_content, score_function = 'max'):
    
    # table_key_to_content = graph_query_engine.table_key_to_content
    # passage_key_to_content = graph_query_engine.passage_key_to_content
    # filtered_retrieval_type = ['edge_reranking', "passage_node_augmentation_1"]
    # filtered_retrieval_type_1 = ['edge_reranking']
    filtered_retrieval_type = ['edge_reranking', "node_augmentation_1", "passage_node_augmentation_1", "entity_linking_1", 'llm_selected']
    filtered_retrieval_type_1 = ['edge_reranking', 'llm_selected']#, 'llm_selected'
    filtered_retrieval_type_2 = ["node_augmentation_1", "passage_node_augmentation_1", "entity_linking_1"]
    # filtered_retrieval_type = ['edge_retrieval', "passage_node_augmentation_0", "llm_selected"]
    # filtered_retrieval_type = ['edge_retrieval', "passage_node_augmentation_1"]
    # filtered_retrieval_type_1 = ['edge_retrieval', "llm_selected"]
    # 1. Revise retrieved graph
    revised_retrieved_graph = {}
    for node_id, node_info in retrieved_graph.items():                  

        if node_info['type'] == 'table segment':
            linked_nodes = [x for x in node_info['linked_nodes'] 
                                if x[2] in filtered_retrieval_type and (x[2] in filtered_retrieval_type_2 and (x[3] < 10) and (x[4] < 10000)) 
                                    or x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[4] < 1) and (x[3] < 1)) 
                                    or x[2] in filtered_retrieval_type_1#['edge_reranking', "llm_selected"]
                            ]
        elif node_info['type'] == 'passage':
            linked_nodes = [x for x in node_info['linked_nodes'] 
                                if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < 1) and (x[4] < 1)) 
                                or x[2] in filtered_retrieval_type and (x[2] in filtered_retrieval_type_2 and (x[4] < 10) and (x[3] < 10000)) 
                                or x[2] in filtered_retrieval_type_1#['edge_reranking', "llm_selected"]
                            ]
        else:
            continue
        
        if len(linked_nodes) == 0: continue
        
        revised_retrieved_graph[node_id] = copy.deepcopy(node_info)
        revised_retrieved_graph[node_id]['linked_nodes'] = linked_nodes
        
        linked_scores = [linked_node[1] for linked_node in linked_nodes]

        if score_function == 'max':
            node_score = max(linked_scores)
        elif score_function == 'avg':
            node_score = sum(linked_scores) / len(linked_scores)
        else:
            node_score = min(linked_scores)

        revised_retrieved_graph[node_id]['score'] = node_score
    # retrieved_graph = revised_retrieved_graph


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
                # node_count += 1
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

        # node_count += 1

    normalized_context = remove_accents_and_non_ascii(context)
    normalized_answers = [remove_accents_and_non_ascii(answer) for answer in answers]
    is_has_answer = has_answer(normalized_answers, normalized_context, SimpleTokenizer(), 'string')
    
    if is_has_answer:
        recall = 1
        error_analysis = copy.deepcopy(qa_data)
        # sorted_retrieved_graph = sorted(retrieved_graph.items(), key = lambda x: x[1]['score'], reverse = True)
        # for node_id, node_info in sorted_retrieved_graph:
        #     if node_info['type'] == 'table segment':
        #         table_id = node_id.split('_')[0]
        #         table = table_key_to_content[table_id]
        #         chunk_id = table['chunk_id']
        #         node_info['chunk_id'] = chunk_id

        #     elif node_info['type'] == 'table':
        #         table_id = node_id.split('_')[0]
        #         table = table_key_to_content[table_id]
        #         chunk_id = table['chunk_id']
        #         node_count += 1
                
        error_analysis['retrieved_graph'] = retrieved_graph
        error_analysis['sorted_retrieved_graph'] = sorted_retrieved_graph[:node_count]
        if  "hard_negative_ctxs" in error_analysis:
            del error_analysis["hard_negative_ctxs"]
    else:
        recall = 0
        error_analysis = copy.deepcopy(qa_data)
        # sorted_retrieved_graph = sorted(retrieved_graph.items(), key = lambda x: x[1]['score'], reverse = True)
        # for node_id, node_info in sorted_retrieved_graph:
        #     if node_info['type'] == 'table segment':
        #         table_id = node_id.split('_')[0]
        #         table = table_key_to_content[table_id]
        #         chunk_id = table['chunk_id']
        #         node_info['chunk_id'] = chunk_id

        #     elif node_info['type'] == 'table':
        #         table_id = node_id.split('_')[0]
        #         table = table_key_to_content[table_id]
        #         chunk_id = table['chunk_id']
        #         node_count += 1

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
    # Test
    #query_results_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/avg.jsonl"#"/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/final_results_150_10_0_0_2_3_150_28_256.jsonl"
    max_results_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/missing_link_4_3.jsonl"#"/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/original_table_new_2.jsonl"#"/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/final_results_150_10_0_0_2_3_150_28_256.jsonl"
    min_results_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/expanded_query_retrieval_5_5_full.jsonl" #heuristic_10_10.jsonl"
    #data_graph_error_cases_path = "/home/shpark/OTT_QA_Workspace/data_graph_error_case.json"
    #max_results_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/expanded_query_retrieval_10_query_2_target_diff_passage_text.jsonl"#"/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/table_embedding/original_table_new_2.jsonl"
    # avg_results_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/min_max_avg/avg.jsonl"
    #query_results_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/final_results_150_10_0_0_2_3_150_28_256 copy.jsonl"
    table_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    passage_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json"
    #"/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/expanded_query_retrieval_3.jsonl"#
    #data_graph_error_cases = json.load(open(data_graph_error_cases_path))
    #data_graph_error_case_id_list = [data_graph_error_case['id'] for data_graph_error_case in data_graph_error_cases]
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
    
    # data = read_jsonl(query_results_path)
    min_data = read_jsonl(min_results_path)
    max_data = read_jsonl(max_results_path)
    # avg_data = read_jsonl(avg_results_path)
    #negative_id_list = data_graph_error_case_id_list#json.load(open("/home/shpark/OTT_QA_Workspace/positive_id_list.json"))
    # negative_id_list = []#json.load(open("/home/shpark/OTT_QA_Workspace/positive_id_list.json"))
    min_recall_list = []
    max_recall_list = []
    avg_recall_list = []
    # max_data = [max_datum for max_datum in max_data if max_datum['qa data']['id'] in negative_id_list]
    
    # # #filter out min_datum when the min_datum is float type
    # # min_data = [min_datum for min_datum in min_data if type(min_datum) == dict]
    #min_data = [min_datum for min_datum in min_data if min_datum['qa data']['id'] in negative_id_list]
    # positive_id_list = []
    # negative_id_list = []
    error_case_list = [] 
    for min_datum in tqdm(min_data):#zip(min_data,max_data)):#[:209]
        qa_data = min_datum["qa data"]
        
        # if qa_data['id'] not in ['1611198ca9c7727a', '4bd8cba7b5c795e5', 'f7f4e62a36c9da1b', 'f77c5527ab108782', 'e815a79f1c6e19d9', 'ac3e93545c40c7ea', '643b3bd6d5367890', '7a158221e7c9b6eb', '1f19b4a4533d8e07']:
        #     continue
        # if qa_data['id'] not in negative_id_list:
        #     continue
        
        #max_retrieved_graph = max_datum["retrieved graph"]
        min_retrieved_graph = min_datum["retrieved graph"]
        
        #max_recall, max_error_case  = evaluate(max_retrieved_graph, qa_data, table_key_to_content, passage_key_to_content, score_function = 'max')
        min_recall, min_error_case  = evaluate(min_retrieved_graph, qa_data, table_key_to_content, passage_key_to_content, score_function = 'max')
        # print("Min: ", min_recall)
        # print("Max: ", max_recall)
        # print("Avg: ", avg_recall)
        
        # max_recall_list.append(max_recall)
        min_recall_list.append(min_recall)
        # if max_recall == 0 and min_recall == 1:
        #     positive_id_list.append(qa_data['id'])
        # elif max_recall == 1 and min_recall == 0:
        #     negative_id_list.append(qa_data['id'])
        #if min_recall == 0:
        error_case_list.append(min_error_case)

    # print("Max: ", sum(max_recall_list)/len(max_recall_list))
    print("Min: ", sum(min_recall_list)/len(min_recall_list))
    print('len: ', len(max_recall_list))
    print()
    # print(len(recall_list))
    # with open("/mnt/sdf/OTT-QAMountSpace/AnalysisResults/Ours/GraphQuerier/error_cases/expanded_query_retrieval_5_5_full_data_graph_error_total.json", "w") as file:
    #     json.dump(error_case_list, file, indent = 4)
    # with open("/home/shpark/OTT_QA_Workspace/positive_id_list.json", "w") as file:
    #     json.dump(positive_id_list, file, indent = 4)
    # with open("/home/shpark/OTT_QA_Workspace/negative_id_list.json", "w") as file:
    #     json.dump(negative_id_list, file, indent = 4)
    
# len(error_case_list)
# 145
# sum(recall_list)/len(recall_list)
# 0.9345076784101174
# sum(recall_list)/len(recall_list)
# 0.9327009936766034
# 0.9345076784101174-0.9327009936766034
# 0.0018066847335139746
# 0.936
# 0.9359190556492412
# top-10 0.821247892074199
# top-10 0.8482293423271501
# 76434

# 0.9426377597109304
# 0.9202898550724637
# 0.9394760614272809
#max_data.index(max_datum) == 98

#0.935064935064935
#'1611198ca9c7727a', '4bd8cba7b5c795e5', 'f7f4e62a36c9da1b', 'f77c5527ab108782', 'e815a79f1c6e19d9'
# (, 1), (2,2), (0,0), (0,0), (0,0), (4,0), (0,0), (,3)
#'85b79397b5d60f47'
# error case: 'f77c5527ab108782', '960022483a9d97c4', 'b471d2d8e30cdc25'
# 무조건 확인 잘못된 entity linking: '643b3bd6d5367890' (순위 이슈), '8c10f7945508d534 (column select)'




#8/20
#확인해야할 qid
#'f7f4e62a36c9da1b' (llm 선택 과정에서 Italy 324305_1 유실 = LLM이 Passage 선택 과정에서 자체적으로 생성함), 'f77c5527ab108782'  (llm 선택 과정에서 'Huddersfield Giants' '271987_2' 유실 == George Gatis라는 선수를 선택함, team shirt에 대한 정보는 passage의 하단에 존재함), 'b1ab98b946bd09fb' (llm 선택 과정에서 'New Jersey' '670733_3' 유실 )
#'a66efd6e908d7190' (279084가 table segment selection에서 잘못 수행됐음)
# '1611198ca9c7727a', '4bd8cba7b5c795e5', '969c47956240ac9b', 'c3bee8dd58c3f5b3' 검색 결과 하위에 있어 밀려남
# 미상 '960022483a9d97c4', 'b471d2d8e30cdc25'

#643b3bd6d5367890 (두 번째 node augmentation할 때 순위 확인 필요)

# 'f7f4e62a36c9da1b', 'a66efd6e908d7190' 해결함

# Entity linking으로 더 맞출 수 있는게 있는지 확인 필요