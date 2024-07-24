import re
import json
import copy
import unicodedata
from tqdm import tqdm
from Ours.dpr.data.qa_validation import has_answer
from Ours.dpr.utils.tokenizers import SimpleTokenizer
FINAL_MAX_EDGE_COUNT = 50

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# def evaluate(retrieved_graph_list, qa_dataset, graph_query_engine):
def evaluate(retrieved_graph, qa_data, table_key_to_content, passage_key_to_content):
    
    # table_key_to_content = graph_query_engine.table_key_to_content
    # passage_key_to_content = graph_query_engine.passage_key_to_content
    filtered_retrieval_type = ['edge_reranking', "passage_node_augmentation_1"]
    filtered_retrieval_type_1 = ['edge_reranking']
    # filtered_retrieval_type = ['edge_reranking', "passage_node_augmentation_1", "llm_selected"]
    # filtered_retrieval_type_1 = ['edge_reranking', "llm_selected"]
    # 1. Revise retrieved graph
    revised_retrieved_graph = {}
    for node_id, node_info in retrieved_graph.items():                  

        if node_info['type'] == 'table segment':
            linked_nodes = [x for x in node_info['linked_nodes'] 
                                if x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation_1' and (x[3] < 10) and (x[4] < 2)) 
                                    or x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[4] < 1) and (x[3] < 1)) 
                                    or x[2] in filtered_retrieval_type_1#['edge_reranking', "llm_selected"]
                            ]
        elif node_info['type'] == 'passage':
            linked_nodes = [x for x in node_info['linked_nodes'] 
                                if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < 1) and (x[4] < 1)) 
                                or x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation_1' and (x[4] < 10) and (x[3] < 2)) 
                                or x[2] == filtered_retrieval_type_1#['edge_reranking', "llm_selected"]
                            ]
        if len(linked_nodes) == 0: continue
        
        revised_retrieved_graph[node_id] = copy.deepcopy(node_info)
        revised_retrieved_graph[node_id]['linked_nodes'] = linked_nodes
        
        linked_scores = [linked_node[1] for linked_node in linked_nodes]
        node_score = max(linked_scores)
        revised_retrieved_graph[node_id]['score'] = node_score
    retrieved_graph = revised_retrieved_graph


    # 2. Evaluate with revised retrieved graph
    node_count = 0
    edge_count = 0
    answers = qa_data['answers']
    context = ""
    retrieved_table_set = set()
    retrieved_passage_set = set()
    sorted_retrieved_graph = sorted(retrieved_graph.items(), key = lambda x: x[1]['score'], reverse = True)
    
    for node_id, node_info in sorted_retrieved_graph:
        
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

        node_count += 1

    normalized_context = remove_accents_and_non_ascii(context)
    normalized_answers = [remove_accents_and_non_ascii(answer) for answer in answers]
    is_has_answer = has_answer(normalized_answers, normalized_context, SimpleTokenizer(), 'string')
    
    if is_has_answer:
        recall = 1
        error_analysis = {}
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
    # Test
    #query_results_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/final_results_300_5_0_0_2_1_150_28_256.jsonl"
    query_results_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/final_results_150_10_0_0_2_3_150_28_256 copy.jsonl"
    table_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    passage_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json"
    
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
    table_key_to_content
    data = read_jsonl(query_results_path)
    
    recall_list = []
    error_case_list = [] 
    for datum in data:
        qa_data = datum["qa data"]
        retrieved_graph = datum["retrieved graph"]
        recall, error_case  = evaluate(retrieved_graph, qa_data, table_key_to_content, passage_key_to_content)
        recall_list.append(recall)
        
        if recall == 0:
            error_case_list.append(error_case)
    print("Recall: ", sum(recall_list)/len(recall_list))
    with open("error_cases_0_2.json", "w") as file:
        json.dump(error_case_list, file, indent = 4)
        
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