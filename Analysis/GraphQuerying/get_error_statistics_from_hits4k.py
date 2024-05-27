import json
from tqdm import tqdm
from Algorithms.Ours.dpr.utils.tokenizers import SimpleTokenizer
from Algorithms.Ours.dpr.data.qa_validation import has_answer, _normalize
import unicodedata
import re
def remove_accents_and_non_ascii(text):
    # Normalize the text to decompose combined characters
    normalized_text = unicodedata.normalize('NFD', text)
    # Remove all characters that are not ASCII
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining non-letter characters
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,!.?]', '', ascii_text)
    return cleaned_text
def get_context(retrieved_graph, table_key_to_content, passage_key_to_content, tokenizer, query_topk, augment_topk, filtered_retrieval_type):
    context = ""
    retrieved_table_set = set()
    retrieved_passage_set = set()
    new_sorted_retrieved_graph = []
    sorted_retrieved_graph = sorted(retrieved_graph.items(), key=lambda x: x[1]['score'], reverse=True)
    final_node_rank = 0
    
    for node_rank, (node_id, node_info) in enumerate(sorted_retrieved_graph):
        node_is_added = False
        if node_info['type'] == 'table segment':
            linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation' and (x[3] < query_topk) and (x[4] < augment_topk)) or x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[4] < query_topk) and (x[3] < augment_topk)) or x[2] == 'two_node_graph_retrieval']
            if len(linked_nodes) == 0:
                continue
            else:
                node_info['linked_nodes'] = linked_nodes
            
            table_id = node_id.split('_')[0]
            table = table_key_to_content[table_id]

            if len(tokenizer.tokenize(_normalize(remove_accents_and_non_ascii(context))).words(uncased=True)) > 4096:
                break

            if table_id not in retrieved_table_set:
                if not node_is_added:
                    node_is_added = True
                    new_sorted_retrieved_graph.append((node_id, node_info))
                    final_node_rank += 1
                
                retrieved_table_set.add(table_id)
                context += table['text']

            if len(tokenizer.tokenize(_normalize(remove_accents_and_non_ascii(context))).words(uncased=True)) > 4096:
                break

            max_linked_node_id, max_score, _, _, _ = max(linked_nodes, key=lambda x: x[1], default=(None, 0, 'two_node_graph_retrieval', 0, 0))

            if max_linked_node_id in retrieved_passage_set:
                if not node_is_added:
                    node_is_added = True
                    new_sorted_retrieved_graph.append((node_id, node_info))
                    final_node_rank += 1
                continue

            retrieved_passage_set.add(max_linked_node_id)
            passage_content = passage_key_to_content[max_linked_node_id]
            passage_text = passage_content['title'] + ' ' + passage_content['text']
            
            row_id = int(node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            table_segment_text = column_name + '\n' + row_values
            
            two_node_graph_text = table_segment_text + '\n' + passage_text
            context += two_node_graph_text
            
            if not node_is_added:
                node_is_added = True
                new_sorted_retrieved_graph.append((node_id, node_info))
                final_node_rank += 1

        elif node_info['type'] == 'passage':
            linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < query_topk) and (x[4] < augment_topk)) or x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation' and (x[4] < query_topk) and (x[3] < augment_topk)) or x[2] == 'two_node_graph_retrieval']
            
            if len(linked_nodes) == 0:
                continue
            else:
                node_info['linked_nodes'] = linked_nodes
            
            if node_id in retrieved_passage_set:
                if not node_is_added:
                    node_is_added = True
                    new_sorted_retrieved_graph.append((node_id, node_info))
                    final_node_rank += 1
                continue

            if len(tokenizer.tokenize(_normalize(remove_accents_and_non_ascii(context))).words(uncased=True)) > 4096:
                break

            max_linked_node_id, max_score, _, _, _ = max(linked_nodes, key=lambda x: x[1], default=(None, 0, 'two_node_graph_retrieval', 0, 0))
            table_id = max_linked_node_id.split('_')[0]
            table = table_key_to_content[table_id]
            
            if table_id not in retrieved_table_set:
                retrieved_table_set.add(table_id)
                context += table['text']

            if len(tokenizer.tokenize(_normalize(remove_accents_and_non_ascii(context))).words(uncased=True)) > 4096:
                break

            row_id = int(max_linked_node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            table_segment_text = column_name + '\n' + row_values

            retrieved_passage_set.add(node_id)
            passage_content = passage_key_to_content[node_id]
            passage_text = passage_content['title'] + ' ' + passage_content['text']

            two_node_graph_text = table_segment_text + '\n' + passage_text
            context += two_node_graph_text
            if not node_is_added:
                node_is_added = True
                new_sorted_retrieved_graph.append((node_id, node_info))
                final_node_rank += 1

    return remove_accents_and_non_ascii(context), new_sorted_retrieved_graph, final_node_rank

if __name__ == '__main__':
    
    error_cases_path = "/mnt/sdd/shpark/error_case_analysis_results/error_cases_passage_4_1.json"#"/home/shpark/OTT_QA_Workspace/Analysis/GraphQueryResults/error_cases.json"
    error_cases_path_1 = "/mnt/sdd/shpark/error_case_analysis_results/error_cases_passage_4_1.json"#"/home/shpark/OTT_QA_Workspace/Analysis/GraphQueryResults/error_cases.json"
    error_cases_path_2 = "/mnt/sdd/shpark/error_case_analysis_results/error_cases_both_4_1.json"#"/home/shpark/OTT_QA_Workspace/Analysis/GraphQueryResults/error_cases.json"
    query_topk = 4
    augment_topk = 1
    filtered_retrieval_type = ['two_node_graph_retrieval', 'passage_node_augmentation']#, 'passage_node_augmentation']#, 'passage_node_augmentation', 'table_segment_node_augmentation']#, 'table_segment_node_augmentation']#, 'table_segment_node_augmentation']#['two_node_graph_retrieval', 'passage_node_augmentation']
    data_graph_error_cases_path = "/home/shpark/OTT_QA_Workspace/Analysis/GraphQueryResults/data_graph_error_cases.json"
    table_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    passage_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json"
    error_case_path = "/mnt/sdd/shpark/error_case_analysis_results/error_cases_passage_4_1.json"
    
    # retrieval_error_cases = json.load(open(error_cases_path))
    # retrieval_error_cases_1 = json.load(open(error_cases_path_1))
    #a = ['9c9398e85be320e4', 'cc9be19675c95343', 'd18e42fc2b58845d'] # 3개 다 both none일듯
    #b = 4e896f5593b5f163
    #de6c9822e5d9503c
    a = ['037c2856d7ddc3fa', '4f8e7ade5ff9a1e2', 'de54043e5ebeb262', 'b4a3a106c99e6142'] #1. 정답은 있지만, 제대로된 검색이 아님.
    retrieval_error_cases = json.load(open(error_cases_path))
    data_graph_error_cases = json.load(open(data_graph_error_cases_path))
    
    table_key_to_content = {}
    table_contents = json.load(open(table_data_path))
    for table_key, table_content in enumerate(table_contents):
        table_key_to_content[str(table_key)] = table_content
    
    passage_key_to_content = {}
    passage_contents = json.load(open(passage_data_path))
    for passage_content in passage_contents:
        passage_key_to_content[passage_content['title']] = passage_content
    
    tokenizer = SimpleTokenizer()
    
    error_case = {'table_none':0, 'passage_none':0, 'both_none':0}
    pasage_none_list = []
    for qid, retrieval_error_case in tqdm(retrieval_error_cases.items()):
        answers = retrieval_error_case['answers']
        question = retrieval_error_case['question']
        positive_ctxs = retrieval_error_case['positive_ctxs']
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
                    passage_id = answer_node[2].replace('/wiki/','').replace('_', ' ')
                    positive_passages.add(passage_id)
        
        
        retrieved_graph = retrieval_error_case['retrieved_graph']
        normalized_context, sorted_retrieved_graph, final_node_rank = get_context(retrieved_graph, table_key_to_content, passage_key_to_content, tokenizer, query_topk, augment_topk, filtered_retrieval_type)
        normalized_answers = [remove_accents_and_non_ascii(answer) for answer in answers]
        is_has_answer = has_answer(normalized_answers, normalized_context, tokenizer, 'string', max_length=4096)
        
        if is_has_answer:
            continue
        
        table_exist = False
        passage_exist = False
        for node_id, retrieved_node_info in sorted_retrieved_graph[:final_node_rank]:
            if len(retrieved_node_info['linked_nodes']) == 0:
                continue
            
            if retrieved_node_info['type'] == 'table segment' and not table_exist:
                row_id = node_id.split('_')[1]
                chunk_id = retrieved_node_info['chunk_id']
                retrieved_table_segment_id = f"{chunk_id}_{row_id}"
                if retrieved_table_segment_id in positive_table_segments:
                    table_exist = True
            
            if retrieved_node_info['type'] == 'passage' and not passage_exist:
                if node_id in positive_passages or len(positive_passages) == 0:
                    passage_exist = True
            
        if table_exist and not passage_exist:
            error_case['passage_none'] += 1
        elif passage_exist and not table_exist:
            error_case['table_none'] += 1
        elif not table_exist and not passage_exist:
            error_case['both_none'] += 1
        else:
            print('error')

    print(error_case)
    # 예외: '7babe6e8962eb0dc', table =1, passage , table 1
    # Table segment가 없는 경우에 대한 조사가 필요함.
    
    # print("Total Number of Error Cases: ", len(retrieval_error_cases))
    # data_graph_error_id_list = list(set(retrieval_error_cases.keys()).intersection(set([data_graph_error_case['id'] for data_graph_error_case in data_graph_error_cases])))
    # print("Total Number of Data Graph Error Cases: ", len(data_graph_error_id_list))
    # non_data_graph_error_id_list = list(set(retrieval_error_cases.keys()).union(set([data_graph_error_case['id'] for data_graph_error_case in data_graph_error_cases])).difference(set([data_graph_error_case['id'] for data_graph_error_case in data_graph_error_cases])))
    # print("Total Number of Non-Data Graph Error Cases: ", len(non_data_graph_error_id_list))
    
    # tokenizer = SimpleTokenizer()
    
    # for non_data_graph_error_id in non_data_graph_error_id_list:
    #     answers = retrieval_error_cases[non_data_graph_error_id]['answers']
    #     question = retrieval_error_cases[non_data_graph_error_id]['question']
    #     positive_ctxs = retrieval_error_cases[non_data_graph_error_id]['positive_ctxs']
    #     retrieved_graph = retrieval_error_cases[non_data_graph_error_id]['retrieved_graph']
    #     context, sorted_retrieved_graph, final_node_rank = get_context(retrieved_graph, table_key_to_content, passage_key_to_content, tokenizer)
    #     print('final_node_rank: ', final_node_rank)
#'6ad2c846a3dbab5c'