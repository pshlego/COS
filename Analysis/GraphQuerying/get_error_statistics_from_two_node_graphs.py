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
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,!.?\-]', '', ascii_text)
    return cleaned_text

def get_context(retrieved_graph, table_key_to_content, passage_key_to_content, tokenizer, query_topk, augment_topk, filtered_retrieval_type):
    two_node_graph_count = 0
    context = ""
    retrieved_table_set = set()
    retrieved_passage_set = set()
    new_sorted_retrieved_graph = []
    sorted_retrieved_graph = sorted(retrieved_graph.items(), key=lambda x: x[1]['score'], reverse=True)
    final_node_rank = 0
    
    for node_rank, (node_id, node_info) in enumerate(sorted_retrieved_graph):
        node_is_added = False
        if node_info['type'] == 'table segment':
            linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation_1' and (x[3] < query_topk) and (x[4] < augment_topk)) or x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[4] < 2) and (x[3] < 2)) or x[2] == 'two_node_graph_reranking']
            if len(linked_nodes) == 0:
                continue
            else:
                node_info['linked_nodes'] = linked_nodes
            
            table_id = node_id.split('_')[0]
            table = table_key_to_content[table_id]
            
            if table_id not in retrieved_table_set:
                if two_node_graph_count == 50:
                    break
                
                if not node_is_added:
                    node_is_added = True
                    new_sorted_retrieved_graph.append((node_id, node_info))
                    final_node_rank += 1
                
                retrieved_table_set.add(table_id)
                context += table['text']
                two_node_graph_count += 1
            
            if two_node_graph_count == 50:
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
            two_node_graph_count += 1

            if not node_is_added:
                node_is_added = True
                new_sorted_retrieved_graph.append((node_id, node_info))
                final_node_rank += 1

        elif node_info['type'] == 'passage':
            linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < 2) and (x[4] < 2)) or x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation_1' and (x[4] < query_topk) and (x[3] < augment_topk)) or x[2] == 'two_node_graph_reranking']

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

            if two_node_graph_count == 50:
                break            

            max_linked_node_id, max_score, _, _, _ = max(linked_nodes, key=lambda x: x[1], default=(None, 0, 'two_node_graph_retrieval', 0, 0))
            table_id = max_linked_node_id.split('_')[0]
            table = table_key_to_content[table_id]
            
            if table_id not in retrieved_table_set:
                retrieved_table_set.add(table_id)
                context += table['text']
                two_node_graph_count += 1

            if two_node_graph_count == 50:
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
            two_node_graph_count += 1
            
            if not node_is_added:
                node_is_added = True
                new_sorted_retrieved_graph.append((node_id, node_info))
                final_node_rank += 1

    return remove_accents_and_non_ascii(context), new_sorted_retrieved_graph, final_node_rank

if __name__ == '__main__':
    query_topk = 10
    augment_topk = 2
    filter_type = 'rerank' # table, passage, both
    if filter_type == 'none':
        filtered_retrieval_type = ['two_node_graph_retrieval']
    elif filter_type == 'table':
        filtered_retrieval_type = ['two_node_graph_retrieval', 'table_segment_node_augmentation']
    elif filter_type == 'passage':
        filtered_retrieval_type = ['two_node_graph_retrieval', 'passage_node_augmentation', 'two_node_graph_reranking']
    elif filter_type == 'rerank':
        filtered_retrieval_type = ['two_node_graph_reranking', 'passage_node_augmentation_1']
    else:
        filtered_retrieval_type = ['two_node_graph_retrieval', 'passage_node_augmentation', 'table_segment_node_augmentation']
        
    error_cases_path = f"/mnt/sdd/shpark/experimental_results/error_cases/150_10_2_w_reranking.json"
    table_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    passage_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json"
    table_error_case_result_path = "/home/shpark/OTT_QA_Workspace/error_case/table_error_cases_reranking_200_10_2_trained.json"
    passage_error_case_result_path = "/home/shpark/OTT_QA_Workspace/error_case/passage_error_cases_reranking_200_10_2_trained.json"
    both_error_case_result_path = "/home/shpark/OTT_QA_Workspace/error_case/both_error_cases_reranking_200_10_2_trained.json"
    gold_graph_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/GroundTruth/wiki_hyperlink.json"
    error_qid_list = []
    
    table_segment_id_to_linked_passages = json.load(open(gold_graph_path))
    retrieval_error_cases = json.load(open(error_cases_path))
    
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
    passage_error_cases = {}
    table_error_cases = {}
    both_error_cases = {}
    for qid, retrieval_error_case in tqdm(retrieval_error_cases.items()):
        answers = retrieval_error_case['answers']
        question = retrieval_error_case['question']
        positive_ctxs = retrieval_error_case['positive_ctxs']
        positive_table_segments = set()
        raw_positive_table_segments = set()
        positive_passages = set()
        for positive_ctx in positive_ctxs:
            chunk_id = positive_ctx['chunk_id']
            chunk_rows = positive_ctx['rows']
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

        if len(positive_passages) == 0:
            for raw_table_segment_id in raw_positive_table_segments:
                table_id = '_'.join(raw_table_segment_id.split('_')[:-2])
                row_id = int(raw_table_segment_id.split('_')[-1])
                linked_passages = table_segment_id_to_linked_passages[table_id][row_id]
                for linked_passage in linked_passages:
                    positive_passages.add(linked_passage[1])

        retrieved_graph = retrieval_error_case['retrieved_graph']
        normalized_context, sorted_retrieved_graph, final_node_rank = get_context(retrieved_graph, table_key_to_content, passage_key_to_content, tokenizer, query_topk, augment_topk, filtered_retrieval_type)
        normalized_answers = [remove_accents_and_non_ascii(answer) for answer in answers]
        is_has_answer = has_answer(normalized_answers, normalized_context, tokenizer, 'string')
        
        if is_has_answer:
            continue
        
        error_qid_list.append(qid)
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
            passage_error_cases[qid] = {'answer': answers, 'question': question, 'positive_ctxs': positive_ctxs, 'positive_table_segments': list(positive_table_segments), 'positive_passages':list(positive_passages),'retrieved_graph': retrieval_error_case['retrieved_graph'], 'sorted_retrieved_graph':sorted_retrieved_graph, 'final_node_rank': final_node_rank}
        elif passage_exist and not table_exist:
            error_case['table_none'] += 1
            table_error_cases[qid] = {'answer': answers, 'question': question, 'positive_ctxs': positive_ctxs, 'positive_table_segments': list(positive_table_segments), 'positive_passages':list(positive_passages),'retrieved_graph': retrieval_error_case['retrieved_graph'], 'sorted_retrieved_graph':sorted_retrieved_graph, 'final_node_rank': final_node_rank}
        elif not table_exist and not passage_exist:
            error_case['both_none'] += 1
            both_error_cases[qid] = {'answer': answers, 'question': question, 'positive_ctxs': positive_ctxs, 'positive_table_segments': list(positive_table_segments), 'positive_passages':list(positive_passages),'retrieved_graph': retrieval_error_case['retrieved_graph'], 'sorted_retrieved_graph':sorted_retrieved_graph, 'final_node_rank': final_node_rank}
        else:
            print('error')

    print(error_case)
    
    with open(table_error_case_result_path, 'w') as f:
        json.dump(table_error_cases, f)
    
    with open(passage_error_case_result_path, 'w') as f:
        json.dump(passage_error_cases, f)
    
    with open(both_error_case_result_path, 'w') as f:
        json.dump(both_error_cases, f)
        
    
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