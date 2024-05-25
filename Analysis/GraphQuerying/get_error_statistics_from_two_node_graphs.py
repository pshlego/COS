import json
from tqdm import tqdm
from Algorithms.Ours.dpr.utils.tokenizers import SimpleTokenizer
from Algorithms.Ours.dpr.data.qa_validation import has_answer, _normalize

def get_context(retrieved_graph, table_key_to_content, passage_key_to_content, tokenizer):
    two_node_graph_count = 0
    context = ""
    retrieved_table_set = set()
    retrieved_passage_set = set()
    sorted_retrieved_graph = sorted(retrieved_graph.items(), key=lambda x: x[1]['score'], reverse=True)
    final_node_rank = len(sorted_retrieved_graph)
    for node_rank, (node_id, node_info) in enumerate(sorted_retrieved_graph):
        if len(node_info['linked_nodes']) == 0:
            continue

        if node_info['type'] == 'table segment':
            table_id = node_id.split('_')[0]
            table = table_key_to_content[table_id]
            
            if table_id not in retrieved_table_set:
                if two_node_graph_count == 50:
                    continue
                retrieved_table_set.add(table_id)
                context += table['text']
                two_node_graph_count += 1
                final_node_rank = node_rank
            max_linked_node_id, max_score, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default=(None, 0, 'two_node_graph_retrieval'))

            if max_linked_node_id in retrieved_passage_set:
                continue
            if two_node_graph_count == 50:
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
            final_node_rank = node_rank
        elif node_info['type'] == 'passage':
            if node_id in retrieved_passage_set:
                continue
            if two_node_graph_count == 50:
                continue
            retrieved_passage_set.add(node_id)
            passage_content = passage_key_to_content[node_id]
            passage_text = passage_content['title'] + ' ' + passage_content['text']

            max_linked_node_id, max_score, a = max(node_info['linked_nodes'], key=lambda x: x[1], default=(None, 0, 'two_node_graph_retrieval'))
            table_id = max_linked_node_id.split('_')[0]
            table = table_key_to_content[table_id]
            
            if table_id not in retrieved_table_set:
                if two_node_graph_count == 50:
                    continue
                retrieved_table_set.add(table_id)
                context += table['text']
                two_node_graph_count += 1
                final_node_rank = node_rank
                
            row_id = int(max_linked_node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            table_segment_text = column_name + '\n' + row_values
            
            two_node_graph_text = table_segment_text + '\n' + passage_text
            context += two_node_graph_text
            two_node_graph_count += 1
            final_node_rank = node_rank
            # if len(tokenizer.tokenize(_normalize(context)).words(uncased=True)) > 4096:
            #     final_node_rank = node_rank
            #     break

    return context, sorted_retrieved_graph, final_node_rank

if __name__ == '__main__':
    error_cases_path = "/mnt/sdd/shpark/output/integrated_graph_augmented_both_3_1_v9.json" #"/home/shpark/OTT_QA_Workspace/Analysis/GraphQueryResults/error_cases.json"
    data_graph_error_cases_path = "/home/shpark/OTT_QA_Workspace/Analysis/GraphQueryResults/data_graph_error_cases.json"
    table_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    passage_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json"
    
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
        context, sorted_retrieved_graph, final_node_rank = get_context(retrieved_graph, table_key_to_content, passage_key_to_content, tokenizer)
        is_has_answer = has_answer(answers, context, tokenizer, 'string')
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
                if node_id in positive_passages:
                    passage_exist = True
            
        if table_exist and not passage_exist:
            error_case['passage_none'] += 1
        elif passage_exist and not table_exist:
            error_case['table_none'] += 1
        elif not table_exist and not passage_exist:
            error_case['both_none'] += 1
    print(error_case)

        
    
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