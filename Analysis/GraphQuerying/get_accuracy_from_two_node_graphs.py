import re
import json
import copy
import unicodedata
from tqdm import tqdm
from Algorithms.Ours.dpr.utils.tokenizers import SimpleTokenizer
from Algorithms.Ours.dpr.data.qa_validation import has_answer
def remove_accents_and_non_ascii(text):
    # Normalize the text to decompose combined characters
    normalized_text = unicodedata.normalize('NFD', text)
    # Remove all characters that are not ASCII
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining non-letter characters
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,!.?\-]', '', ascii_text)
    return cleaned_text

if __name__ == "__main__":
    retrieved_graphs_path = "/mnt/sdd/shpark/output/title_wo_short_column_short_value_title_w_v3.json"
    table_data_path= "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    passage_data_path= "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json"
    passage_ids_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/ColBERT_Embedding_Dataset/passage_cos_version/index_to_chunk_id.json"
    qa_dataset_path=  "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_dev_q_to_tables_with_bm25neg.json"
    qa_dataset = json.load(open(qa_dataset_path))
    qa_dataset = qa_dataset
    print(f"Loading corpus...")
    
    table_key_to_content = {}
    table_contents = json.load(open(table_data_path))
    for table_key, table_content in enumerate(table_contents):
        table_key_to_content[str(table_key)] = table_content
    
    passage_key_to_content = {}
    passage_contents = json.load(open(passage_data_path))
    for passage_content in passage_contents:
        passage_key_to_content[passage_content['title']] = passage_content

    with open(retrieved_graphs_path, 'r') as f:
        retrieved_graphs = json.load(f)
    
    augment_type_list = ['passage']
    passage_query_topk_list_1 = [10]
    passage_augment_topk_list_1 = [2]
    table_query_topk_list_1 = [1]
    table_augment_topk_list_1 = [1]
    total_recall_dict = {}
    tokenizer = SimpleTokenizer()
    #error_cases = {}
    for augment_type in augment_type_list:
        if augment_type == 'both':
            filtered_retrieval_type = ['two_node_graph_retrieval', 'table_segment_node_augmentation', 'passage_node_augmentation']
            passage_query_topk_list_1 = passage_query_topk_list_1
            passage_augment_topk_list_1 = passage_augment_topk_list_1
            table_query_topk_list_1 = table_query_topk_list_1
            table_augment_topk_list_1 = table_augment_topk_list_1
        elif augment_type == 'table':
            filtered_retrieval_type = ['two_node_graph_retrieval', 'table_segment_node_augmentation']
            passage_query_topk_list_1 = passage_query_topk_list_1
            passage_augment_topk_list_1 = passage_augment_topk_list_1
            table_query_topk_list_1 = table_query_topk_list_1
            table_augment_topk_list_1 = table_augment_topk_list_1
        elif augment_type == 'passage':
            filtered_retrieval_type = ['two_node_graph_retrieval', 'passage_node_augmentation']
            passage_query_topk_list_1 = passage_query_topk_list_1
            passage_augment_topk_list_1 = passage_augment_topk_list_1
            table_query_topk_list_1 = table_query_topk_list_1
            table_augment_topk_list_1 = table_augment_topk_list_1
        elif augment_type == 'none':
            filtered_retrieval_type = ['two_node_graph_retrieval']
            query_topk_list = [1] 
            augment_topk_list = [1]
            
        for passage_query_topk in passage_query_topk_list_1:
            for table_query_topk in table_query_topk_list_1:
                for passage_augment_topk in passage_augment_topk_list_1:
                    for table_augment_topk in table_augment_topk_list_1:
                        error_cases = {}
                        setting_key = f"{augment_type}_passage_{passage_query_topk}_{passage_augment_topk}_table_{table_query_topk}_{table_augment_topk}"
                        recall_list = []
                        revised_retrieved_graphs = []
                        for retrieved_graph in retrieved_graphs:
                            revised_retrieved_graph = {}
                            for node_id, node_info in retrieved_graph.items():
                                if node_info['type'] == 'table segment':
                                    linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation' and (x[3] < passage_query_topk) and (x[4] < passage_augment_topk)) or x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[4] < table_query_topk) and (x[3] < table_augment_topk)) or x[2] == 'two_node_graph_retrieval']
                                elif node_info['type'] == 'passage':
                                    linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < table_query_topk) and (x[4] < table_augment_topk)) or x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation' and (x[4] < passage_query_topk) and (x[3] < passage_augment_topk)) or x[2] == 'two_node_graph_retrieval']
                                
                                if len(linked_nodes) == 0:
                                    continue
                                
                                revised_retrieved_graph[node_id] = copy.deepcopy(node_info)
                                revised_retrieved_graph[node_id]['linked_nodes'] = linked_nodes
                                
                                linked_scores = [linked_node[1] for linked_node in linked_nodes]
                                node_score = max(linked_scores)
                                revised_retrieved_graph[node_id]['score'] = node_score
                                
                            revised_retrieved_graphs.append(revised_retrieved_graph)
                            
                        for revised_retrieved_graph, qa_datum in zip(revised_retrieved_graphs, qa_dataset):
                            two_node_graph_count = 0
                            answers = qa_datum['answers']
                            # if qa_datum['id'] not in ['e46aefefda60a34d', 'dd6a935fc3ca3446', 'b0fe5731a19a8fb9', '822fa10c6b80eb78', '960022483a9d97c4', 'a70cfc60e541827f', '622783be803c181c', '08d4e37cbc7bb2c5', '1d7f52d9c59e6fc5', 'd8338761374ef6a8', '90b0d5dcf0eaf6b5', '1f1484f82a7625dd']:
                            #     continue
                            # if qa_datum['id'] not in ['b5cfc92181b6511b', '6ad2c846a3dbab5c', '7a158221e7c9b6eb']:
                            #     continue
                            context = ""
                            # get sorted retrieved graph
                            all_included = []
                            retrieved_table_set = set()
                            retrieved_passage_set = set()
                            sorted_retrieved_graph = sorted(revised_retrieved_graph.items(), key=lambda x: x[1]['score'], reverse=True)
                            
                            for node_id, node_info in sorted_retrieved_graph:
                                if node_info['type'] == 'table segment':
                                    
                                    table_id = node_id.split('_')[0]
                                    table = table_key_to_content[table_id]
                                    chunk_id = table['chunk_id']
                                    node_info['chunk_id'] = chunk_id
                                    
                                    if table_id not in retrieved_table_set:
                                        retrieved_table_set.add(table_id)
                                        
                                        if two_node_graph_count == 50:
                                            continue
                                        
                                        context += table['text']
                                        two_node_graph_count += 1
                                        
                                    max_linked_node_id, max_score, _, _, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default=(None, 0, 'two_node_graph_retrieval', 0, 0))
                                    
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
                                    
                                    two_node_graph_text = table_segment_text + '\n' + passage_text
                                    
                                    if two_node_graph_count == 50:
                                        continue
                                    
                                    two_node_graph_count += 1
                                    context += two_node_graph_text
                                    
                                elif node_info['type'] == 'passage':

                                    if node_id in retrieved_passage_set:
                                        continue

                                    max_linked_node_id, max_score, _, _, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default=(None, 0, 'two_node_graph_retrieval', 0, 0))
                                    table_id = max_linked_node_id.split('_')[0]
                                    table = table_key_to_content[table_id]
                                    
                                    if table_id not in retrieved_table_set:
                                        retrieved_table_set.add(table_id)
                                        if two_node_graph_count == 50:
                                            continue
                                        context += table['text']
                                        two_node_graph_count += 1

                                    row_id = int(max_linked_node_id.split('_')[1])
                                    table_rows = table['text'].split('\n')
                                    column_name = table_rows[0]
                                    row_values = table_rows[row_id+1]
                                    table_segment_text = column_name + '\n' + row_values
                                    
                                    if two_node_graph_count == 50:
                                        continue

                                    retrieved_passage_set.add(node_id)
                                    passage_content = passage_key_to_content[node_id]
                                    passage_text = passage_content['title'] + ' ' + passage_content['text']
                                    
                                    two_node_graph_text = table_segment_text + '\n' + passage_text
                                    context += two_node_graph_text
                                    two_node_graph_count += 1
                            
                            normalized_context = remove_accents_and_non_ascii(context)
                            normalized_answers = [remove_accents_and_non_ascii(answer) for answer in answers]
                            is_has_answer = has_answer(normalized_answers, normalized_context, tokenizer, 'string')
                            
                            if is_has_answer:
                                recall_list.append(1)
                            else:
                                recall_list.append(0)
                                new_qa_datum = copy.deepcopy(qa_datum)
                                new_qa_datum['retrieved_graph'] = revised_retrieved_graph
                                
                                if  "hard_negative_ctxs" in new_qa_datum:
                                    del new_qa_datum["hard_negative_ctxs"]
                                
                                error_cases[qa_datum['id']] = new_qa_datum


                        total_recall_dict[setting_key] = sum(recall_list) / len(recall_list)
                        print(f"Setting: {setting_key}, Recall: {total_recall_dict[setting_key]}")
                        
                        with open(f"/mnt/sdd/shpark/error_case_two_node_graph/error_cases_{setting_key}_sota.json", 'w') as f:
                            json.dump(error_cases, f, indent=4)
                
    print(total_recall_dict)
    print(max(list(total_recall_dict.values())))
    with open("/mnt/sdd/shpark/error_case_two_node_graph/different_parameter.json", 'w') as f:
        json.dump(total_recall_dict, f, indent=4)
        