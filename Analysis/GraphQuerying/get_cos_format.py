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
    cos_format_results = []
    augment_type_list = ['both']
    query_topk_list_1 = [10]#list(range(1,31))#[5,4,3,2,1] 
    augment_topk_list_1 = [2]#[1,2,3,4,5]
    total_recall_dict = {}
    tokenizer = SimpleTokenizer()
    #error_cases = {}
    for augment_type in augment_type_list:
        if augment_type == 'both':
            filtered_retrieval_type = ['two_node_graph_retrieval', 'table_segment_node_augmentation', 'passage_node_augmentation']
            query_topk_list = query_topk_list_1
            augment_topk_list = augment_topk_list_1
        elif augment_type == 'table':
            filtered_retrieval_type = ['two_node_graph_retrieval', 'table_segment_node_augmentation']
            query_topk_list = query_topk_list_1
            augment_topk_list = augment_topk_list_1
        elif augment_type == 'passage':
            filtered_retrieval_type = ['two_node_graph_retrieval', 'passage_node_augmentation']
            query_topk_list = query_topk_list_1 
            augment_topk_list = augment_topk_list_1
        elif augment_type == 'none':
            filtered_retrieval_type = ['two_node_graph_retrieval']
            query_topk_list = [1] 
            augment_topk_list = [1]
        for query_topk in query_topk_list:
            for augment_topk in augment_topk_list:
                error_cases = {}
                setting_key = f"{augment_type}_{query_topk}_{augment_topk}"
                recall_list = []
                revised_retrieved_graphs = []
                for retrieved_graph in retrieved_graphs:
                    revised_retrieved_graph = {}
                    for node_id, node_info in retrieved_graph.items():
                        if node_info['type'] == 'table segment':
                            linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation' and (x[3] < query_topk) and (x[4] < augment_topk)) or x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[4] < 2) and (x[3] < 2)) or x[2] == 'two_node_graph_retrieval']
                        elif node_info['type'] == 'passage':
                            linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < 2) and (x[4] < 2)) or x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation' and (x[4] < query_topk) and (x[3] < augment_topk)) or x[2] == 'two_node_graph_retrieval']
                        
                        if len(linked_nodes) == 0:
                            continue
                        
                        revised_retrieved_graph[node_id] = copy.deepcopy(node_info)
                        revised_retrieved_graph[node_id]['linked_nodes'] = linked_nodes
                        
                        linked_scores = [linked_node[1] for linked_node in linked_nodes]
                        node_score = max(linked_scores)
                        revised_retrieved_graph[node_id]['score'] = node_score
                        
                    revised_retrieved_graphs.append(revised_retrieved_graph)
                for revised_retrieved_graph, qa_datum in tqdm(zip(revised_retrieved_graphs, qa_dataset), total=len(qa_dataset)):
                    cos_format_result = copy.deepcopy(qa_datum)
                    two_node_graph_count = 0
                    answers = qa_datum['answers']

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
                                
                                # normalized_context = remove_accents_and_non_ascii(table['text'])
                                # normalized_answers = [remove_accents_and_non_ascii(answer) for answer in answers]    
                                # has_answer = has_answer(normalized_answers, normalized_context, tokenizer, 'string')
                                
                                all_included.append({'id': chunk_id, 'title': table['title'], 'text': table['text']})
                                
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
                            
                            all_included.append({'id': chunk_id, 'title': table['title'], 'text': two_node_graph_text})
                            two_node_graph_count += 1
                            
                        elif node_info['type'] == 'passage':
                            # linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < query_topk) and (x[4] < augment_topk)) or x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation' and (x[4] < query_topk) and (x[3] < augment_topk)) or x[2] == 'two_node_graph_retrieval']
                            
                            # if len(linked_nodes) == 0:
                            #     continue

                            if node_id in retrieved_passage_set:
                                continue

                            max_linked_node_id, max_score, _, _, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default=(None, 0, 'two_node_graph_retrieval', 0, 0))
                            table_id = max_linked_node_id.split('_')[0]
                            table = table_key_to_content[table_id]
                            
                            if table_id not in retrieved_table_set:
                                retrieved_table_set.add(table_id)
                                if two_node_graph_count == 50:
                                    continue
                                all_included.append({'id': chunk_id, 'title': table['title'], 'text': table['text']})
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
                            all_included.append({'id': chunk_id, 'title': table['title'], 'text': two_node_graph_text})
                            two_node_graph_count += 1
                    
                    cos_format_result['ctxs'] = all_included
                    cos_format_results.append(cos_format_result)
                
                with open(f"/mnt/sdd/shpark/output/cos_format_{setting_key}_short_v3.json", 'w') as f:
                    json.dump(cos_format_results, f, indent=4)
        