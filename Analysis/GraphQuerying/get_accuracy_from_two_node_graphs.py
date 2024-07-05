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

def assign_scores(graph, retrieval_type = "None"):
    for node_id, node_info in graph.items():

        # if 'score' in node_info:
        #     continue
        if retrieval_type is not None:
            filtered_retrieval_type = ['edge_retrieval', 'passage_node_augmentation_0', 'table_segment_node_augmentation_0']
            linked_scores = [linked_node[1] for linked_node in node_info['linked_nodes'] if linked_node[2] not in filtered_retrieval_type]
        else:
            linked_scores = [linked_node[1] for linked_node in node_info['linked_nodes']]
        node_score = max(linked_scores)
        # if node_scoring_method == 'min':
        #     node_score = min(linked_scores)
        # elif node_scoring_method == 'max':
        #     node_score = max(linked_scores)
        # elif node_scoring_method == 'mean':
        #     node_score = sum(linked_scores) / len(linked_scores)
        
        graph[node_id]['score'] = node_score
if __name__ == "__main__":
    retrieved_graphs_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/reranking_first_w_original_reranker.json"#"/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/reranking_w_original_reranker.json"#"/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/reranking_first_w_finetuned_reranker.json"#"/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/reranking_first_w_finetuned_reranker.json"#"/mnt/sdd/shpark/experimental_results/output/add_reranking_passage_augmentation_150_10_2_trained_v2_faster.json" #"/mnt/sdd/shpark/output/integrated_graph_augmented_passage_10_2_v15_20_fix_scoring.json"
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
    
    augment_type_list = ['rerank']
    passage_query_topk_list_1 = [10]
    passage_augment_topk_list_1 = [2]
    table_query_topk_list_1 = [1]
    table_augment_topk_list_1 = [1]
    total_recall_dict = {}
    tokenizer = SimpleTokenizer()
    #error_cases = {}
    for augment_type in augment_type_list:
        if augment_type == 'both':
            filtered_retrieval_type = ['edge_retrieval', 'table_segment_node_augmentation', 'passage_node_augmentation_1']
            passage_query_topk_list_1 = passage_query_topk_list_1
            passage_augment_topk_list_1 = passage_augment_topk_list_1
            table_query_topk_list_1 = table_query_topk_list_1
            table_augment_topk_list_1 = table_augment_topk_list_1
        elif augment_type == 'table':
            filtered_retrieval_type = ['edge_retrieval', 'table_segment_node_augmentation']
            passage_query_topk_list_1 = passage_query_topk_list_1
            passage_augment_topk_list_1 = passage_augment_topk_list_1
            table_query_topk_list_1 = table_query_topk_list_1
            table_augment_topk_list_1 = table_augment_topk_list_1
        elif augment_type == 'passage':
            filtered_retrieval_type = ['edge_retrieval', 'passage_node_augmentation_1']
            passage_query_topk_list_1 = passage_query_topk_list_1
            passage_augment_topk_list_1 = passage_augment_topk_list_1
            table_query_topk_list_1 = table_query_topk_list_1
            table_augment_topk_list_1 = table_augment_topk_list_1
        elif augment_type == 'none':
            filtered_retrieval_type = ['edge_retrieval']
            query_topk_list = [1] 
            augment_topk_list = [1]
        elif augment_type == 'rerank':
            filtered_retrieval_type = ['edge_reranking', "passage_node_augmentation_1"]
            query_topk_list = [10] 
            augment_topk_list = [2]

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

                                # if 'passage_node_augmentation_0' in [x[2] for x in node_info['linked_nodes']]:
                                #     passage_augment_topk = 2
                                # else:
                                #     passage_augment_topk = 1                    

                                if node_info['type'] == 'table segment':
                                    linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation_1' and (x[3] < passage_query_topk) and (x[4] < passage_augment_topk)) or x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[4] < table_query_topk) and (x[3] < table_augment_topk)) or x[2] == 'edge_reranking']
                                elif node_info['type'] == 'passage':
                                    linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < table_query_topk) and (x[4] < table_augment_topk)) or x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation_1' and (x[4] < passage_query_topk) and (x[3] < passage_augment_topk)) or x[2] == 'edge_reranking']
                                
                                if len(linked_nodes) == 0:
                                    continue
                                
                                revised_retrieved_graph[node_id] = copy.deepcopy(node_info)
                                revised_retrieved_graph[node_id]['linked_nodes'] = linked_nodes
                                
                                linked_scores = [linked_node[1] for linked_node in linked_nodes]
                                node_score = max(linked_scores)
                                revised_retrieved_graph[node_id]['score'] = node_score
                                
                            revised_retrieved_graphs.append(revised_retrieved_graph)
                            
                        
                        for revised_retrieved_graph, qa_datum in zip(revised_retrieved_graphs, qa_dataset):
                        # for revised_retrieved_graph, qa_datum in zip(retrieved_graphs, qa_dataset):
                            # if qa_datum['id'] != 'e01c6b4a614d2d38':
                            #     continue
                            edge_count = 0
                            answers = qa_datum['answers']
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
                                        
                                        if edge_count == 50:
                                            continue
                                        
                                        context += table['text']
                                        edge_count += 1
                                        
                                    max_linked_node_id, max_score, _, _, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default=(None, 0, 'edge_retrieval', 0, 0))
                                    
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
                                    
                                    if edge_count == 50:
                                        continue
                                    
                                    edge_count += 1
                                    context += edge_text
                                    
                                elif node_info['type'] == 'passage':

                                    if node_id in retrieved_passage_set:
                                        continue

                                    max_linked_node_id, max_score, _, _, _ = max(node_info['linked_nodes'], key=lambda x: x[1], default=(None, 0, 'edge_retrieval', 0, 0))
                                    table_id = max_linked_node_id.split('_')[0]
                                    table = table_key_to_content[table_id]
                                    
                                    if table_id not in retrieved_table_set:
                                        retrieved_table_set.add(table_id)
                                        if edge_count == 50:
                                            continue
                                        context += table['text']
                                        edge_count += 1

                                    row_id = int(max_linked_node_id.split('_')[1])
                                    table_rows = table['text'].split('\n')
                                    column_name = table_rows[0]
                                    row_values = table_rows[row_id+1]
                                    table_segment_text = column_name + '\n' + row_values
                                    
                                    if edge_count == 50:
                                        continue

                                    retrieved_passage_set.add(node_id)
                                    passage_content = passage_key_to_content[node_id]
                                    passage_text = passage_content['title'] + ' ' + passage_content['text']
                                    
                                    edge_text = table_segment_text + '\n' + passage_text
                                    context += edge_text
                                    edge_count += 1
                            
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
                        
                        with open(f"/mnt/sdd/shpark/experimental_results/error_cases/150_10_2_w_reranking_original.json", 'w') as f:
                            json.dump(error_cases, f, indent=4)
                
    print(total_recall_dict)
    print(max(list(total_recall_dict.values())))
    with open("/mnt/sdd/shpark/error_case_edge/different_parameter.json", 'w') as f:
        json.dump(total_recall_dict, f, indent=4)
        