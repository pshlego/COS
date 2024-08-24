import json
import re
import copy
import unicodedata
from tqdm import tqdm
from Algorithms.Ours.dpr.utils.tokenizers import SimpleTokenizer
from Algorithms.Ours.dpr.data.qa_validation import has_answer
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data
def remove_accents_and_non_ascii(text):
    # Normalize the text to decompose combined characters
    normalized_text = unicodedata.normalize('NFD', text)
    # Remove all characters that are not ASCII
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining non-letter characters
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,!.?\-]', '', ascii_text)
    return cleaned_text
if __name__ == "__main__":
    retrieved_graphs_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/expanded_query_retrieval_5_5_full.jsonl"#"/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/table_embedding/original_table_new_2.jsonl"#/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/wo_step_2.jsonl" #"/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/final_results_150_10_0_0_2_3_150_28_256.jsonl"
    retrieved_graphs = read_jsonl(retrieved_graphs_path)
    # retrieved_graphs_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/baai_rerank_full_layer_wo_table_retrieval.json"
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

    # with open(retrieved_graphs_path, 'r') as f:
    #     retrieved_graphs = json.load(f)
    
    augment_type_list = ['rerank']
    query_topk_list_1 = [10]#list(range(1,21)) #[5,4,3,2,1] 
    augment_topk_list_1 = [2]
    total_recall_dict = {}
    tokenizer = SimpleTokenizer()
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
        elif augment_type == 'rerank':
            filtered_retrieval_type = ['two_node_graph_reranking', "passage_node_augmentation_1"]
            query_topk_list = [10] 
            augment_topk_list = [2]
        for query_topk in query_topk_list:
            for augment_topk in augment_topk_list:
                error_cases = {}
                setting_key = f"{augment_type}_{query_topk}_{augment_topk}"
                recall_list = []
                for retrieved_graph_info in tqdm(retrieved_graphs, total=len(retrieved_graphs)):
                # for retrieved_graph, qa_datum in tqdm(zip(retrieved_graphs, qa_dataset), total=len(qa_dataset)):
                    qa_datum = retrieved_graph_info["qa data"]
                    retrieved_graph = retrieved_graph_info["retrieved graph"]
                    answers = qa_datum['answers']
                    context = ""
                    # get sorted retrieved graph
                    all_included = []
                    retrieved_table_set = set()
                    retrieved_passage_set = set()
                    # filtered_retrieval_type = ['edge_reranking', 'passage_node_augmentation_1', 'llm_selected']#['edge_reranking', "passage_node_augmentation_1"]
                    # # filtered_retrieval_type = ['edge_retrieval', "passage_node_augmentation_1"]
                    # filtered_retrieval_type_1 = ['edge_reranking', 'llm_selected']#['edge_reranking']
                    # filtered_retrieval_type_2 = ['passage_node_augmentation_1']#["passage_node_augmentation_1"]
                    filtered_retrieval_type = ['edge_reranking', "node_augmentation_1", "passage_node_augmentation_1", "entity_linking_1"]
                    filtered_retrieval_type_1 = ['edge_reranking']#, 'llm_selected'
                    filtered_retrieval_type_2 = ["node_augmentation_1", "passage_node_augmentation_1", "entity_linking_1"]
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
                        node_score = sum(linked_scores) / len(linked_scores)
                        revised_retrieved_graph[node_id]['score'] = node_score
                        
                    retrieved_graph = revised_retrieved_graph
                    sorted_retrieved_graph = sorted(retrieved_graph.items(), key=lambda x: x[1]['score'], reverse=True)
                    for node_id, node_info in sorted_retrieved_graph:
                        if node_info['type'] == 'table segment':
                            # linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation_1' and (x[3] < query_topk) and (x[4] < augment_topk)) or x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[4] < query_topk) and (x[3] < augment_topk)) or x[2] == 'two_node_graph_reranking']
                            
                            # if len(linked_nodes) == 0:
                            #     continue
                            linked_nodes = node_info['linked_nodes']
                            table_id = node_id.split('_')[0]
                            table = table_key_to_content[table_id]
                            chunk_id = table['chunk_id']
                            node_info['chunk_id'] = chunk_id

                            # if len(tokenizer.tokenize(remove_accents_and_non_ascii(context))) > 4096:
                            #     break

                            if table_id not in retrieved_table_set:
                                retrieved_table_set.add(table_id)
                                context += table['text']
                            
                            max_linked_node_id, max_score, _, _, _ = max(linked_nodes, key=lambda x: x[1], default=(None, 0, 'two_node_graph_retrieval', 0, 0))
                            
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
                            context += two_node_graph_text
                            
                        elif node_info['type'] == 'passage':
                            # linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < query_topk) and (x[4] < augment_topk)) or x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation_1' and (x[4] < query_topk) and (x[3] < augment_topk)) or x[2] == 'two_node_graph_reranking']
                            
                            # if len(linked_nodes) == 0:
                            #     continue
                            linked_nodes = node_info['linked_nodes']
                            if node_id in retrieved_passage_set:
                                continue
                            
                            retrieved_passage_set.add(node_id)
                            passage_content = passage_key_to_content[node_id]
                            passage_text = passage_content['title'] + ' ' + passage_content['text']

                            max_linked_node_id, max_score, _, _, _ = max(linked_nodes, key=lambda x: x[1], default=(None, 0, 'two_node_graph_retrieval', 0, 0))
                            table_id = max_linked_node_id.split('_')[0]
                            table = table_key_to_content[table_id]
                            
                            if table_id not in retrieved_table_set:
                                retrieved_table_set.add(table_id)
                                context += table['text']
                                
                            row_id = int(max_linked_node_id.split('_')[1])
                            table_rows = table['text'].split('\n')
                            column_name = table_rows[0]
                            row_values = table_rows[row_id+1]
                            table_segment_text = column_name + '\n' + row_values
                            
                            two_node_graph_text = table_segment_text + '\n' + passage_text
                            context += two_node_graph_text
                    normalized_answers = [remove_accents_and_non_ascii(answer) for answer in answers]
                    normalized_context = remove_accents_and_non_ascii(context)
                    is_has_answer = has_answer(normalized_answers, normalized_context, tokenizer, 'string', max_length=4096)
                    if is_has_answer:
                        recall_list.append(1)
                    else:
                        recall_list.append(0)
                        qa_datum['retrieved_graph'] = retrieved_graph
                        
                        if  "hard_negative_ctxs" in qa_datum:
                            del qa_datum["hard_negative_ctxs"]
                        
                        error_cases[qa_datum['id']] = qa_datum
                total_recall_dict[setting_key] = sum(recall_list) / len(recall_list)
                print(f"Setting: {setting_key}, Recall: {total_recall_dict[setting_key]}")
                # with open(f"/mnt/sdd/shpark/error_case_analysis_results/error_cases_{setting_key}.json", 'w') as f:
                #     json.dump(error_cases, f, indent=4)
    print(total_recall_dict)
        