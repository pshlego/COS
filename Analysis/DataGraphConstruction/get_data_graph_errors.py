import json
from tqdm import tqdm

if __name__ == "__main__":
    qa_dataset_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_dev_q_to_tables_with_bm25neg.json"
    generated_data_graph_path = "/mnt/sdf/OTT-QAMountSpace/AnalysisResults/COS/DataGraphConstructor/table_chunks_to_passages_cos_table_passage.json"

    with open(generated_data_graph_path, 'r') as f:
        generated_data_graphs = json.load(f)

    generated_table_chunk_to_passages = {}
    for generated_data_graph in generated_data_graphs:
        generated_table_chunk_to_passages[generated_data_graph['table_chunk_id']] = generated_data_graph

    queries = json.load(open(qa_dataset_path))

    error_cases = []
    
    for query_info in queries:
        table_chunk_id_to_linked_passages = {}
        for positive_ctx in query_info['positive_ctxs']:
            table_chunk_id = positive_ctx['chunk_id']
            
            if table_chunk_id not in table_chunk_id_to_linked_passages:
                table_chunk_id_to_linked_passages[table_chunk_id] = []
            
            for answer_node in positive_ctx['answer_node']:
                if answer_node[3] == 'passage':
                    table_chunk_id_to_linked_passages[table_chunk_id].append(answer_node[2].replace('/wiki/','').replace('_', ' '))
        
        positive_passage_list = []
        linked_passages_list = []
        for table_chunk_id, linked_passages in table_chunk_id_to_linked_passages.items():
            
            if len(linked_passages) == 0:
                continue
            
            positive_passage_list.extend(linked_passages)
            
            generated_data_graph = generated_table_chunk_to_passages[table_chunk_id]
            
            for link in generated_data_graph['results']:
                linked_passages_list.extend(link['retrieved'][:1])
        
        if len(positive_passage_list) == 0:
            continue
        
        if set(positive_passage_list).intersection(set(linked_passages_list)) == set():
            query_info['positive_passages'] = list(set(positive_passage_list))
            error_cases.append(query_info)
    
    with open("/home/shpark/OTT_QA_Workspace/data_graph_error_case.json", 'w') as f:
        json.dump(error_cases, f, indent=4)