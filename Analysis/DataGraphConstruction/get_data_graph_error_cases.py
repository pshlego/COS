import json
from tqdm import tqdm

if __name__ == "__main__":
    query_path = "/mnt/sdf/OTT-QAMountSpace/AnalysisResults/Ours/GraphQuerier/error_cases/original_table_new_2.json" #"/mnt/sdd/shpark/experimental_results/error_cases/baai_rerank_first.json"#"/mnt/sdd/shpark/experimental_results/error_cases/baai_reranker.json"#f"/mnt/sdd/shpark/experimental_results/error_case_two_node_graph/error_cases_none_1_1_v2.json"#"/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_dev_q_to_tables_with_bm25neg.json"
    generated_data_graph_path = "/mnt/sdf/OTT-QAMountSpace/AnalysisResults/COS/DataGraphConstructor/table_chunks_to_passages_cos_table_passage.json"

    with open(generated_data_graph_path, 'r') as f:
        generated_data_graphs = json.load(f)

    generated_table_chunk_to_passages = {}
    for generated_data_graph in generated_data_graphs:
        generated_table_chunk_to_passages[generated_data_graph['table_chunk_id']] = generated_data_graph

    with open(query_path, 'r') as f:
        queries = json.load(f)

    error_cases = []
    
    for query_info in tqdm(queries):
        table_chunk_id_to_linked_passages = {}
        for positive_ctx in query_info['positive_ctxs']:
            table_chunk_id = positive_ctx['chunk_id']
            
            if table_chunk_id not in table_chunk_id_to_linked_passages:
                table_chunk_id_to_linked_passages[table_chunk_id] = []
            
            for answer_node in positive_ctx['answer_node']:
                if answer_node[3] == 'passage':
                    table_chunk_id_to_linked_passages[table_chunk_id].append(answer_node[2].replace('/wiki/','').replace('_', ' '))
        
        table_chunk_id_to_retrieved_passages = {}
        for retreived_node_id, retrieved_node_info in query_info['retrieved_graph'].items():
            if retrieved_node_info['type'] != 'table segment':
                continue
            
            table_chunk_id = retrieved_node_info['chunk_id']
            if table_chunk_id not in table_chunk_id_to_linked_passages:
                continue
            
            if table_chunk_id not in table_chunk_id_to_retrieved_passages:
                table_chunk_id_to_retrieved_passages[table_chunk_id] = []
            
            table_chunk_id_to_retrieved_passages[table_chunk_id].extend([linked_node[0] for linked_node in retrieved_node_info['linked_nodes']])
            
        for table_chunk_id, linked_passages in table_chunk_id_to_linked_passages.items():
            
            if len(linked_passages) == 0:
                continue
            
            generated_data_graph = generated_table_chunk_to_passages[table_chunk_id]
            
            linked_passages_in_generated_graph = []
            
            for link in generated_data_graph['results']:
                linked_passages_in_generated_graph.extend(link['retrieved'][:1])
            
            if table_chunk_id in table_chunk_id_to_retrieved_passages:
                linked_passages_in_generated_graph.extend(table_chunk_id_to_retrieved_passages[table_chunk_id])
            
            if set(linked_passages).intersection(set(linked_passages_in_generated_graph)) == set():
                query_info['linked_passages_in_generated_graph'] = linked_passages_in_generated_graph
                if 'hard_negative_ctxs' in query_info:
                    del query_info['hard_negative_ctxs']
                error_cases.append(query_info)
    
    with open("/home/shpark/OTT_QA_Workspace/Analysis/GraphQueryResults/original_table_new_2.json", 'w') as f:
        json.dump(error_cases, f, indent=4)