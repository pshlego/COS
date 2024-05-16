import json
# Analysis Settings
hierarchical_levels = ['star', 'edge', 'both'] # ['star', 'edge']
is_colbert, search_space = True, None # True, larger
limits = [1, 5, 10, 20, 50, 100]
k = 100

for is_colbert in [True]:
    # Analyze Error Instances for Each Hierarchical Level
    if is_colbert:
        search_space_list = ['larger']
    else:
        search_space_list = [None]
    
    for search_space in search_space_list:#, 'larger'
        for level in hierarchical_levels:
            answer_recall = [0]*len(limits)
            space_suffix = f"_{search_space}" if is_colbert and search_space is not None else ""
            
            if is_colbert:
                query_results_path = f"/mnt/sdc/shpark/graph/query_results_2/colbert_graph_query_results_fix_table_error_k_500_{level}{space_suffix}.json"
            else:
                query_results_path = f"/mnt/sdd/shpark/graph/query_results_2/cos_graph_query_results_fix_table_error_k_500_{level}{space_suffix}.json"
            
            with open(query_results_path, 'r') as file:
                query_results = json.load(file)
            
            if is_colbert:
                print(f"ColBERT {level} {search_space} Accuracy:")
            else:
                print(f"Cos {level} Accuracy:")
            
            for query_result in query_results:
                all_included = query_result['ctxs']
                for l, limit in enumerate(limits):
                    if any([ctx['has_answer'] for ctx in all_included[:limit]]):
                        answer_recall[l] += 1
            
            for l, limit in enumerate(limits):
                print ('answer recall', limit, answer_recall[l]/len(query_results))
            