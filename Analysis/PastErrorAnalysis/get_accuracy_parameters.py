import json
# Analysis Settings
hierarchical_levels = ['star', 'edge', 'both'] # ['star', 'edge']
is_colbert, search_space = True, None # True, larger
limits = [1, 5, 10, 20, 50, 100]
k = 100

for selection_method in ['thr', 'topp', 'topk']:
    if selection_method == 'thr':
        parameters = [60.94, 56.85, 54.61, 52.75]
        # parameters = [60.94]
    elif selection_method == 'topp':
        parameters = [0.7, 0.5, 0.3]
        # parameters = [0.9]
    elif selection_method == 'topk':
        parameters = [1, 2, 3, 4, 5]
        # parameters = [1]
    for parameter in parameters:
        parameter_name = str(parameter).replace('.', '_')
        answer_recall = [0]*len(limits)
        result_path = "/home/shpark/mnt_sdc/shpark/query_result/colbert/selection_algorithms/" + selection_method + "/colbert_graph_query_results_fix_table_error_k_700_length_512_filtered_both_larger"
        time_path = "/home/shpark/mnt_sdc/shpark/query_result/colbert/selection_algorithms/" + selection_method + "/colbert_graph_query_results_fix_table_error_k_700_length_512_filtered_both_larger"
        if selection_method == 'topk':
            result_path += f'_{selection_method}_{parameter_name}.json'
            time_path += '_{selection_method}_{parameter_name}_time.json'
        elif selection_method == 'topp':
            result_path += f'_{selection_method}_{parameter_name}.json'
            time_path += '_{selection_method}_{parameter_name}_time.json'
        elif selection_method == 'thr':
            result_path += f'_{selection_method}_{parameter_name}.json'
            time_path += '_{selection_method}_{parameter_name}_time.json'
        else:
            result_path += '.json'
            time_path += '_time.json'
        
        with open(result_path, 'r') as file:
            query_results = json.load(file)
        
        print(f"ColBERT both {search_space} {selection_method} {parameter_name} Accuracy:")
        
        for query_result in query_results:
            all_included = query_result['ctxs']
            for l, limit in enumerate(limits):
                if any([ctx['has_answer'] for ctx in all_included[:limit]]):
                    answer_recall[l] += 1
        
        for l, limit in enumerate(limits):
            print ('answer recall', limit, answer_recall[l]/len(query_results))
            