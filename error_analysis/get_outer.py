import json

with open('/home/shpark/COS/error_analysis/results/cos_error_instances.json', 'r') as file:
    cos_error_instances = json.load(file)

cos_qid_list = []
# ct_length_list = []
for cos_error_instance in cos_error_instances:
    cos_qid_list.append(cos_error_instance['id'])
    # ct_length_list.append(len(cos_error_instance['ctxs']))
# with open('/mnt/sdc/shpark/graph/query_results/colbert_graph_query_results_fix_table_error_star.json', 'r') as file:
#     colbert_both_table_passage_error_instances = json.load(file)
with open('/mnt/sdc/shpark/graph/query_results_2/colbert_graph_query_results_fix_table_error_k_1000_length_512_edge_larger.json', 'r') as file:
    colbert_both_table_passage_error_instances = json.load(file)
outer_error_instances = []
ct_length_list_2 = []
for colbet_both_table_passage_error_instance in colbert_both_table_passage_error_instances:
    # ct_length_list_2.append(len(colbet_both_table_passage_error_instance['ctxs']))
    if colbet_both_table_passage_error_instance['id'] not in cos_qid_list:
        outer_error_instances.append(colbet_both_table_passage_error_instance)
        if len(colbet_both_table_passage_error_instance['ctxs']) != 100:
            print(colbet_both_table_passage_error_instance['id'])
with open('/home/shpark/COS/error_analysis/outer_top_1000/ㅁㄴㅇ.json', 'w') as file:
    json.dump(outer_error_instances, file, indent=4)