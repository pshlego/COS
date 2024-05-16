import json

with open('/mnt/sdd/shpark/colbert/data/topp_0_1.json', 'r') as file:
    topp_0_1 = json.load(file)
with open('/mnt/sdd/shpark/colbert/data/topk_1.json', 'r') as file:
    topk_1 = json.load(file)

cos_qid_list = []
new_results = []
for cos_result in cos_results:
    ctx_list = []
    has_answer = False
    if len(cos_result['ctxs']) < 100:
        print(cos_result)
#     for ctx in cos_result['ctxs']:
#         if ctx['has_answer']:
#             ctx_list.append(ctx)
#             has_answer = True
#     if has_answer:
#         cos_result['ctxs'] = ctx_list
#         new_results.append(cos_result)

# with open('/home/shpark/COS/error_analysis/results_all/ott_dev_core_reader_hop1keep200_shard0_of_1_wo_expanded_query_retrieval_error_instances.json', 'r') as file:
#     colbert_both_table_passage_error_instances = json.load(file)

# error_instances = []
# for colbet_both_table_passage_error_instance in colbert_both_table_passage_error_instances:
#     if colbet_both_table_passage_error_instance['id'] in cos_qid_list:
#         error_instances.append(colbet_both_table_passage_error_instance)

with open('/home/shpark/COS/error_analysis/results_all/answer_only.json', 'w') as file:
    json.dump(new_results, file, indent=4)
# with open('/home/shpark/COS/error_analysis/results_all/error_instances_wo_expanded_query_retrieval_module.json', 'w') as file:
#     json.dump(error_instances, file, indent=4)