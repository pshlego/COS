import json
from tqdm import tqdm
from pymongo import MongoClient
username = "root"
password = "1234"
client = MongoClient("mongodb://localhost:27017/", username=username, password=password)
db = client["mydatabase"]  # 데이터베이스 선택
print("MongoDB Connected")
cos_graph = json.load(open('/mnt/sdc/shpark/cos_graph.json'))
dev_collection = db["ott_dev_q_to_tables_with_bm25neg"]
total_dev = dev_collection.count_documents({})
print(f"Loading {total_dev} instances...")
dev_instances = [doc for doc in tqdm(dev_collection.find(), total=total_dev)]
print("finish loading dev set")
gold_graph = {}
for i, instance in enumerate(tqdm(dev_instances)):
    gold_passage_set = []
    for positive_ctx in instance['positive_ctxs']:
        if positive_ctx['target_pasg_titles'] != []:
            for gold_passage in positive_ctx['target_pasg_titles']:
                if gold_passage is not None:
                    gold_passage_set.append(gold_passage)
    if instance['positive_table'] not in gold_graph:
        gold_graph[instance['positive_table']] = []
    gold_graph[instance['positive_table']].extend(gold_passage_set)
topk_list = [1,2,3,4,5]
print('cos')
for topk in topk_list:
    cnt = 0
    # get recall
    cos_recall = []
    cos_row_recall_list = []
    for table_chunk_name, instance in tqdm(cos_graph.items()):
        try:
            table_passage_list = []
            for row_num, row_list in instance.items():
                row_value = []
                for row in row_list:
                    row_value.extend(row[:topk])
                table_passage_list.extend(row_value)
            for gold_passage in list(set(gold_graph[table_chunk_name])):
                if gold_passage in set(table_passage_list):
                    cos_recall.append(1)
                else:
                    cos_recall.append(0)
        except:
            cnt += 1

    print('final')
    print(cnt)
    print(topk)
    print(sum(cos_recall)/len(cos_recall))
    print(len(cos_recall))


# cos_graph = json.load(open('/mnt/sdc/shpark/cos_graph.json'))

# mvd_graph = json.load(open('/mnt/sdc/shpark/mvd_graph.json'))
# gold_graph = json.load(open('/mnt/sdc/shpark/gold_graph_2.json'))
# topk_list = [1,2,3,4,5]
# print('cos')
# for topk in topk_list:
#     cnt = 0
#     # get recall
#     cos_recall = []
#     cos_row_recall_list = []
#     for table_chunk_name, instance in tqdm(cos_graph.items()):
#         try:
#             for row_num, row_list in instance.items():
#                 row_value = []
#                 for row in row_list:
#                     row_value.extend(row[:topk])
#                 row_recall = []
#                 if row_num in list(gold_graph[table_chunk_name].keys()):
#                     for passage in gold_graph[table_chunk_name][row_num]:
#                         if passage in row_value:
#                             row_recall.append(1)
#                             cos_recall.append(1)
#                         else:
#                             row_recall.append(0)
#                             cos_recall.append(0)
#                     row_recall_value = sum(row_recall)/len(row_recall)
#                     cos_row_recall_list.append(row_recall_value)
#         except:
#             cnt += 1
#     print('final')
#     print(cnt)
#     print(topk)
#     print(sum(cos_recall)/len(cos_recall))
#     print(sum(cos_row_recall_list)/len(cos_row_recall_list))
#     print(len(cos_recall), len(cos_row_recall_list))

# print('mvd')
# for topk in topk_list:
#     cnt = 0
#     # get recall
#     mvd_recall = []
#     mvd_row_recall_list = []
#     for table_chunk_name, instance in tqdm(mvd_graph.items()):
#         try:
#             for row_num, row_list in instance.items():
#                 row_value = []
#                 for row in row_list:
#                     row_value.extend(row[:topk])
#                 row_recall = []
#                 if row_num in list(gold_graph[table_chunk_name].keys()):
#                     for passage in gold_graph[table_chunk_name][row_num]:
#                         if passage in row_value:
#                             row_recall.append(1)
#                             mvd_recall.append(1)
#                         else:
#                             row_recall.append(0)
#                             mvd_recall.append(0)
#                     row_recall_value = sum(row_recall)/len(row_recall)
#                     mvd_row_recall_list.append(row_recall_value)
#         except:
#             cnt += 1
#     print('final')
#     print(cnt)
#     print(topk)
#     print(sum(mvd_recall)/len(mvd_recall))
#     print(sum(mvd_row_recall_list)/len(mvd_row_recall_list))
#     print(len(mvd_recall), len(mvd_row_recall_list))