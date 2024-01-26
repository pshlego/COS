import json
from pymongo import MongoClient
from tqdm import tqdm
username = "root"
password = "1234"
client = MongoClient("mongodb://localhost:27017/", username=username, password=password)
db = client["mydatabase"]  # 데이터베이스 선택
print("MongoDB Connected")

dev_collection = db["ott_dev_q_to_tables_with_bm25neg"]
total_dev = dev_collection.count_documents({})
print(f"Loading {total_dev} instances...")
dev_instances = [doc for doc in tqdm(dev_collection.find(), total=total_dev)]
print("finish loading dev set")

query_results_path = "/mnt/sdd/shpark/cos/models/ott_dev_core_reader_hop1keep200_shard0_of_1.json"
with open(query_results_path, 'r') as file:
    query_results = json.load(file)

error_instances_idxs_path = "/home/shpark/COS/error_analysis/results/error_instances_author.json"
with open(error_instances_idxs_path, 'r') as file:
    error_instances_idxs = json.load(file)

table_instances = []
passage_instances = []
table_passage_instances = []
for i, instance in enumerate(tqdm(dev_instances)):
    if i in error_instances_idxs:
        gold_passage_set = set()
        for positive_ctx in instance['positive_ctxs']:
            for gold_passage in positive_ctx['target_pasg_titles']:
                gold_passage_set.add(gold_passage)
        query_retrieved_results = query_results[i]["ctxs"]
        if list(gold_passage_set)[0] is None:
            if '_id' in instance:
                del instance['_id']
            if 'id' in instance:
                del instance['id']
            table_instances.append(query_results[i])
        else:
            if '_id' in instance:
                del instance['_id']
            if 'id' in instance:
                del instance['id']
            gold_table = False
            for query_retrieved_result in query_retrieved_results:
                retrieved_table_name = '_'.join(query_retrieved_result['id'].split('_')[:-1])
                gold_table_name = instance['positive_table']
                if retrieved_table_name == gold_table_name:
                    gold_table = True
                    break
            if gold_table:
                passage_instances.append(query_results[i])
            else:
                table_passage_instances.append(query_results[i])

with open(f'/home/shpark/COS/error_analysis/results/table_error_instances_author.json', 'w') as file:
    json.dump(table_instances, file, indent=4)
with open(f'/home/shpark/COS/error_analysis/results/passage_error_instances_author.json', 'w') as file:
    json.dump(passage_instances, file, indent=4)
with open(f'/home/shpark/COS/error_analysis/results/table_passage_error_instances_author.json', 'w') as file:
    json.dump(table_passage_instances, file, indent=4)

print(len(table_instances), len(passage_instances), len(table_passage_instances))
