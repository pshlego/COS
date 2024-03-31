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

query_results_path = "/home/shpark/mnt_sdc/shpark/cos/cos/models/ott_dev_core_reader_hop1keep200_shard0_of_1_wo_expanded_query_retrieval.json"
with open(query_results_path, 'r') as file:
    query_results = json.load(file)
error_instances = []
for i, instance in enumerate(tqdm(dev_instances)):
    query_result_ctxs = query_results[i]["ctxs"]
    has_answer = False
    for query_result_ctx in query_result_ctxs:
        
        if query_result_ctx["has_answer"]:
            has_answer = True
            break
    if not has_answer:
        error_instances.append(query_results[i])
    
print(len(error_instances))
with open('/home/shpark/COS/error_analysis/results_all/ott_dev_core_reader_hop1keep200_shard0_of_1_wo_expanded_query_retrieval_error_instances.json', 'w') as file:
    json.dump(error_instances, file, indent=4)