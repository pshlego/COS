import json
from pymongo import MongoClient
from tqdm import tqdm
username = "root"
password = "1234"
client = MongoClient("mongodb://localhost:27017/", username=username, password=password)
db = client["mydatabase"]  # 데이터베이스 선택
print("MongoDB Connected")
db['all_table_chunks_span_prediction'].rename('all_table_chunks_span_prediction_2', dropTarget = True)

#db.rename_collection('all_table_chunks_span_prediction_new', 'all_table_chunks_span_prediction')

# passage_collection_0 = db["preprocess_table_graph_author_w_score_2_star"]
# total_passages_0 = passage_collection_0.count_documents({})
# print(f"Loading {total_passages_0} passages...")
# graph_1 = [doc["chunk_id"] for doc in tqdm(passage_collection_0.find(), total=total_passages_0)]
# print("finish loading passages")
# passage_collection_1 = db["preprocess_table_graph_author_w_score_star"]
# total_passages_1 = passage_collection_1.count_documents({})
# print(f"Loading {total_passages_1} passages...")
# graph_2 = [doc["chunk_id"] for doc in tqdm(passage_collection_1.find(), total=total_passages_1)]
# print("finish loading passages")

# for chunk in tqdm(graph_2):
#     if chunk not in graph_1:
#         print(chunk)