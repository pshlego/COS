from pymongo import MongoClient
import json
from tqdm import tqdm
# MongoDB 서버에 연결
username = "root"
password = "1234"
client = MongoClient("mongodb://localhost:27017/", username=username, password=password)
db = client["mydatabase"]  # 데이터베이스 선택
print("MongoDB Connected")
table_collection = db['preprocess_table_graph_author_w_score_star']
total_tables = table_collection.count_documents({})
print(f"Loading {total_tables} tables...")
count = 0
for doc in tqdm(table_collection.find(), total=total_tables):
    count+=len(doc['passage_id_list'])
print("total graphs", total_tables)
print("average nodes", 1 + count / total_tables)