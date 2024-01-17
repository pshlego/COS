from pymongo import MongoClient
import json

# MongoDB 서버에 연결
username = "root"
password = "1234"
client = MongoClient("mongodb://localhost:27017/", username=username, password=password)
db = client["mydatabase"]  # 데이터베이스 선택
print("MongoDB Connected")
# collection = db["all_table_chunks_span_prediction"]  # 컬렉션 선택
# file_path = "/mnt/sdd/shpark/cos/models/all_table_chunks_span_prediction.json"
# # JSON 파일 읽기
# with open(file_path, 'r') as file:
#     file_data = json.load(file)

# # JSON 데이터를 MongoDB에 저장
# collection.insert_many(file_data)

# collection = db["all_passage_chunks_span_prediction"]  # 컬렉션 선택
# file_path = "/mnt/sdd/shpark/cos/models/all_passage_chunks_span_prediction.json"
# # JSON 파일 읽기
# with open(file_path, 'r') as file:
#     file_data = json.load(file)

# # JSON 데이터를 MongoDB에 저장
# collection.insert_many(file_data)
# print("ott_table_chunks_original.json inserted")

collection = db["all_table_chunks_span_prediction_new"]  # 컬렉션 선택
file_path = "/mnt/sdd/shpark/cos/models/all_table_chunks_span_prediction.json"
# JSON 파일 읽기
with open(file_path, 'r') as file:
    file_data = json.load(file)

# JSON 데이터를 MongoDB에 저장
collection.insert_many(file_data)
# print("ott_wiki_passages.json inserted")

# collection = db["ott_table_view"]  # 컬렉션 선택
# file_path = "/mnt/sdd/shpark/cos/knowledge/ott_table_view.json"
# # JSON 파일 읽기
# with open(file_path, 'r') as file:
#     file_data = json.load(file)

# # JSON 데이터를 MongoDB에 저장
# # collection.insert_many(file_data['doc_list'])
# print("ott_table_view.json inserted")

# collection = db["ott_passage_view"]  # 컬렉션 선택
# file_path = "/mnt/sdd/shpark/cos/knowledge/ott_passage_view.json"
# # JSON 파일 읽기
# with open(file_path, 'r') as file:
#     file_data = json.load(file)

# # JSON 데이터를 MongoDB에 저장
# collection.insert_many(file_data['doc_list'])
# print("ott_passage_view.json inserted")