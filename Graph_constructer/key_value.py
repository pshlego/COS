from pymongo import MongoClient
import json

# MongoDB 서버에 연결
username = "root"
password = "1234"
client = MongoClient("mongodb://localhost:27017/", username=username, password=password)
db = client["mydatabase"]  # 데이터베이스 선택
collection = db["all_passage_chunks_span_prediction"]  # 컬렉션 선택
file_path = "/mnt/sdd/shpark/cos/models/all_passage_chunks_span_prediction.json"
# JSON 파일 읽기
with open(file_path, 'r') as file:
    file_data = json.load(file)

# JSON 데이터를 MongoDB에 저장
collection.insert_many(file_data)