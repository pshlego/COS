import json
from pymongo import MongoClient
from tqdm import tqdm
username = "root"
password = "1234"
client = MongoClient("mongodb://localhost:27017/", username=username, password=password)
db = client["mydatabase"]  # 데이터베이스 선택
print("MongoDB Connected")

passage_collection_0 = db["ott_table"]
total_passages_0 = passage_collection_0.count_documents({})
print(f"Loading {total_passages_0} passages...")
all_tables = [doc for doc in tqdm(passage_collection_0.find(), total=total_passages_0)]
print("finish loading passages")

passage_collection = db["table_chunks_to_passages_shard_author"]
total_chunks = passage_collection.count_documents({})
print(f"Loading {total_chunks} chunks...")
table_chunks = {doc['table_chunk_id']:doc for doc in tqdm(passage_collection.find(), total=total_chunks)}
print("finish loading chunks")

passage_collection_2 = db["table_graph_author"]
total_table_graphs = passage_collection_2.count_documents({})
print(f"Loading {total_table_graphs} graphs...")
table_graphs = [doc for doc in tqdm(passage_collection_2.find(), total=total_table_graphs)]
print("finish loading graphs")


page_id_to_name = {}
for i, passage in enumerate(tqdm(all_tables)):
    page_id_to_name[str(i)] = passage["chunk_id"]
    
new_data = []
for i, table_graph in enumerate(tqdm(table_graphs)):
    new_graph = table_graph
    for j, linked_entity in enumerate(table_graph['linked_entities']):
        try:
            linked_entity['scores'] = table_chunks[page_id_to_name[str(i)]]['results'][j]['scores']
            new_graph['linked_entities'][j] = linked_entity
        except:
            print(i, j)
            continue
    del new_graph['_id']
    new_data.append(new_graph)

new_file_path = '/mnt/sdd/shpark/cos/models/table_graph_author_w_score.json'  # 여기에 JSON 파일의 경로를 입력하세요.
with open(new_file_path, 'w') as file:
    json.dump(new_data, file, indent=4)
db["table_graph_author_w_score"].insert_many(new_data)