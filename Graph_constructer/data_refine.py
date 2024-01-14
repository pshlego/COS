import json
from pymongo import MongoClient
from tqdm import tqdm
username = "root"
password = "1234"
client = MongoClient("mongodb://localhost:27017/", username=username, password=password)
db = client["mydatabase"]  # 데이터베이스 선택
print("MongoDB Connected")
# JSON 파일을 로드합니다.
# file_path = '/mnt/sdd/shpark/cos/models/table_chunks_to_passages_shard0_of_1.json'  # 여기에 JSON 파일의 경로를 입력하세요.
# with open(file_path, 'r') as file:
#     graph = json.load(file)

passage_collection = db["ott_wiki_passages"]
total_passages_0 = passage_collection.count_documents({})
print(f"Loading {total_passages_0} passages...")
all_passages = [doc for doc in tqdm(passage_collection.find(), total=total_passages_0)]
print("finish loading passages")

passage_collection = db["all_table_chunks_span_prediction"]
total_passages = passage_collection.count_documents({})
print(f"Loading {total_passages} passages...")
all_tables = [doc for doc in tqdm(passage_collection.find(), total=total_passages)]
print("finish loading passages")

passage_collection = db["table_chunks_to_passages_shard0_of_1"]
total_passages_2 = passage_collection.count_documents({})
print(f"Loading {total_passages_2} passages...")
table_chunks = [doc for doc in tqdm(passage_collection.find(), total=total_passages_2)]
print("finish loading passages")

page_name_to_id = {}
for i, passage in enumerate(tqdm(all_passages)):
    page_name_to_id[passage["chunk_id"]] = i

chunk_dict = {}
for table_chunk in table_chunks:
    linked_entities = []
    for mention_id, tcs in enumerate(table_chunk["results"]):
        chunk_info = {}
        chunk_info['mention_id'] = mention_id
        linked_entity = []
        for tc in tcs['retrieved']:
            linked_entity.append(page_name_to_id[tc])
        chunk_info['linked_entity'] = linked_entity
        linked_entities.append(chunk_info)
    chunk_dict[table_chunk["table_chunk_id"]] = linked_entities

new_data = []
for entity_id, passage in enumerate(tqdm(all_tables, total=total_passages)):
    new_datum = {}
    new_datum['node_id'] = entity_id
    if passage["chunk_id"] in chunk_dict.keys():
        new_datum['linked_entities'] = chunk_dict[passage["chunk_id"]]
    else:
        new_datum['linked_entities'] = passage["grounding"]
    new_data.append(new_datum)

new_file_path = '/mnt/sdd/shpark/cos/models/table_graph_original.json'  # 여기에 JSON 파일의 경로를 입력하세요.
with open(new_file_path, 'w') as file:
    json.dump(new_data, file, indent=4)
db["table_graph_original"].insert_many(new_data)
# new_data = []
# for i, datum in enumerate(tqdm(graph)):
#     datum['node_id'] = i
#     datum['linked_entities'] = datum['results']
#     del datum['results']
#     del datum['table_chunk_id']
#     del datum['question']
    
# new_file_path = '/mnt/sdd/shpark/cos/knowledge/ott_table_view_1.json'  # 여기에 JSON 파일의 경로를 입력하세요.
# # 변경 사항을 다시 파일에 저장합니다.
# with open(new_file_path, 'w') as file:
#     json.dump(new_data, file, indent=4)
# collection = db["ott_table_view"]
# collection.insert_many(new_data)

# file_path = '/mnt/sdd/shpark/cos/knowledge/ott_passage_view.json'  # 여기에 JSON 파일의 경로를 입력하세요.
# with open(file_path, 'r') as file:
#     data = json.load(file)
# new_data = []
# for i, datum in enumerate(tqdm(data['doc_list'])):
#     del datum['text']
#     datum['node_id'] = datum['doc_id']
#     del datum['doc_id']
#     new_data.append(datum)
# new_file_path = '/mnt/sdd/shpark/cos/knowledge/ott_passage_view_1.json'  # 여기에 JSON 파일의 경로를 입력하세요.
# # 변경 사항을 다시 파일에 저장합니다.
# with open(new_file_path, 'w') as file:
#     json.dump(new_data, file, indent=4)
# collection = db["ott_passage_view"]
# collection.insert_many(new_data)
# new_data = []
# for i, datum in enumerate(tqdm(data)):
#     del datum['title']
#     del datum['text']
#     datum['node_id'] = i
#     grounding_list = []
#     for j, mention in enumerate(datum['grounding']):
#         mention['mention_id'] = j
#         grounding_list.append(mention)
#     datum['grounding'] = grounding_list
#     new_data.append(datum)
# new_file_path = '/mnt/sdd/shpark/cos/knowledge/ott_table_view_1.json'  # 여기에 JSON 파일의 경로를 입력하세요.
# # # 변경 사항을 다시 파일에 저장합니다.
# with open(new_file_path, 'w') as file:
#     json.dump(new_data, file, indent=4)

# # JSON 파일을 로드합니다.
# file_path = '/mnt/sdd/shpark/cos/knowledge/ott_passage_view.json'  # 여기에 JSON 파일의 경로를 입력하세요.
# with open(file_path, 'r') as file:
#     data = json.load(file)
# new_data = []
# for i, datum in enumerate(tqdm(data)):
#     del datum['title']
#     del datum['text']
#     datum['node_id'] = i
#     grounding_list = []
#     for j, mention in enumerate(datum['grounding']):
#         mention['mention_id'] = j
#         grounding_list.append(mention)
#     datum['grounding'] = grounding_list
#     new_data.append(datum)
# new_file_path = '/mnt/sdd/shpark/cos/knowledge/ott_passage_view_1.json'  # 여기에 JSON 파일의 경로를 입력하세요.
# # # 변경 사항을 다시 파일에 저장합니다.
# with open(new_file_path, 'w') as file:
#     json.dump(new_data, file, indent=4)
# # JSON 파일을 로드합니다.
# file_path = '/mnt/sdd/shpark/cos/models/all_table_chunks_span_prediction.json'  # 여기에 JSON 파일의 경로를 입력하세요.
# with open(file_path, 'r') as file:
#     data = json.load(file)
# new_data = []
# for i, datum in enumerate(tqdm(data)):
#     del datum['title']
#     del datum['text']
#     datum['node_id'] = i
#     grounding_list = []
#     for j, mention in enumerate(datum['grounding']):
#         mention['mention_id'] = j
#         grounding_list.append(mention)
#     datum['grounding'] = grounding_list
#     new_data.append(datum)
# new_file_path = '/mnt/sdd/shpark/cos/models/all_table_chunks_span_prediction_1.json'  # 여기에 JSON 파일의 경로를 입력하세요.
# # # 변경 사항을 다시 파일에 저장합니다.
# with open(new_file_path, 'w') as file:
#     json.dump(new_data, file, indent=4)

# # JSON 파일을 로드합니다.
# file_path = '/mnt/sdd/shpark/cos/models/all_passage_chunks_span_prediction.json'  # 여기에 JSON 파일의 경로를 입력하세요.
# with open(file_path, 'r') as file:
#     data = json.load(file)
# new_data = []
# for i, datum in enumerate(tqdm(data)):
#     del datum['title']
#     del datum['text']
#     datum['node_id'] = i
#     grounding_list = []
#     for j, mention in enumerate(datum['grounding']):
#         mention['mention_id'] = j
#         grounding_list.append(mention)
#     datum['grounding'] = grounding_list
#     new_data.append(datum)
# new_file_path = '/mnt/sdd/shpark/cos/models/all_passage_chunks_span_prediction_1.json'  # 여기에 JSON 파일의 경로를 입력하세요.
# # # 변경 사항을 다시 파일에 저장합니다.
# with open(new_file_path, 'w') as file:
#     json.dump(new_data, file, indent=4)