import json
from pymongo import MongoClient
from tqdm import tqdm
# MongoDB Connection Setup
username = "root"
password = "1234"
client = MongoClient(f"mongodb://{username}:{password}@localhost:27017/")
db = client["mydatabase"]
print("MongoDB Connected")

# Load Development Instances
dev_collection = db["table_chunks_to_passages_cos_table_passage"]
preprocess_graph_dict = {}
total_num = dev_collection.count_documents({})
for instance in tqdm(dev_collection.find(), total=total_num):
    table_chunk_id = instance['table_chunk_id']
    table_chunk_name = '_'.join(table_chunk_id.split('_')[:-1])
    chunk_num = table_chunk_id.split('_')[-1]
    if table_chunk_name not in preprocess_graph_dict:
        preprocess_graph_dict[table_chunk_name] = {}
    if chunk_num not in preprocess_graph_dict[table_chunk_name]:
        preprocess_graph_dict[table_chunk_name][chunk_num] = {}
    for passage in instance['results']:
        if str(passage['row']) not in preprocess_graph_dict[table_chunk_name][chunk_num]:
            preprocess_graph_dict[table_chunk_name][chunk_num][str(passage['row'])] = []
        preprocess_graph_dict[table_chunk_name][chunk_num][str(passage['row'])].append(passage['retrieved'])

final_graph_dict = {}
for table_chunk_name, instance in tqdm(preprocess_graph_dict.items()):
    previous_row_num = 0
    table_chunk_dict = {}
    for chunk_num, chunk in instance.items():
        for row_num, row in chunk.items():
            table_chunk_dict[str(int(row_num)+previous_row_num)]=row
        previous_row_num = len(list(instance.values())[0])
    final_graph_dict[table_chunk_name] = table_chunk_dict

with open('/mnt/sdc/shpark/cos_graph.json', 'w') as file:
    json.dump(final_graph_dict, file)

ott_wiki_passages = db["ott_wiki_passages"]
ott_wiki_passage_list = [ott_wiki_passage["chunk_id"] for ott_wiki_passage in ott_wiki_passages.find()]

# Load Development Instances
dev_collection_2 = db["table_chunks_to_passages_mvd_table_passage"]
preprocess_graph_dict_2 = {}
total_num_2 = dev_collection_2.count_documents({})
for instance in tqdm(dev_collection_2.find(), total=total_num_2):
    table_chunk_id = instance['table_chunk_id']
    table_chunk_name = '_'.join(table_chunk_id.split('_')[:-1])
    chunk_num = table_chunk_id.split('_')[-1]
    if table_chunk_name not in preprocess_graph_dict_2:
        preprocess_graph_dict_2[table_chunk_name] = {}
    if chunk_num not in preprocess_graph_dict_2[table_chunk_name]:
        preprocess_graph_dict_2[table_chunk_name][chunk_num] = {}
    for passage in instance['results']:
        if str(passage['row']) not in preprocess_graph_dict_2[table_chunk_name][chunk_num]:
            preprocess_graph_dict_2[table_chunk_name][chunk_num][str(passage['row'])] = []
        retrieved_list = []
        for passage_id in passage['retrieved']:
            retrieved_list.append(ott_wiki_passage_list[int(passage_id)]) 
        preprocess_graph_dict_2[table_chunk_name][chunk_num][str(passage['row'])].append(retrieved_list)

final_graph_dict_2 = {}
for table_chunk_name, instance in tqdm(preprocess_graph_dict_2.items()):
    previous_row_num = 0
    table_chunk_dict = {}
    for chunk_num, chunk in instance.items():
        for row_num, row in chunk.items():
            table_chunk_dict[str(int(row_num)+previous_row_num)]=row
        previous_row_num = len(list(instance.values())[0])
    final_graph_dict_2[table_chunk_name] = table_chunk_dict

with open('/mnt/sdc/shpark/mvd_graph.json', 'w') as file:
    json.dump(final_graph_dict_2, file)
    
# gold_graph_list = json.load(open('/mnt/sdd/shpark/graph/gold_link/gold_link_2.json'))
# gold_graph_dict = {}
# for gold_graph in tqdm(gold_graph_list):
#     row_dict = {}
#     for link in gold_graph['gold_link']:
#         if str(link['row']) not in row_dict:
#             row_dict[str(link['row'])] = []
#         row_dict[str(link['row'])].append(link['entity'])
#     gold_graph_dict[gold_graph['chunk_id']] = row_dict

# with open('/mnt/sdc/shpark/gold_graph_2.json', 'w') as file:
#     json.dump(gold_graph_dict, file)