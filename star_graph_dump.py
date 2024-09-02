import json
from tqdm import tqdm
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def dump_jsonl(data, path):
    """
    Dumps a list of dictionaries to a JSON Lines file.

    :param data: List of dictionaries to be dumped into JSONL.
    :param path: Path where the JSONL file will be saved.
    """
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')
        print(f"Data successfully written to {path}")
    except Exception as e:
        print(f"An error occurred: {e}")

username = "root"
password = "1234"
client = MongoClient("mongodb://localhost:27017/", username=username, password=password)
db = client["mydatabase"]  # 데이터베이스 선택
print("MongoDB Connected")

graph_collection = db["preprocess_table_graph_cos_apply_topk_star"]
total_graph = graph_collection.count_documents({})
print(f"Loading {total_graph} instances...")
preprocess_table_graph_cos_apply_topk_star_list = []
for graph in tqdm(graph_collection.find(), total=total_graph):
    del graph['_id']
    preprocess_table_graph_cos_apply_topk_star_list.append(graph)

dump_jsonl(preprocess_table_graph_cos_apply_topk_star_list, '/mnt/sdd/shpark/preprocess_table_graph_cos_apply_topk_star.jsonl')