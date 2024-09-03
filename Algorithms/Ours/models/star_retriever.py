import logging
from typing import Dict, List, Union
import torch
from flask import Flask, request
from flask_cors import CORS
from waitress import serve
from utils.utils import read_jsonl
from colbert_retriever import ColBERTRetriever

# Initialize Flask
app = Flask(__name__)
cors = CORS(app)

# Initialize Index
index_name = "top1_star_embeddings_v2_trained_1_epoch_bsize_512.nbits2"
ids_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/star/index_to_chunk_id_star_topk_1.json"
collection_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/star/star_topk_1.tsv"
index_root_path = "/mnt/sdc/shpark/OTT-QAMountSpace/Embeddings"
checkpoint_path = "/mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/ColBERT/edge_trained_v2"
retriever = ColBERTRetriever(index_name, ids_path, collection_path, index_root_path, checkpoint_path)

# Initilaize Edge Content
EDGES_NUM = 6252155
star_dataset_path = "/mnt/sdd/shpark/preprocess_table_graph_cos_apply_topk_star.jsonl"
star_key_to_content = read_jsonl(star_dataset_path, key = 'chunk_id', num = EDGES_NUM)

@app.route("/star_retrieve", methods=["GET", "POST", "OPTIONS"])
def star_retrieve():
    # Handle GET and POST requests
    if request.method == "POST":
        params: Dict = request.json
    else:
        params: Dict = request.args.to_dict()

    query = params["query"]
    k = params.get("k", 10000)
    torch.cuda.empty_cache()
    retrieved_key_list, retrieved_score_list = retriever.search(query, k=k)
    
    edge_content_list = []
    edge_score_list = []
    for key, score in zip(retrieved_key_list, retrieved_score_list):
        content = star_key_to_content[key]
        if 'mentions_in_row_info_dict' not in content:
            continue
        table_id = key.split('_')[0]
        row_id = key.split('_')[1]
        mentions_in_row_info_dict = content['mentions_in_row_info_dict']
        
        for mention_id, mention_info in mentions_in_row_info_dict.items():
            linked_passage = mention_info['mention_linked_entity_id_list'][0]
            table_id = table_id
            linked_entity_id = linked_passage
            edge_content = {"chunk_id":key, "table_id":table_id, "linked_entity_id":linked_entity_id}
            edge_content_list.append(edge_content)
            edge_score_list.append(score)

    response = {"edge_content_list": edge_content_list, "retrieved_score_list": edge_score_list}

    return response

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    serve(app, host="0.0.0.0", port=5000)