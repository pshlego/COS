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
index_name = "top1_edge_embeddings_v2_trained_1_epoch_bsize_512.nbits2"
ids_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/edge/top_1/index_to_chunk_id_edge_topk_1.json"
collection_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/edge/top_1/edge_topk_1.tsv"
index_root_path = "/mnt/sdc/shpark/OTT-QAMountSpace/Embeddings"
checkpoint_path = "/mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/ColBERT/edge_trained_v2"
retriever = ColBERTRetriever(index_name, ids_path, collection_path, index_root_path, checkpoint_path)

# Initilaize Edge Content
EDGES_NUM = 17151500
edge_dataset_path = "/mnt/sdd/shpark/preprocess_table_graph_cos_apply_topk_edge_1.jsonl"
edge_key_to_content = read_jsonl(edge_dataset_path, key = 'chunk_id', num = EDGES_NUM)

@app.route("/edge_retrieve", methods=["GET", "POST", "OPTIONS"])
def edge_retrieve():
    # Handle GET and POST requests
    if request.method == "POST":
        params: Dict = request.json
    else:
        params: Dict = request.args.to_dict()

    query = params["query"]
    k = params.get("k", 10000)
    torch.cuda.empty_cache()
    retrieved_key_list, retrieved_score_list = retriever.search(query, k=10000)
    
    edge_content_list = []
    for key, edge_score in zip(retrieved_key_list, retrieved_score_list):
        edge_content = edge_key_to_content[key]
        if 'linked_entity_id' not in edge_content:
            continue
        edge_content['retrieval_score'] = edge_score
        edge_content_list.append(edge_content)

        if len(edge_content_list) >= k:
            break

    response = {"edge_content_list": edge_content_list}

    return response

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    serve(app, host="0.0.0.0", port=5000)