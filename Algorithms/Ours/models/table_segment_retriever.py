import logging
from typing import Dict, List, Union

from flask import Flask, request
from flask_cors import CORS
from waitress import serve
from colbert_retriever import ColBERTRetriever

# Initialize Flask
app = Flask(__name__)
cors = CORS(app)

# Initialize Index
index_name = "mmqa_passage_to_table_segment_embeddings_trained_w_sample_rate_0_3_query_len_96.nbits2" #"passage_to_table_segment_embeddings_trained_w_sample_rate_0_3_cos_version_query_len_96.nbits2"
ids_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/mmqa_table/index_to_chunk_id.json"
collection_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/mmqa_table/collection.tsv"
index_root_path = "/mnt/sdc/shpark/OTT-QAMountSpace/Embeddings"
checkpoint_path = "/mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/ColBERT/passage_to_table_segment_sample_rate_0_3_query_len_96"
retriever = ColBERTRetriever(index_name, ids_path, collection_path, index_root_path, checkpoint_path)

@app.route("/table_segment_retrieve", methods=["GET", "POST", "OPTIONS"])
def table_segment_retrieve():
    # Handle GET and POST requests
    if request.method == "POST":
        params: Dict = request.json
    else:
        params: Dict = request.args.to_dict()

    query = params["query"]
    k = params.get("k", 10000)

    retrieved_key_list, retrieved_score_list = retriever.search(query, k=k)
    
    response = {"retrieved_key_list": retrieved_key_list, "retrieved_score_list": retrieved_score_list}

    return response

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    serve(app, host="0.0.0.0", port=5001)