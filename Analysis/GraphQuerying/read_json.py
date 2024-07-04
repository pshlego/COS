import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

path = "/home/shpark/OTT_QA_Workspace/error_case/both_error_cases_reranking_200_10_2_trained.json"

cross_encoder_two_node_graph_retriever = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')#("/mnt/sdd/shpark/cross_encoder/training_ott_qa_cross-encoder-cross-encoder-ms-marco-MiniLM-L-6-v2-2024-06-06_02-04-26")
cross_encoder_two_node_graph_retriever.eval()
cross_encoder_two_node_graph_retriever.to(device = torch.device("cuda"))
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')

with open(path, "r") as f:
    data = json.load(f)


for qid, datum in data.items():
    two_node_graph_features = tokenizer([datum['question']], [""], padding=True, truncation=True, return_tensors="pt").to(device = torch.device("cuda"))
    two_node_graph_scores = cross_encoder_two_node_graph_retriever(**two_node_graph_features).logits
    print(datum['positive_ctxs'][0]['text'])