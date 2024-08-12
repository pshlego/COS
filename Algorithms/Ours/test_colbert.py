import json
from tqdm import tqdm
from ColBERT.colbert import Searcher
from ColBERT.colbert.infra import ColBERTConfig
edge_index_name = "top1_edge_embeddings_v2_trained_w_original_table"
nbits = 2
collection_edge_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/edge_w_table/edge_topk_1.tsv"
edge_index_root_path = "/mnt/sdc/shpark/OTT-QAMountSpace/Embeddings"
edge_checkpoint_path = "/mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/ColBERT/edge_trained_v2"
edge_dataset_path = "/mnt/sdd/shpark/preprocess_table_graph_cos_apply_topk_edge_1_w_original_table.jsonl"
edge_ids_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/edge_w_table/index_to_chunk_id_edge_topk_1.json"
edge_key_to_content = {}
table_key_to_edge_keys = {}
print("1. Processing edges...")
edge_contents = []
with open(edge_dataset_path, "r") as file:
    for line in tqdm(file):
        edge_contents.append(json.loads(line))

for id, edge_content in tqdm(enumerate(edge_contents), total = len(edge_contents)):
    edge_key_to_content[edge_content['chunk_id']] = edge_content
id_to_edge_key = json.load(open(edge_ids_path))

colbert_edge_retriever = Searcher(index=f"{edge_index_name}.nbits{nbits}", config=ColBERTConfig(), collection=collection_edge_path, index_root=edge_index_root_path, checkpoint=edge_checkpoint_path)

error_cases = json.load(open("/home/shpark/OTT_QA_Workspace/error_case/error_case_final/table_segment_error_cases.json"))

for qid, error_case in error_cases.items():
    nl_question = error_case["question"]
    queries = colbert_edge_retriever.checkpoint.query_tokenizer.tensorize([nl_question])
    # [SEP] 없을 때의 정확도가 높았고, 숫자에 민감하게 반응하는 것을 확인했음
    doc = "SJPF Segunda Liga Player of the Month [SEP] Month, Year, Nationality, Player, Team, Position"
    passages = colbert_edge_retriever.checkpoint.doc_tokenizer.tensorize([doc])
    Q = queries
    D = passages
    encoded_Q = colbert_edge_retriever.checkpoint.query(*Q)
    Q_duplicated = encoded_Q.repeat_interleave(1, dim=0).contiguous()
    encoded_D, encoded_D_mask = colbert_edge_retriever.checkpoint.doc(*D, keep_dims='return_mask')
    pred_scores = colbert_edge_retriever.checkpoint.score(Q_duplicated, encoded_D, encoded_D_mask)
    print(f"Question: {nl_question}")