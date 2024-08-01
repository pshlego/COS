import json
from tqdm import tqdm
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

EDGES_NUM = 17151500
edge_contents = []
edge_dataset_path = "/mnt/sdd/shpark/preprocess_table_graph_cos_apply_topk_edge_1.jsonl"
table_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
with open(edge_dataset_path, "r") as file:
    for line in tqdm(file, total = EDGES_NUM):
        edge_contents.append(json.loads(line))

table_id_to_edge_keys = {}

print("1. Processing edges...")
for id, edge_content in tqdm(enumerate(edge_contents), total = len(edge_contents)):
    graph_chunk_id = edge_content['chunk_id']
    row_id = graph_chunk_id.split('_')[1]
    table_id = edge_content['table_id']
    table_segment_node_id = f"{table_id}_{row_id}"
    
    if str(table_id) not in table_id_to_edge_keys:
        table_id_to_edge_keys[str(table_id)] = set()
    
    table_id_to_edge_keys[str(table_id)].add(table_segment_node_id)

table_contents = json.load(open(table_data_path))
print("2. Add table information to edges...")
for table_id, table_content in tqdm(enumerate(table_contents), total = len(table_contents)):
    linked_node_id_list = list(table_id_to_edge_keys[str(table_id)])
    # column_name = table_content['text'].split('\n')[0]
    table_info = {
        "chunk_id":f"{table_id}_schema",
        "title":table_content['title'],
        "text":table_content['text'].replace('\n', ' ').replace('\r', ' '),
        "table_id": table_id,
        "table_segment_node_list":linked_node_id_list,
        "topk": 0
    }
    edge_contents.append(table_info)

print("3. Save the result...")
dump_jsonl(edge_contents, "/mnt/sdd/shpark/preprocess_table_graph_cos_apply_topk_edge_1_w_original_table.jsonl")

