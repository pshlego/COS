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
dev_collection = db["ott_dev_q_to_tables_with_bm25neg"]
dev_instances = list(dev_collection.find())
print(f"Loaded {len(dev_instances)} development instances.")

# Load Graph Data
graph_collection = db["table_chunks_to_passages_shard_author"]
raw_graphs = list(graph_collection.find())
print(f"Loaded {len(raw_graphs)} graphs.")

# Process Graphs
graphs = {}
for raw_graph in tqdm(raw_graphs, desc="Processing graphs"):
    chunk_id = '_'.join(raw_graph['table_chunk_id'].split('_')[:-1])
    graphs.setdefault(chunk_id, set()).update(
        linked_passage['retrieved'][0] for linked_passage in raw_graph['results']
    )

# Analysis Settings
hierarchical_levels = ['star', 'edge', 'both'] # ['star', 'edge']
is_colbert, search_space = True, None # True, larger
for is_colbert in [False]:
    if is_colbert:
        search_space_list = ['larger']
    else:
        search_space_list = [None]
    # Analyze Error Instances for Each Hierarchical Level
    for search_space in search_space_list:#, 'larger'
        for level in hierarchical_levels:
            if is_colbert:
                print("colbert")
            else:
                print("cos")
            print(f"Analyzing {level} level and {search_space} search space")
            prefix = "colbert_" if is_colbert else ""
            space_suffix = f"_{search_space}" if is_colbert and search_space is not None else ""
            
            if is_colbert:
                query_results_path = f"/mnt/sdc/shpark/graph/query_results_2/colbert_graph_query_results_fix_table_error_k_500_{level}{space_suffix}.json"
                base_path = f'/home/shpark/COS/error_analysis/results_colbert_top_500/{prefix}{level}{space_suffix}'
            else:
                query_results_path = f"/mnt/sdd/shpark/graph/query_results_2/cos_graph_query_results_fix_table_error_k_500_{level}{space_suffix}.json"
                base_path = f'/home/shpark/COS/error_analysis/results_cos_top_500/{prefix}{level}{space_suffix}'
            
            with open(query_results_path, 'r') as file:
                query_results = json.load(file)

            error_instances, error_instances_ids, data_error_instances, table_instances, passage_instances, table_passage_instances = [], [], [], [], [], []
            
            for idx, instance in enumerate(tqdm(dev_instances, desc=f"Analyzing {level} level")):
                if not any(ctx["has_answer"] for ctx in query_results[idx]["ctxs"][:100]):
                    error_instances.append(query_results[idx])
                    error_instances_ids.append(idx)
            print(f"Total Error for {level}: {len(error_instances)}")

            for idx in error_instances_ids:
                instance = dev_instances[idx]
                gold_passage_set = {gold_passage for positive_ctx in instance['positive_ctxs'] for gold_passage in positive_ctx['target_pasg_titles']}
                linked_passage_set = graphs[instance['positive_table']]

                if gold_passage_set.isdisjoint(linked_passage_set) and list(gold_passage_set)[0] is not None:
                    data_error_instances.append(query_results[idx])
                    error_instance = query_results[idx]
                    error_instance['id'] = idx  # Include instance ID for reference
                else:
                    if list(gold_passage_set)[0] is None:
                        table_instances.append(query_results[idx])
                    else:
                        gold_table = any('_'.join(ctx['table_name'].split('_')[:-1]) == instance['positive_table'] for ctx in query_results[idx]["ctxs"][:100])
                        if gold_table:
                            passage_instances.append(query_results[idx])
                        else:
                            table_passage_instances.append(query_results[idx])

            print(f"Pseudo Data Graph Construction Error for {level}: {len(data_error_instances)}")

            # Save Results
            for name, data in [("error_instances", error_instances), ("data_error_instances", data_error_instances),
                            ("table_error_instances", table_instances), ("passage_error_instances", passage_instances),
                            ("table_passage_error_instances", table_passage_instances)]:
                with open(f"{base_path}_{name}.json", 'w') as file:
                    json.dump(data, file, indent=4)

            print(f"{level}: Tables - {len(table_instances)}, Passages - {len(passage_instances)}, Table/Passages - {len(table_passage_instances)}")
