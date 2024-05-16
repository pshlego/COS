import json
import hydra
from omegaconf import DictConfig
from pymongo import MongoClient
from tqdm import tqdm

@hydra.main(config_path="conf", config_name="subgraph_embedder")
def main(cfg: DictConfig):
    try:
        # MongoDB connection
        with MongoClient(f"mongodb://{cfg.username}:{cfg.password}@localhost:{cfg.port}/") as client:
            mongodb = client[cfg.dbname]
            collection_names = ["preprocess_table_graph_author_w_score_2_edge", "preprocess_table_graph_author_w_score_2_star"]
            
            output_file_path = '/mnt/sdd/shpark/colbert/data/both.tsv'
            index_to_chunk_id_path = '/mnt/sdd/shpark/colbert/data/index_to_chunk_id_both.json'
            index_to_chunk_id = {}

            with open(output_file_path, 'w', encoding='utf-8') as file, \
                 open(index_to_chunk_id_path, 'w', encoding='utf-8') as json_file:

                id = 0
                for collection_name in collection_names:
                    collection = mongodb[collection_name]
                    total_documents = collection.count_documents({})
                    for index, data in enumerate(tqdm(collection.find(), total=total_documents)):
                        line = f"{id}\t{data['title']}{data['text']}\n"
                        file.write(line)
                        index_to_chunk_id[id] = data['chunk_id']
                        id += 1

                json.dump(index_to_chunk_id, json_file, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
