import json
import hydra
from omegaconf import DictConfig
from pymongo import MongoClient
from tqdm import tqdm

@hydra.main(config_path="conf", config_name="subgraph_embedder")
def main(cfg: DictConfig):
    with MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password)) as client:
        mongodb = client[cfg.dbname]
        collection = mongodb["preprocess_table_graph_author_w_score_2_star"]

        output_file_path = '/mnt/sdd/shpark/colbert/data/star.tsv'
        index_to_chunk_id_path = '/mnt/sdd/shpark/colbert/data/index_to_chunk_id.json'

        index_to_chunk_id = {}

        with open(output_file_path, 'w', encoding='utf-8') as file, open(index_to_chunk_id_path, 'w', encoding='utf-8') as json_file:
            for index, data in enumerate(tqdm(collection.find(), total=collection.count_documents({}))):
                line = f"{index}\t{data['title']}{data['text']}\n"
                file.write(line)
                index_to_chunk_id[index] = data['chunk_id']

            json.dump(index_to_chunk_id, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
