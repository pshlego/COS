import json
import hydra
from omegaconf import DictConfig
from pymongo import MongoClient
from tqdm import tqdm

@hydra.main(config_path="conf", config_name="subgraph_embedder")
def main(cfg: DictConfig):
    with MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password)) as client:
        mongodb = client[cfg.dbname]
        collection = mongodb["ott_dev_q_to_tables_with_bm25neg"]

        output_file_path = '/mnt/sdd/shpark/colbert/data/query.tsv'


        with open(output_file_path, 'w', encoding='utf-8') as file:
            for index, data in enumerate(tqdm(collection.find(), total=collection.count_documents({}))):
                line = f"{index}\t{data['question']}\n"
                file.write(line)

if __name__ == "__main__":
    main()
