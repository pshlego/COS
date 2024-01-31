from ColBERT.colbert.infra import Run, RunConfig, ColBERTConfig
from ColBERT.colbert import Indexer
import hydra
import json
from omegaconf import DictConfig
from pymongo import MongoClient
from tqdm import tqdm
import os

@hydra.main(config_path="conf", config_name="subgraph_embedder_colbert")
def main(cfg: DictConfig):
    with MongoClient(f"mongodb://{cfg.username}:{cfg.password}@localhost:{cfg.port}/") as client:
        mongodb = client[cfg.dbname]
        collection_names = cfg.collection_names
        collection_tsv_path = cfg.collection_root_dir_path + f"/{cfg.hierarchical_level}.tsv"
        index_to_chunk_id_path = cfg.collection_root_dir_path + f"/index_to_chunk_id_{cfg.hierarchical_level}.json"
        chunk_id_to_index_path = cfg.collection_root_dir_path + f"/chunk_id_to_index_{cfg.hierarchical_level}.json"
        index_to_chunk_id = {}
        chunk_id_to_index = {}
        
        if not os.path.exists(collection_tsv_path) or not os.path.exists(index_to_chunk_id_path):
            with open(collection_tsv_path, 'w', encoding='utf-8') as file, open(index_to_chunk_id_path, 'w', encoding='utf-8') as json_file, open(chunk_id_to_index_path, 'w', encoding='utf-8') as json_file2:
                id = 0
                for collection_name in collection_names:
                    collection = mongodb[collection_name]
                    total_documents = collection.count_documents({})
                    
                    for index, data in enumerate(tqdm(collection.find(), total=total_documents)):
                        line = f"{id}\t{data['title']}{data['text']}\n"
                        file.write(line)
                        chunk_id_to_index[data['chunk_id']] = id
                        index_to_chunk_id[id] = data['chunk_id']
                        id += 1

                json.dump(chunk_id_to_index, json_file2, ensure_ascii=False, indent=4)
                json.dump(index_to_chunk_id, json_file, ensure_ascii=False, indent=4)
            
    experiment_name = cfg.experiment_name + f"_{cfg.hierarchical_level}_512"
        
    with Run().context(RunConfig(nranks=3, experiment=experiment_name)):
        config = ColBERTConfig(
            nbits=cfg.nbits,
            root=cfg.collection_root_dir_path,
        )
        indexer = Indexer(checkpoint=cfg.colbert_checkpoint, config=config)
        indexer.index(name=f"{experiment_name}.nbits{cfg.nbits}", collection=collection_tsv_path)
        
if __name__=='__main__':
    main()