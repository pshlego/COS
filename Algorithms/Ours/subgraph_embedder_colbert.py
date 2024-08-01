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
    collection_path = "/mnt/sdd/shpark/preprocess_table_graph_cos_apply_topk_edge_1_w_original_table.jsonl"
    COLLECTION_NUM = 17992395
    collection_list = []
    with open(collection_path, "r") as file:
        for line in tqdm(file, total = COLLECTION_NUM):
            collection_list.append(json.loads(line))
    
    selection_method = "topk"
    parameter_name = "1"
    collection_tsv_path = cfg.collection_root_dir_path + f"/{cfg.hierarchical_level}_{selection_method}_{parameter_name}.tsv"
    index_to_chunk_id_path = cfg.collection_root_dir_path + f"/index_to_chunk_id_{cfg.hierarchical_level}_{selection_method}_{parameter_name}.json"
    chunk_id_to_index_path = cfg.collection_root_dir_path + f"/chunk_id_to_index_{cfg.hierarchical_level}_{selection_method}_{parameter_name}.json"
    index_to_chunk_id = {}
    chunk_id_to_index = {}
    # pid_deleted_list = json.load(open(f"/mnt/sdd/shpark/colbert/data/deleted_pids/{selection_method}_{parameter_name}.json", 'r'))
    if not os.path.exists(collection_tsv_path) or not os.path.exists(index_to_chunk_id_path):
        lines_to_write = []
        chunk_id_to_index = {}
        index_to_chunk_id = {}
        # real_id = 0
        # Convert pid_deleted_list to set for faster lookup
        # pid_deleted_set = set()
        # global_id = 0
        # for collection_name in collection_names:
        #     collection = mongodb[collection_name]
        #     total_documents = collection.count_documents({})
            
        for index, data in enumerate(tqdm(collection_list, total=len(collection_list))):
            # if global_id not in pid_deleted_set:
            line = f"{index}\t{data['title']}{data['text']}\n"
            lines_to_write.append(line)
            chunk_id_to_index[data['chunk_id']] = index
            index_to_chunk_id[index] = data['chunk_id']
            # real_id += 1
            # global_id += 1

        # Write to TSV in bulk
        with open(collection_tsv_path, 'w', encoding='utf-8') as file:
            file.writelines(lines_to_write)

        # Write JSON data
        with open(index_to_chunk_id_path, 'w', encoding='utf-8') as json_file:
            json.dump(index_to_chunk_id, json_file, ensure_ascii=False, indent=4)
        with open(chunk_id_to_index_path, 'w', encoding='utf-8') as json_file2:
            json.dump(chunk_id_to_index, json_file2, ensure_ascii=False, indent=4)
            
    experiment_name = cfg.experiment_name + f"_{cfg.hierarchical_level}_512_{selection_method}_{parameter_name}_v2_trained_batchsize_512_w_original_table" #f"_{cfg.hierarchical_level}_512_ground_truth_graph" #f"_{cfg.hierarchical_level}_512_{selection_method}_{parameter_name}_sampled_0_0_0_5_trained"

    with Run().context(RunConfig(nranks=4, experiment=experiment_name)):
        config = ColBERTConfig(
            nbits=cfg.nbits,
            root=cfg.collection_root_dir_path,
        )
        indexer = Indexer(checkpoint=cfg.colbert_checkpoint, config=config)
        indexer.index(name=f"{experiment_name}.nbits{cfg.nbits}", collection=collection_tsv_path)
        
if __name__=='__main__':
    main()