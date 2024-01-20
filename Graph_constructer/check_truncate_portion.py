import hydra
import math
from omegaconf import DictConfig
from tqdm import tqdm
import json
from subgraph_embedder import preprocess_graph
from pymongo import MongoClient
from dpr.models.hf_models_cos import get_any_tensorizer
from dpr.data.biencoder_data import (
    BiEncoderPassage,
)

@hydra.main(config_path="conf", config_name="subgraph_embedder")
def main(cfg: DictConfig):
    # set up mongodb
    client = MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password))
    mongodb = client[cfg.dbname]
    
    # get tensorizer
    cfg.encoder.sequence_length = 1000000000
    tensorizer = get_any_tensorizer(cfg)
    tensorizer.pad_to_max = False
    # preprocess graph
    node_list = preprocess_graph(cfg, mongodb)
    all_nodes_dict = {}
    for chunk in node_list:
        sample_id = chunk['chunk_id']
        all_nodes_dict[sample_id] = BiEncoderPassage(chunk['text'], chunk['title'])
    
    all_nodes = [(k, v) for k, v in all_nodes_dict.items()]
    shard_size = math.ceil(len(all_nodes) / cfg.num_shards)
    start_idx = cfg.shard_id * shard_size
    end_idx = start_idx + shard_size

    shard_nodes = all_nodes[start_idx:end_idx]

    gpu_id = cfg.gpu_id
    if gpu_id == -1:
        gpu_nodes = shard_nodes
    else:
        per_gpu_size = math.ceil(len(shard_nodes) / cfg.num_gpus)
        gpu_start = per_gpu_size * gpu_id
        gpu_end = gpu_start + per_gpu_size
        gpu_nodes = shard_nodes[gpu_start:gpu_end]
    
    length_list = []
    portion_list = []
    for gpu_node in tqdm(gpu_nodes):
        a = tensorizer.text_to_tensor(gpu_node[1].text, title=gpu_node[1].title)
        if int(a.shape[0]) > 512:
            portion_list.append(float((int(a.shape[0])-512)/int(a.shape[0])))
        else:
            portion_list.append(0)
        length_list.append(int(a.shape[0]))
    json.dump(length_list, open(f'/mnt/sdd/shpark/portion/length_list_{cfg.hierarchical_level}', 'w'), indent=4)
    json.dump(portion_list, open(f'/mnt/sdd/shpark/portion/portion_list_{cfg.hierarchical_level}', 'w'), indent=4)
    print(f"max length: {max(length_list)}")
    print(f"min length: {min(length_list)}")
    print(f"mean length: {sum(length_list) / len(length_list)}")
    print(f"median length: {sorted(length_list)[len(length_list) // 2]}")
    print(f"portion of truncated: {sum(portion_list) / len(portion_list)}")
    print(f"max portion of truncated: {max(portion_list)}")
if __name__ == "__main__":
    main()