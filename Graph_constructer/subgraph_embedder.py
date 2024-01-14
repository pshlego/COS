import torch
import hydra
import json
from omegaconf import DictConfig
from pymongo import MongoClient
from tqdm import tqdm
import os
import logging
import math
import os
import pathlib
import pickle
from typing import List, Tuple
from tqdm import tqdm
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from dpr.data.biencoder_data import BiEncoderPassage
from dpr.models import init_biencoder_components, init_hf_cos_biencoder
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger
from dpr.data.biencoder_data import (
    BiEncoderPassage,
)
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)
import logging
logging.disable(logging.WARNING)
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def gen_ctx_vectors(
    cfg: DictConfig,
    ctx_rows: List[Tuple[object, BiEncoderPassage]],
    model: nn.Module,
    tensorizer: Tensorizer,
    insert_title: bool = True, expert_id=None
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = cfg.batch_size
    total = 0
    results = []
    for j, batch_start in tqdm(enumerate(range(0, n, bsz)), total=n // bsz):
        batch = ctx_rows[batch_start : batch_start + bsz]
        batch_token_tensors = [
            tensorizer.text_to_tensor(
                ctx[1].text, title=ctx[1].title if insert_title else None
            )
            for ctx in batch
        ]

        ctx_ids_batch = move_to_device(
            torch.stack(batch_token_tensors, dim=0), cfg.device
        )
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), cfg.device)
        ctx_attn_mask = move_to_device(
            tensorizer.get_attn_mask(ctx_ids_batch), cfg.device
        )
        with torch.no_grad():
            if expert_id is None:
                outputs = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
            else:
                outputs = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask, expert_id=expert_id)
            if cfg.mean_pool:
                out = mean_pooling(outputs[0], ctx_attn_mask)
            else:
                out = outputs[1]
        out = out.cpu()

        ctx_ids = [r[0] for r in batch]
        extra_info = []
        if len(batch[0]) > 3:
            extra_info = [r[3:] for r in batch]

        assert len(ctx_ids) == out.size(0)
        total += len(ctx_ids)

        # TODO: refactor to avoid 'if'
        if extra_info:
            results.extend(
                [
                    (ctx_ids[i], out[i].view(-1).numpy(), *extra_info[i])
                    for i in range(out.size(0))
                ]
            )
        else:
            results.extend(
                [(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))]
            )
        if total % 10 == 0:
            print("Encoded passages %d", total)
    return results

def preprocess_graph(cfg, graph, mongodb, hierarchical_level):
    preprocess_graph_collection_name = os.path.basename(cfg.output_path.replace('.json', f'_{hierarchical_level}.json')).split('.')[0]
    collection_list = mongodb.list_collection_names()
    if preprocess_graph_collection_name not in collection_list:
        # Pre-fetch collections to minimize database queries
        table_collection = mongodb[cfg.table_collection_name]
        total_tables = table_collection.count_documents({})
        print(f"Loading {total_tables} tables...")
        all_tables = [doc for doc in tqdm(table_collection.find(), total=total_tables)]
        print("finish loading tables")
        
        passage_collection = mongodb[cfg.passage_collection_name]
        total_passages = passage_collection.count_documents({})
        print(f"Loading {total_passages} passages...")
        all_passages = [doc for doc in tqdm(passage_collection.find(), total=total_passages)]
        print("finish loading passages")
        
        table_mention_collection = mongodb[cfg.table_mention_collection_name]
        total_table_mentions = table_mention_collection.count_documents({})
        print(f"Loading {total_table_mentions} table mentions...")
        table_mentions = [doc for doc in tqdm(table_mention_collection.find(), total=total_table_mentions)]
        print("finish loading table mentions")
        node_list = []
        row_mention_dict_list = []
        if hierarchical_level == 'star':
            for node in tqdm(graph, total=len(graph)):
                node_id = node['node_id']
                mention_info = table_mentions[node_id]
                raw_table = all_tables[node_id]
                table_title = raw_table['title']
                column_names, *table_rows = raw_table['text'].split('\n')
                grounding_info = mention_info['grounding']
                row_dict = {}
                row_mention_dict = {}
                for mention in grounding_info:
                    row_dict.setdefault(mention['row_id'], []).append(mention)
                    row_mention_dict.setdefault(mention['row_id'], []).append(mention['mention_id'])
                row_mention_dict_list.append(row_mention_dict)
                for row_id, mentions in row_dict.items():
                    row_passage_id_list = []
                    row_linked_passage_context = []
                    for mention in mentions:
                        linked_passage_context = []
                        mention_id = mention['mention_id']
                        try:
                            mention_linked_entity = node['linked_entities'][mention_id]['linked_entity']
                        except:
                            print(node_id, mention_id, len(node['linked_entities']))
                            continue
                        for linked_passage_id in mention_linked_entity[:cfg.top_k_passages]:
                            if linked_passage_id not in row_passage_id_list:
                                row_passage_id_list.append(linked_passage_id)
                            else:
                                continue
                            raw_passage = all_passages[linked_passage_id]
                            linked_passage_context.append(f"{raw_passage['title']} [SEP] {raw_passage['text']}")
                        row_linked_passage_context.append(' [SEP] '.join(linked_passage_context))

                    row_node_context = f"{column_names} [SEP] {table_rows[row_id]} [SEP] {' [SEP] '.join(row_linked_passage_context)}"
                    node_list.append({
                        'chunk_id': f'{node_id}_{row_id}',
                        'title': table_title,
                        'text': row_node_context,
                        'table_id': node_id,
                        'passage_id_list': row_passage_id_list,
                        'mention_by_mention': row_linked_passage_context
                    })

        elif hierarchical_level == 'edge':
            for node in tqdm(graph, total=len(graph)):
                node_id = node['node_id']
                mention_info = table_mentions[node_id]
                raw_table = all_tables[node_id]
                table_title = raw_table['title']
                column_names, *table_rows = raw_table['text'].split('\n')
                grounding_info = mention_info['grounding']
                row_dict = {}
                row_mention_dict = {}
                for mention in grounding_info:
                    row_dict.setdefault(mention['row_id'], []).append(mention)
                    row_mention_dict.setdefault(mention['row_id'], []).append(mention['mention_id'])
                row_mention_dict_list.append(row_mention_dict)
                for row_id, mentions in row_dict.items():
                    row_linked_passage_context = []
                    for mention in mentions:
                        linked_passage_context = []
                        mention_id = mention['mention_id']
                        for linked_passage_id in node['linked_entities'][mention_id]['linked_entity'][:cfg.top_k_passages]:
                            raw_passage = all_passages[linked_passage_id]
                            linked_passage_context.append(f"{raw_passage['title']} [SEP] {raw_passage['text']}")
                        row_linked_passage_context.append(' [SEP] '.join(linked_passage_context))

                        mention_node_context = f"{column_names} [SEP] {table_rows[row_id]} [SEP] {' [SEP] '.join(linked_passage_context)}"
                        node_list.append({
                            'chunk_id': f'{node_id}_{row_id}_{mention_id}',
                            'title': table_title,
                            'text': mention_node_context,
                            'table_id': node_id,
                            'passage_id_list': node['linked_entities'][mention_id]['linked_entity'][:cfg.top_k_passages],
                            'mention_top_k': linked_passage_context
                        })

        with open(cfg.output_path.replace('.json', f'_{hierarchical_level}.json'), 'w') as fout:
            json.dump(node_list, fout, indent=4)
        
        preprocess_graph_collection = mongodb[preprocess_graph_collection_name]
        preprocess_graph_collection.insert_many(node_list)
        
        with open(cfg.output_path.replace('.json', f'_{hierarchical_level}_mention.json'), 'w') as fout:
            json.dump(row_mention_dict_list, fout, indent=4)

    else:
        preprocess_graph_collection = mongodb[preprocess_graph_collection_name]
        total_row_nodes = preprocess_graph_collection.count_documents({})
        print(f"Loading {total_row_nodes} row nodes...")
        node_list = [doc for doc in tqdm(preprocess_graph_collection.find(), total=total_row_nodes)]
        print("finish loading row nodes")
        
        with open(cfg.output_path.replace('.json', f'_{hierarchical_level}_mention.json'), 'r') as fin:
            row_mention_dict_list = json.load(fin)

    return node_list, row_mention_dict_list

@hydra.main(config_path="conf", config_name="subgraph_embedder")
def main(cfg: DictConfig):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    
    # Set MongoDB
    client = MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password))
    mongodb = client[cfg.dbname]
    
    # load graph
    graph_collection = mongodb[cfg.graph_collection_name]
    if cfg.graph_collection_name not in mongodb.list_collection_names():
        with open(cfg.graph_path, 'r') as fin:
            graph = json.load(fin)
        graph_collection.insert_many(graph)
    else:
        total_graphs = graph_collection.count_documents({})
        print(f"Loading {total_graphs} graphs...")
        graph = [doc for doc in tqdm(graph_collection.find(), total=total_graphs)]
    print("finish loading graph")
    
    # preprocess graph
    node_list, row_mention_dict_list = preprocess_graph(cfg, graph, mongodb, cfg.hierarchical_level)

    # load model
    
    cfg = setup_cfg_gpu(cfg)
    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)
    cfg.encoder.encoder_model_type = 'hf_cos'
    sequence_length = 512
    cfg.encoder.sequence_length = sequence_length
    if cfg.encoder.pretrained_file is not None:
        print ('setting pretrained file to None', cfg.encoder.pretrained_file)
        cfg.encoder.pretrained_file = None
    if cfg.model_file:
        print ('Since we are loading from a model file, setting pretrained model cfg to bert', cfg.encoder.pretrained_model_cfg)
        cfg.encoder.pretrained_model_cfg = 'bert-base-uncased'
    
    tensorizer, encoder, _ = init_biencoder_components(
        cfg.encoder.encoder_model_type, cfg, inference_only=True
    )
    print (cfg.encoder.encoder_model_type)
    encoder = encoder.ctx_model if cfg.encoder_type == "ctx" else encoder.question_model
    
    encoder, _ = setup_for_distributed_mode(
        encoder,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16,
        cfg.fp16_opt_level,
    )
    encoder.eval()
    
    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    prefix_len = len("ctx_model.")
    ctx_state = {
        key[prefix_len:]: value
        for (key, value) in saved_state.model_dict.items()
        if key.startswith("ctx_model.")
    }
    if 'encoder.embeddings.position_ids' in ctx_state:
        if 'encoder.embeddings.position_ids' not in model_to_load.state_dict():
            print ('deleting position ids', ctx_state['encoder.embeddings.position_ids'].shape)
            del ctx_state['encoder.embeddings.position_ids']
    else:
        if 'encoder.embeddings.position_ids' in model_to_load.state_dict():
            ctx_state['encoder.embeddings.position_ids'] = model_to_load.state_dict()['encoder.embeddings.position_ids']
    model_to_load.load_state_dict(ctx_state)
    
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

    expert_id = None
    if cfg.encoder.use_moe:
        # TODO(chenghao): Fix this.
        if cfg.target_expert != -1:
            expert_id = int(cfg.target_expert)
        else:
            expert_id = 1
    
    data = gen_ctx_vectors(cfg, gpu_nodes, encoder, tensorizer, True, expert_id=expert_id)
    if gpu_id == -1:
        file = cfg.out_file + "_" + str(cfg.shard_id)
    else:
        file = cfg.out_file + "_shard" + str(cfg.shard_id) + "_gpu" + str(gpu_id)
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    print('Writing results to %s', file)
    with open(file, mode="wb") as f:
        pickle.dump(data, f)
    print('Total passages processed %d. Written to %s', len(data), file)

if __name__ == "__main__":
    main()