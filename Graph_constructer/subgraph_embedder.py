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

from DPR.generate_dense_embeddings import gen_ctx_vectors

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

def preprocess_graph(cfg, mongodb):
    hierarchical_level = cfg.hierarchical_level
    preprocess_graph_collection_name = os.path.basename(cfg.preprocessed_graph_path.replace('.json', f'_{hierarchical_level}.json')).split('.')[0]
    collection_list = mongodb.list_collection_names()
    if preprocess_graph_collection_name not in collection_list:
        graph_collection = mongodb[cfg.graph_collection_name]
        if cfg.graph_collection_name not in mongodb.list_collection_names():
            with open(cfg.graph_path, 'r') as fin:
                data = json.load(fin)
            graph_collection.insert_many(data)
            total_graphs = graph_collection.count_documents({})
            print(f"Loading {total_graphs} graphs...")
            graph = {doc['table_chunk_id']: doc for doc in tqdm(graph_collection.find(), total=total_graphs)}
        else:
            total_graphs = graph_collection.count_documents({})
            print(f"Loading {total_graphs} graphs...")
            graph = {doc['table_chunk_id']: doc for doc in tqdm(graph_collection.find(), total=total_graphs)}
        print("finish loading graph")
        
        # Pre-fetch collections to minimize database queries
        table_collection = mongodb[cfg.table_collection_name]
        total_tables = table_collection.count_documents({})
        print(f"Loading {total_tables} tables...")
        all_tables = [doc for doc in tqdm(table_collection.find(), total=total_tables)]
        print("finish loading tables")
        
        passage_collection = mongodb[cfg.passage_collection_name]
        total_passages = passage_collection.count_documents({})
        print(f"Loading {total_passages} passages...")
        all_passages = {doc['chunk_id']: doc for doc in tqdm(passage_collection.find(), total=total_passages)}
        print("finish loading passages")
        
        table_mention_collection = mongodb[cfg.table_mention_collection_name]
        total_table_mentions = table_mention_collection.count_documents({})
        print(f"Loading {total_table_mentions} table mentions...")
        node_id_to_chunk_id = {doc['node_id']:doc['chunk_id'] for doc in tqdm(table_mention_collection.find(), total=total_table_mentions)}
        print("finish loading table mentions")
        
        node_list = []
        if hierarchical_level == 'star':
            for node_id, raw_table in tqdm(enumerate(all_tables), total=len(all_tables)):
                table_chunk_id = node_id_to_chunk_id[node_id]
                table_title = raw_table['title']
                column_names, *table_rows = raw_table['text'].split('\n')
                try:
                    node = graph[table_chunk_id]
                except:
                    for row_id, table_row in enumerate(table_rows):
                        row_node_context = f"{column_names} [SEP] {table_row}"
                        node_list.append({
                        'chunk_id': f'{node_id}_{row_id}',
                        'title': table_title,
                        'text': row_node_context,
                        'table_id': node_id,
                        'passage_chunk_id_list': [],
                        'passage_score_list': [],
                        'mention_by_mention': []
                        })
                    continue
                grounding_info = node['results']
                row_dict = {}
                for row_id, row in enumerate(table_rows):
                    row_dict[row_id] = []
                for mentions in grounding_info:
                    row_dict[mentions['row']].append(mentions)
                for row_id, mentions in row_dict.items():
                    # if node_id == 576039 and row_id == 6:
                    row_passage_chunk_id_list = []
                    row_passage_score_list = []
                    row_linked_passage_context = []
                    for mention in mentions:
                        linked_passage_context = []
                        mention_linked_entity = mention['retrieved']
                        mention_linked_entity_scores = mention['scores']
                        for id, linked_passage_chunk_id in enumerate(mention_linked_entity[:cfg.top_k_passages]):
                            # exculde duplicate passages
                            if linked_passage_chunk_id not in row_passage_chunk_id_list:
                                row_passage_chunk_id_list.append(linked_passage_chunk_id)
                                row_passage_score_list.append(mention_linked_entity_scores[id])
                            else:
                                continue
                            raw_passage = all_passages[linked_passage_chunk_id]
                            linked_passage_context.append(f"{raw_passage['title']} [SEP] {raw_passage['text']}")
                        if len(linked_passage_context)!=0:
                            row_linked_passage_context.append(' [SEP] '.join(linked_passage_context))
                    if len(row_passage_chunk_id_list) == 0:
                        row_node_context = f"{column_names} [SEP] {table_rows[row_id]}"
                    else:
                        row_node_context = f"{column_names} [SEP] {table_rows[row_id]} [SEP] {' [SEP] '.join(row_linked_passage_context)}"
                    node_list.append({
                        'chunk_id': f'{node_id}_{row_id}',
                        'title': table_title,
                        'text': row_node_context,
                        'table_id': node_id,
                        'passage_chunk_id_list': row_passage_chunk_id_list,
                        'passage_score_list': row_passage_score_list,
                        'mention_by_mention': row_linked_passage_context
                    })
        elif hierarchical_level == 'edge':
            for node_id, raw_table in tqdm(enumerate(all_tables), total=len(all_tables)):
                table_chunk_id = node_id_to_chunk_id[node_id]
                table_title = raw_table['title']
                column_names, *table_rows = raw_table['text'].split('\n')
                try:
                    node = graph[table_chunk_id]
                except:
                    for row_id, table_row in enumerate(table_rows):
                        row_node_context = f"{column_names} [SEP] {table_row}"
                        node_list.append({
                        'chunk_id': f'{node_id}_{row_id}',
                        'title': table_title,
                        'text': row_node_context,
                        'table_id': node_id,
                        'passage_chunk_id_list': [],
                        'passage_score_list': [],
                        'mention_by_mention': []
                        })
                    continue
                grounding_info = node['results']
                row_dict = {}
                for row_id, row in enumerate(table_rows):
                    row_dict[row_id] = []
                for mentions in grounding_info:
                    row_dict[mentions['row']].append(mentions)
                for row_id, mentions in row_dict.items():
                    for mention in mentions:
                        linked_passage_context = []
                        mention_linked_entity = mention['retrieved']
                        original_cell = mention["original_cell"].replace(' ', '_')
                        for linked_passage_chunk_id in mention_linked_entity[:cfg.top_k_passages]:
                            raw_passage = all_passages[linked_passage_chunk_id]
                            linked_passage_context.append(f"{raw_passage['title']} [SEP] {raw_passage['text']}")
                        if len(linked_passage_context) == 0:
                            mention_node_context = f"{column_names} [SEP] {table_rows[row_id]}"
                        else:
                            mention_node_context = f"{column_names} [SEP] {table_rows[row_id]} [SEP] {' [SEP] '.join(linked_passage_context)}"
                        node_list.append({
                            'chunk_id': f'{node_id}_{row_id}__{original_cell}',
                            'title': table_title,
                            'text': mention_node_context,
                            'table_id': node_id,
                            'passage_chunk_id_list': mention_linked_entity[:cfg.top_k_passages],
                            'mention_top_k': linked_passage_context
                        })
                    if len(mentions) == 0:
                        row_node_context = f"{column_names} [SEP] {table_rows[row_id]}"
                        node_list.append({
                            'chunk_id': f'{node_id}_{row_id}',
                            'title': table_title,
                            'text': row_node_context,
                            'table_id': node_id,
                            'passage_chunk_id_list': [],
                            'mention_top_k': []
                        })

        with open(cfg.preprocessed_graph_path.replace('.json', f'_{hierarchical_level}.json'), 'w') as fout:
            json.dump(node_list, fout, indent=4)
        
        preprocess_graph_collection = mongodb[preprocess_graph_collection_name]
        preprocess_graph_collection.insert_many(node_list)

    else:
        preprocess_graph_collection = mongodb[preprocess_graph_collection_name]
        total_row_nodes = preprocess_graph_collection.count_documents({})
        print(f"Loading {total_row_nodes} row nodes...")
        node_list = [doc for doc in tqdm(preprocess_graph_collection.find(), total=total_row_nodes)]
        print("finish loading row nodes")



    return node_list

def setup_encoder(cfg):
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
    return encoder, tensorizer

@hydra.main(config_path="conf", config_name="subgraph_embedder")
def main(cfg: DictConfig):
    # Set MongoDB
    client = MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password))
    mongodb = client[cfg.dbname]
    
    # preprocess graph
    node_list = preprocess_graph(cfg, mongodb)
    
    all_nodes_dict = {}
    for chunk in node_list:
        sample_id = chunk['chunk_id']
        all_nodes_dict[sample_id] = BiEncoderPassage(chunk['text'], chunk['title'])
    
    all_nodes = [(k, v) for k, v in all_nodes_dict.items()]

    # load model
    encoder, tensorizer = setup_encoder(cfg)

    data = gen_ctx_vectors(cfg, all_nodes, encoder, tensorizer, True, expert_id = cfg.expert_id)
    
    file = cfg.out_file
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    print('Writing results to %s', file)
    
    with open(file, mode="wb") as f:
        pickle.dump(data, f)
    print('Total passages processed %d. Written to %s', len(data), file)

if __name__ == "__main__":
    main()