import hydra
import os
import logging
import os
import pathlib
import pickle
import hydra
from tqdm import tqdm
from pymongo import MongoClient
from omegaconf import DictConfig
from DPR.generate_dense_embeddings import gen_ctx_vectors
from dpr.data.biencoder_data import BiEncoderPassage
from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu
from dpr.data.biencoder_data import (
    BiEncoderPassage,
)
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
import logging
logging.disable(logging.WARNING)

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
    
    # Get preprocessed graph
    hierarchical_level = cfg.hierarchical_level
    
    if cfg.top_k_passages > 1:
        preprocessed_graph_path = cfg.preprocessed_graph_path.replace('.json', f'_{hierarchical_level}_{cfg.top_k_passages}.json')
    else:
        preprocessed_graph_path = cfg.preprocessed_graph_path.replace('.json', f'_{hierarchical_level}.json')
    
    preprocess_graph_collection_name = os.path.basename(preprocessed_graph_path).split('.')[0]
    preprocess_graph_collection = mongodb[preprocess_graph_collection_name]
    total_row_nodes = preprocess_graph_collection.count_documents({})
    print(f"Loading {total_row_nodes} row nodes...")
    node_list = [doc for doc in tqdm(preprocess_graph_collection.find(), total=total_row_nodes)]
    print("finish loading row nodes")
    
    all_nodes_dict = {}
    for chunk in node_list:
        sample_id = chunk['chunk_id']
        all_nodes_dict[sample_id] = BiEncoderPassage(chunk['text'], chunk['title'])
    
    all_nodes = [(k, v) for k, v in all_nodes_dict.items()]

    # Load model
    encoder, tensorizer = setup_encoder(cfg)

    data = gen_ctx_vectors(cfg, all_nodes, encoder, tensorizer, True, expert_id = cfg.expert_id)
    if cfg.top_k_passages > 1:
        embedding_path = cfg.embedding_path + f'_{cfg.hierarchical_level}_{cfg.top_k_passages}'
    else:
        embedding_path = cfg.embedding_path + f'_{cfg.hierarchical_level}'

    pathlib.Path(os.path.dirname(embedding_path)).mkdir(parents=True, exist_ok=True)
    print('Writing results to %s', embedding_path)
    
    with open(embedding_path, mode="wb") as f:
        pickle.dump(data, f)
    print('Total passages processed %d. Written to %s', len(data), embedding_path)

if __name__ == "__main__":
    main()