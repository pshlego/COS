import json
import glob
import hydra
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from dpr.models import init_biencoder_components
from dpr.options import setup_cfg_gpu, set_cfg_params_from_state
from Algorithms.ChainOfSkills.DPR.run_chain_of_skills_hotpot import generate_question_vectors
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)

class TableRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        
        encoder, tensorizer, gpu_index_flat, doc_ids = self.set_up_encoder(sequence_length=512)
        
        self.encoder = encoder
        self.tensorizer = tensorizer
        self.gpu_index_flat = gpu_index_flat
        self.doc_ids = doc_ids
        
    def retrieve(self, query, top_k=10):
        expert_id = None
        if self.cfg.encoder.use_moe:
            expert_id = 0

        questions_tensor = generate_question_vectors(self.encoder, self.tensorizer,
            [query], self.cfg.batch_size, expert_id=expert_id, mean_pool=self.cfg.mean_pool, silence=True
        )
        
        retrieved_results = []
        k = top_k
        D, I = self.gpu_index_flat.search(questions_tensor.cpu().numpy(), k)
        for j, ind in enumerate(I):
            retrieved_tables = [self.doc_ids[idx].replace('ott-original:_', '').strip() if 'ott-original:' in self.doc_ids[idx] else self.doc_ids[idx].replace('ott-wiki:_', '').strip() for idx in ind]
            retrieved_scores = D[j].tolist()
            for table, score in zip(retrieved_tables, retrieved_scores):
                retrieved_results.append({'title': table, 'score': score})
        
        return retrieved_results

    def set_up_encoder(self, sequence_length = 512):
        self.cfg = setup_cfg_gpu(self.cfg)

        saved_state = load_states_from_checkpoint(self.cfg.model_file)
        if saved_state.encoder_params['pretrained_file'] is not None:
            print ('the pretrained file is not None and set to None', saved_state.encoder_params['pretrained_file'])
            saved_state.encoder_params['pretrained_file'] = None
        set_cfg_params_from_state(saved_state.encoder_params, self.cfg)
        print ('because of loading model file, setting pretrained model cfg to bert', self.cfg.encoder.pretrained_model_cfg)
        self.cfg.encoder.pretrained_model_cfg = 'bert-base-uncased'
        self.cfg.encoder.encoder_model_type = 'hf_cos'
        if sequence_length is not None:
            self.cfg.encoder.sequence_length = sequence_length
        tensorizer, encoder, _ = init_biencoder_components(
            self.cfg.encoder.encoder_model_type, self.cfg, inference_only=True
        )

        encoder_path = self.cfg.encoder_path
        if encoder_path:
            encoder = getattr(encoder, encoder_path)
        else:
            encoder = encoder.question_model

        encoder, _ = setup_for_distributed_mode(
            encoder, None, self.cfg.device, self.cfg.n_gpu, self.cfg.local_rank, self.cfg.fp16
        )
        encoder.eval()

        # load weights from the model file
        model_to_load = get_model_obj(encoder)

        encoder_prefix = (encoder_path if encoder_path else "question_model") + "."
        prefix_len = len(encoder_prefix)

        question_encoder_state = {
            key[prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith(encoder_prefix)
        }
        # TODO: long term HF state compatibility fix
        if 'encoder.embeddings.position_ids' in question_encoder_state:
            if 'encoder.embeddings.position_ids' not in model_to_load.state_dict():
                del question_encoder_state['encoder.embeddings.position_ids']        
        else:
            if 'encoder.embeddings.position_ids' in model_to_load.state_dict():
                question_encoder_state['encoder.embeddings.position_ids'] = model_to_load.state_dict()['encoder.embeddings.position_ids']     
        model_to_load.load_state_dict(question_encoder_state, strict=True)
        vector_size = model_to_load.get_out_size()

        # get questions & answers
        ctx_files_patterns = self.cfg.encoded_ctx_files
        all_context_vecs = []
        for i, pattern in enumerate(ctx_files_patterns):
            print (pattern)
            pattern_files = glob.glob(pattern)
            for f in pattern_files:
                print (f)
                all_context_vecs.extend(pickle.load(open(f, 'rb')))
        all_context_embeds = np.array([line[1] for line in all_context_vecs]).astype('float32')
        doc_ids = [line[0] for line in all_context_vecs]
        ngpus = faiss.get_num_gpus()
        print("number of GPUs:", ngpus)
        index_flat = faiss.IndexFlatIP(all_context_embeds.shape[1]) 
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        gpu_index_flat = faiss.index_cpu_to_all_gpus(index_flat, co, ngpu=ngpus)
        gpu_index_flat.add(all_context_embeds)
        
        return encoder, tensorizer, gpu_index_flat, doc_ids

@hydra.main(config_path="conf", config_name="table_retriever")
def main(cfg: DictConfig):
    table_retriever = TableRetriever(cfg)
    query = 'What is the capital of France?'
    table_retriever.retrieve(query)

if __name__ == '__main__':
    main()