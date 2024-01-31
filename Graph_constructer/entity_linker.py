import logging
import torch
import json
import os
import pickle
import faiss
import numpy as np
from typing import List
from dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
from dpr.utils.data_utils import Tensorizer
from dpr.models import init_biencoder_components
from tqdm import tqdm
from utils.utils import check_across_row, locate_row, get_row_indices
from DPR.run_chain_of_skills_ott import generate_entity_vectors
from transformers import BertTokenizer
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from utils.utils import process_mention, MentionInfo

logger = logging.getLogger()
setup_logger(logger)

class COSEntityLinker:
    def __init__(self, cfg, mongodb):
        self.cfg = cfg
        self.mongodb = mongodb
    
    def link(self, source_type, dest_type):
        encoder, tensorizer, gpu_index_flat, doc_ids = self.set_up_encoder(sequence_length = 512, dest_type = dest_type)
        expert_id = self.cfg.expert_id
        mention_path = self.cfg[source_type]['mention_path']
        
        # get questions & answers
        for shard_id in range(int(self.cfg.num_shards)):
            table_chunks, table_chunk_ids, cells, indices, rows = self.prepare_mentions(mention_path, self.cfg.num_shards, shard_id)
            questions_tensor, cells, rows = generate_entity_vectors(encoder, tensorizer, table_chunks, cells, indices, rows, self.cfg.batch_size, expert_id=expert_id, use_cls = False)
            
            k = self.cfg.k                  
            b_size = self.cfg.batch_size
            all_retrieved = []
            
            for i in tqdm(range(0, len(questions_tensor), b_size)):
                D, I = gpu_index_flat.search(questions_tensor[i:i+b_size].numpy(), k)
                
                for j, ind in enumerate(I):
                    retrieved_titles = [doc_ids[idx].replace('ott-wiki:_', '').split('_')[0].strip() for idx in ind]
                    retrieved_scores = D[j].tolist()
                    all_retrieved.append((retrieved_titles, retrieved_scores))
            
            print ('all retrieved', len(all_retrieved))
            curr = 0
            results = []
            
            for i, sample in enumerate(cells):
                sample_res = {'table_chunk_id': table_chunk_ids[i], 'question': table_chunks[i], 'results': []}
                retrieved = all_retrieved[curr:curr+len(sample)]
                curr += len(sample)
                for j, cell in enumerate(sample):
                    sample_res['results'].append({'original_cell': cell, 'retrieved': retrieved[j][0], 'scores': retrieved[j][1], 'row': rows[i][j]})
                results.append(sample_res)
            
            result_path = self.cfg.result_path
            
            json.dump(results, open(result_path, 'w'), indent=4)
            collection_name = os.path.basename(result_path).split('.')[0]
            collection = self.mongodb[collection_name]
            collection.insert_many(results)
            
    def set_up_encoder(self, sequence_length = None, dest_type = 'passage'):
        cfg = setup_cfg_gpu(self.cfg)

        saved_state = load_states_from_checkpoint(cfg.model_file)
        if saved_state.encoder_params['pretrained_file'] is not None:
            print ('the pretrained file is not None and set to None', saved_state.encoder_params['pretrained_file'])
            saved_state.encoder_params['pretrained_file'] = None
        set_cfg_params_from_state(saved_state.encoder_params, cfg)
        print ('because of loading model file, setting pretrained model cfg to bert', cfg.encoder.pretrained_model_cfg)
        cfg.encoder.pretrained_model_cfg = 'bert-base-uncased'
        cfg.encoder.encoder_model_type = 'hf_cos'
        if sequence_length is not None:
            cfg.encoder.sequence_length = sequence_length
        tensorizer, encoder, _ = init_biencoder_components(
            cfg.encoder.encoder_model_type, cfg, inference_only=True
        )

        encoder_path = cfg.encoder_path
        logger.info("Selecting standard question encoder")
        encoder = encoder.question_model

        encoder, _ = setup_for_distributed_mode(
            encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16
        )
        encoder.eval()

        # load weights from the model file
        model_to_load = get_model_obj(encoder)
        logger.info("Loading saved model state ...")

        encoder_prefix = (encoder_path if encoder_path else "question_model") + "."
        prefix_len = len(encoder_prefix)

        logger.info("Encoder state prefix %s", encoder_prefix)
        question_encoder_state = {
            key[prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith(encoder_prefix)
        }

        if 'encoder.embeddings.position_ids' in question_encoder_state:
            if 'encoder.embeddings.position_ids' not in model_to_load.state_dict():
                del question_encoder_state['encoder.embeddings.position_ids']        
        else:
            if 'encoder.embeddings.position_ids' in model_to_load.state_dict():
                question_encoder_state['encoder.embeddings.position_ids'] = model_to_load.state_dict()['encoder.embeddings.position_ids']     
        
        model_to_load.load_state_dict(question_encoder_state, strict=True)
        vector_size = model_to_load.get_out_size()
        logger.info("Encoder vector_size=%d", vector_size)
        
        all_context_vecs = []
        context_types = ['table', 'passage'] if dest_type == 'both' else [dest_type]
        
        for context_type in context_types:
            with open(cfg[context_type].embedding_path, 'rb') as file:
                embeddings = pickle.load(file)
                all_context_vecs.extend(embeddings)
                
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
    
    def prepare_mentions(self, filename, num_shards, shard_id):
        collection_list = self.mongodb.list_collection_names()
        collection_name = os.path.basename(filename).split('.')[0]
        
        if collection_name in collection_list:
            collection = self.mongodb[collection_name]
            total_num = collection.count_documents({})
            data = [doc for doc in tqdm(collection.find(), total = total_num)]
        else:
            data = json.load(open(filename, 'r'))
        
        shard_size = int(len(data)/int(num_shards))
        
        print ('shard size', shard_size)
        
        if shard_id != int(num_shards)-1:
            start = shard_id*shard_size
            end = (shard_id+1)*shard_size
        else:
            start = shard_id*shard_size
            end = len(data)
            
        data = data[start:end]
    
        print ('working on examples', start, end)
        
        table_chunks = []
        table_chunk_ids = []
        cells = []
        indices = []
        rows = []
        total_skipped = 0
        
        for chunk in data:
            if len(chunk['grounding']) == 0:
                total_skipped += 1
                continue
            table_chunks.append(chunk['title'] + ' [SEP] ' + chunk['text'])
            table_chunk_ids.append(chunk['chunk_id'])
            cells.append([pos['mention'] for pos in chunk['grounding']])
            indices.append([(pos['full_word_start'], pos['full_word_end']) for pos in chunk['grounding']])
            rows.append([pos['row_id'] for pos in chunk['grounding']])
        print ('total skipped', total_skipped)
        
        return table_chunks, table_chunk_ids, cells, indices, rows
    
class MVDEntityLinker:
    def __init__(self, cfg, index, view2entity, embedder, mongodb):
        # TODO: Change the cfg according to the below code.
        self.cfg = cfg
        self.index = index
        self.view2entity = view2entity
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.mention_embedder = embedder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mongodb = mongodb
        
    def link(self, source_type, detected_mentions):
        mention_queries = self.prepate_mention_queries(source_type, detected_mentions)
        
        results = self.entity_linking(mention_queries)
        
        result_path = self.cfg.result_path
        try:
            collection_name = os.path.basename(result_path).split('.')[0]
            collection = self.mongodb[collection_name]
            collection.insert_many(results)
            
            json.dump(results, open(result_path, 'w'), indent=4)
        except:
            json.dump(results, open(result_path, 'w'), indent=4)

    def prepate_mention_queries(self, source_type, detected_mentions):
        if source_type == 'table':
            mention_queries_path = self.cfg.table_mention_queries_path
            os.makedirs(os.path.dirname(mention_queries_path), exist_ok=True)
            if not os.path.exists(mention_queries_path):
                mention_queries = self.prepare_queries(source_type, detected_mentions)
                with open(mention_queries_path, 'wb') as handle:
                    pickle.dump(mention_queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(mention_queries_path, "rb") as file:
                    mention_queries = pickle.load(file)
        else:
            mention_queries_path_1 = self.cfg.passage_mention_queries_path.replace('.pkl','_1.pkl')
            mention_queries_path_2 = self.cfg.passage_mention_queries_path.replace('.pkl','_2.pkl')
            os.makedirs(os.path.dirname(mention_queries_path_1), exist_ok=True)
            if not os.path.exists(mention_queries_path_1):
                mention_queries = self.prepare_queries(source_type, detected_mentions)
                half_index = int(len(mention_queries)/2)
                with open(mention_queries_path_1, 'wb') as handle:
                    pickle.dump(mention_queries[:half_index], handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(mention_queries_path_2, 'wb') as handle:
                    pickle.dump(mention_queries[half_index:], handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(mention_queries_path_1, "rb") as file:
                    mention_queries_1 = pickle.load(file)
                with open(mention_queries_path_2, "rb") as file:
                    mention_queries_2 = pickle.load(file)
                mention_queries = mention_queries_1 + mention_queries_2
        return mention_queries
    
    def entity_linking(self, mention_queries):
        batch_size = self.cfg.batch_size
        entity_linking_result = []
        node_map = {}
        for batch_start in tqdm(range(0, len(mention_queries), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(mention_queries))
            mention_queries_batch = mention_queries[batch_start:batch_end]

            mention_ids_batch = torch.tensor([mq.mention_ids for mq in mention_queries_batch], dtype=torch.long).to(self.device)

            with torch.no_grad():
                mention_embed_batch = self.mention_embedder(mention_ids=mention_ids_batch)
                mention_embed_batch = mention_embed_batch.detach().cpu().numpy().astype('float32')
                top_k = self.cfg.top_k * 30
                scores_batch, closest_entities_batch = self.index.search(mention_embed_batch, top_k)

                for i, mention_query in enumerate(mention_queries_batch):
                    cand_entities, score_list = self.get_distinct_entities([closest_entities_batch[i]], [scores_batch[i]], 'topk')
                    chunk_id = mention_query.chunk_id
                    if chunk_id in node_map:
                        scores = [float(x) for x in score_list[0]]
                        retrieved = cand_entities[0]
                        node_map[chunk_id]['results'].append({'original_cell':mention_query.original_cell,'retrieved':retrieved, 'scores':scores, 'row':mention_query.row_id})
                    else:
                        scores = [float(x) for x in score_list[0]]
                        retrieved = cand_entities[0]
                        node_map[chunk_id] = {'question': mention_query.question, 'results': [{'original_cell':mention_query.original_cell,'retrieved':retrieved, 'scores':scores, 'row':mention_query.row_id}]}

        entity_linking_result = [{'table_chunk_id': chunk_id, 'question': links['question'], 'results': links['results']} for chunk_id, links in tqdm(node_map.items(), desc="Processing")]
        return entity_linking_result


    def get_distinct_entities(self, closest_entities, scores, type):
        mention_num = len(closest_entities)
        pred_entity_idxs = list()
        score_list = list()
        for i in range(mention_num):
            pred_entity_idx = [eidx for eidx in closest_entities[i]]
            if self.view2entity is not None:
                pred_entity_idx = [self.view2entity[str(eidx)] for eidx in closest_entities[i]]
            new_pred_entity_idx = list()
            new_scores = list()
            for j, item in enumerate(pred_entity_idx):
                if type == 'topk':
                    if item not in new_pred_entity_idx and len(new_pred_entity_idx) < self.cfg.top_k:
                        new_pred_entity_idx.append(item)
                        new_scores.append(scores[i][j])
                else:
                    if item not in new_pred_entity_idx:
                        new_pred_entity_idx.append(item)
                        new_scores.append(scores[i][j])
            pred_entity_idxs.append(new_pred_entity_idx)
            score_list.append(new_scores)
        return pred_entity_idxs, score_list

    def prepare_queries(self, source_type, detected_mentions):
        mention_queries = []
        data_mention = detected_mentions
        
        for datum_mention in tqdm(data_mention, desc="Preparing queries"):
            mentions = datum_mention['grounding']
            
            for id, mention_dict in enumerate(mentions):
                mention_ids, mention_tokens = process_mention(self.tokenizer, mention_dict, self.cfg.max_seq_length)
                mention_queries.append(MentionInfo(
                                chunk_id=datum_mention['chunk_id'],
                                original_cell=mention_dict['mention'],
                                question=datum_mention['title'] + ' [SEP] ' + datum_mention['text'],
                                node_id=mention_dict['node_id'],
                                row_id= mention_dict['row_id'] if source_type == 'table' else None,
                                mention_ids=mention_ids,
                                mention_id=mention_dict['mention_id'],
                                mention_tokens = mention_tokens,
                                data_type = source_type)
                                )

        return mention_queries