import glob
import json
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator
from pymongo import MongoClient
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from DPR.run_chain_of_skills_hotpot import contrastive_generate_grounding
from dpr.models import init_biencoder_components, init_hf_cos_biencoder
from dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
import faiss
from tqdm import tqdm
import os
from dpr.data.qa_validation import has_answer
from dpr.utils.tokenizers import SimpleTokenizer
from collections import defaultdict, Counter
from DPR.run_chain_of_skills_hotpot import set_up_encoder, generate_question_vectors, rerank_hop1_results
logger = logging.getLogger()
setup_logger(logger)

def load_ott_passage_with_id(passage_path='/home/kaixinm/kaixinm/Git_repos/OTT-QA/data/all_passages.json'):
    all_passages = json.load(open(passage_path, 'r'))
    print ('all_passages', len(all_passages))
    new_passages = {}
    # all_passages = sorted(all_passages.items())
    # for i, (k, v) in enumerate(tqdm(all_passages)):
    #     if len(v) == 0:
    #        continue
    #     k = k.replace('/wiki/', '').replace('_', ' ')
    #     new_passages[k] = (v, i+840895)
    # all_passages = new_passages
    for i, passage in enumerate(all_passages):
        new_passages[passage['title']] = (passage['text'], i+840895)
    all_passages = new_passages
    print ('all_passages', len(all_passages))
    return all_passages

def load_links(links_path, linked_size=1):
    table_to_pasg_links = []
    for path in links_path:
        for f in glob.glob(path):
            print (f)
            table_to_pasg_links += json.load(open(f, 'r'))
    all_links = {}
    for sample in tqdm(table_to_pasg_links):
        links = []
        for ci, res in enumerate(sample['results']):
            for i in range(linked_size):
                links.append((res['retrieved'][i], res['scores'][i], ci, res['original_cell'], i, res['row']))
        unique_links = {}
        for l in links:
            if l[0] not in unique_links:
                unique_links[l[0]] = (l[1], l[2], l[3], l[4], l[5])
            else:
                if unique_links[l[0]][0] < l[1]:
                    unique_links[l[0]] = (l[1], l[2], l[3], l[4], l[5])
        all_links[sample['table_chunk_id']] = sorted(unique_links.items(), key=lambda x: x[1][0], reverse=True)
    print (len(all_links))
    return all_links

def generate_entity_vectors(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    cells: List[str],
    indices, rows,
    bsz: int,
    use_cls=True, expert_id=None
) -> T:
    n = len(questions)
    query_vectors = []
    skip = 0
    remaining_cells = []
    remaining_rows = []
    with torch.no_grad():
        for j, batch_start in enumerate(tqdm(range(0, n, bsz))):
            batch_questions = questions[batch_start : batch_start + bsz]
            batch_token_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]
            batch_cells = cells[batch_start : batch_start + bsz]

            batch_indices = indices[batch_start : batch_start + bsz]
            batch_rows = rows[batch_start : batch_start + bsz]
            q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            outputs = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask, expert_id=expert_id)
            seq_out = outputs[0]

            batch_cell_ids = [[torch.tensor(tensorizer.tokenizer.encode(cell, add_special_tokens=False),dtype=torch.long, device=q_ids_batch.device) for cell in sample_cell] for sample_cell in batch_cells]
            question_rep_pos = []
            for b, sample_cell_ids in enumerate(batch_cell_ids):
                q_tensor = q_ids_batch[b]
                this_question = []
                sample_indices = batch_indices[b]
                sample_cell_remain = []
                sample_row_remain = []
                for _, cell_ids in enumerate(sample_cell_ids):
                    rep_pos = torch.zeros(len(q_tensor), device=q_ids_batch.device)
                    if sample_indices[_][0] == 0:
                        start = 0
                        for i in range(len(q_tensor)-len(cell_ids)):
                            if torch.equal(q_tensor[i:i+len(cell_ids)],cell_ids):
                                start = i
                                break
                        if start == 0:
                            print (q_tensor)
                            print (cell_ids)
                            print ('cannot find cell')
                            print (questions[b])
                            print (batch_cells[b][_])
                            print (j, b)
                            exit(0)
                        rep_pos[start] = 1
                        rep_pos[start+len(cell_ids)-1] = 1
                    else:
                        start = sample_indices[_][0]
                        end = sample_indices[_][1]
                        rep_pos[start] = 1
                        rep_pos[end] = 1
                        if end != start+len(cell_ids)-1:
                            skip += 1
                            continue
                        assert end == start+len(cell_ids)-1
                    sample_cell_remain.append(batch_cells[b][_])
                    sample_row_remain.append(batch_rows[b][_])
                    this_question.append(rep_pos)
                if len(this_question) == 0:
                    print ('big skip')
                    print (batch_questions[b])
                    question_rep_pos.append(None)
                else:
                    question_rep_pos.append(torch.stack(this_question, dim=0))
                remaining_cells.append(sample_cell_remain)
                remaining_rows.append(sample_row_remain)
            for b in range(len(batch_cell_ids)):
                if question_rep_pos[b] is not None:
                    cell_reps = torch.sum(seq_out[b].unsqueeze(0)*question_rep_pos[b].unsqueeze(-1), dim=1)/question_rep_pos[b].sum(dim=1).unsqueeze(-1)
                    if use_cls:
                        cell_reps = (cell_reps + outputs[1][b].unsqueeze(0))/2  
                    query_vectors.append(cell_reps.cpu())  

    logger.info("Total encoded queries tensor %s", len(query_vectors))
    print ("skip", skip)
    assert len(remaining_cells) == len(questions)
    flat_query_vectors = []
    for q in query_vectors:
        flat_query_vectors.extend(q)
    return torch.stack(flat_query_vectors, dim=0), remaining_cells, remaining_rows

def get_row_indices(question, tokenizer):
    original_input = tokenizer.tokenize(question)
    rows = question.split('\n')
    indices = []
    tokens = []
    for row in rows:
        tokens.extend(tokenizer.tokenize(row))
        indices.append(len(tokens)+1)
    assert tokens == original_input
    return indices

def prepare_all_table_chunks(filename, tokenizer):
    # chunk_dict = {}
    # table_chunks = []
    # table_chunk_ids = []
    # row_start = []
    # row_indices = []
    # for chunk in tqdm(data):
    #     table_chunks.append(chunk['title'] + ' [SEP] ' + chunk['text'])
    #     table_chunk_ids.append(chunk['chunk_id'])
    #     table_row_indices = get_row_indices(chunk['title'] + ' [SEP] ' + chunk['text'], tokenizer)
    #     row_start.append(table_row_indices[0])
    #     row_indices.append(table_row_indices[1:])
    # chunk_dict['data'] = data
    # chunk_dict['table_chunks'] = table_chunks
    # chunk_dict['table_chunk_ids'] = table_chunk_ids
    # chunk_dict['row_start'] = row_start
    # chunk_dict['row_indices'] = row_indices
    # json.dump(chunk_dict, open('/mnt/sdd/shpark/knowledge/ott_table_chunks_original_with_row_indices.json', 'w'), indent=4)
    with open('/mnt/sdd/shpark/cos/knowledge/ott_table_chunks_original_with_row_indices.json', 'r') as file:
        chunk_dict = json.load(file)
    data = json.load(open(filename, 'r'))
    table_chunks = chunk_dict['table_chunks']
    table_chunk_ids = chunk_dict['table_chunk_ids']
    row_start = chunk_dict['row_start']
    row_indices = chunk_dict['row_indices']
    return data, table_chunks, table_chunk_ids, row_start, row_indices

def prepare_all_table_chunks_step2(filename, num_shards, shard_id):
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
            #print ('skipping')
            total_skipped += 1
            #print (chunk['title'] + ' [SEP] ' + chunk['text'])
            # if total_skipped == 20:
            #     exit(0)
            continue
        table_chunks.append(chunk['title'] + ' [SEP] ' + chunk['text'])
        table_chunk_ids.append(chunk['chunk_id'])
        cells.append([pos[2] for pos in chunk['grounding']])
        indices.append([(pos[0], pos[1]) for pos in chunk['grounding']])
        rows.append([pos[3] for pos in chunk['grounding']])
    print ('total skipped', total_skipped)
    return data, table_chunks, table_chunk_ids, cells, indices, rows

def build_query(filename):
    data = json.load(open(filename))
    for sample in data:
        if 'hard_negative_ctxs' in sample:
            del sample['hard_negative_ctxs']
    return data 

def q_to_tables(cfg: DictConfig, encoder, tensorizer, gpu_index_flat, doc_ids):
    expert_id = None
    if cfg.encoder.use_moe:
        logger.info("Setting expert_id=0")
        expert_id = 0
        logger.info(f"mean pool {cfg.mean_pool}")

    # '/home/kaixinm/kaixinm/Git_repos/OTT-QA/intermediate_data/dev_q_to_tables_with_bm25neg.json'
    data = build_query(cfg.qa_dataset)
    questions_tensor = generate_question_vectors(encoder, tensorizer,
        [s['question'] for s in data], cfg.batch_size, expert_id=expert_id, mean_pool=cfg.mean_pool
    )
    assert questions_tensor.shape[0] == len(data)

    k = 100                         
    b_size = 2048
    all_retrieved = []
    for i in tqdm(range(0, len(questions_tensor), b_size)):
        D, I = gpu_index_flat.search(questions_tensor[i:i+b_size].cpu().numpy(), k)  # actual search
        for j, ind in enumerate(I):
            retrieved_chunks = [doc_ids[idx].replace('ott-original:_', '').strip() if 'ott-original:' in doc_ids[idx] else doc_ids[idx].replace('ott-wiki:_', '').strip() for idx in ind]
            retrieved_scores = D[j].tolist()
            all_retrieved.append((retrieved_chunks, retrieved_scores))
    print ('all retrieved', len(all_retrieved))

    limits = [1, 2, 5, 10, 20, 50, 100]
    topk = [0]*len(limits)

    for i, sample in enumerate(data):
        if 'positive_ctxs' in sample:
            gold = [pos['chunk_id'] for pos in sample['positive_ctxs']]
            sample['results'] = [{'title': ctx, 'score': all_retrieved[i][1][_], 'gold': ctx in gold} for _, ctx in enumerate(all_retrieved[i][0])]
            for j, limit in enumerate(limits):
                retrieved = all_retrieved[i][0][:limit]
                if any([g in retrieved for g in gold]):
                    topk[j] += 1
        else:
            sample['results'] = [{'title': ctx, 'score': all_retrieved[i][1][_]} for _, ctx in enumerate(all_retrieved[i][0])]
    for i, limit in enumerate(limits):
        print ('topk', topk[i]/len(data), 'limit', limit)
    return data

def main_realistic_all_table_chunks(cfg: DictConfig):
    encoder, tensorizer, gpu_index_flat, doc_ids = set_up_encoder(cfg, sequence_length=512)
    expert_id = None
    if cfg.encoder.use_moe:
        logger.info("Setting expert_id=2")
        expert_id = 2

    # get questions & answers
    for shard_id in range(int(cfg.num_shards)):
        data, table_chunks, table_chunk_ids, cells, indices, rows = prepare_all_table_chunks_step2(cfg.qa_dataset, cfg.num_shards, shard_id)
        questions_tensor, cells, rows = generate_entity_vectors(encoder, tensorizer,
            table_chunks, cells, indices, rows, cfg.batch_size, expert_id=expert_id, use_cls=False
        )
        k = 2                        
        b_size = 512
        all_retrieved = []
        for i in tqdm(range(0, len(questions_tensor), b_size)):
            D, I = gpu_index_flat.search(questions_tensor[i:i+b_size].numpy(), k)  # actual search
            for j, ind in enumerate(I):
                # document단위로 검색하기에 document index를 저장한다.
                retrieved_titles = [doc_ids[idx].replace('ott-wiki:_', '').split('_')[0].strip() for idx in ind]
                retrieved_scores = D[j].tolist()
                all_retrieved.append((retrieved_titles, retrieved_scores))
        print ('all retrieved', len(all_retrieved))
        curr = 0
        results_data = []
        for i, sample in enumerate(cells):
            sample_res = {'table_chunk_id': table_chunk_ids[i], 'question': table_chunks[i], 'results': []}
            retrieved = all_retrieved[curr:curr+len(sample)]
            curr += len(sample)
            for j, cell in enumerate(sample):
                sample_res['results'].append({'original_cell': cell, 'retrieved': retrieved[j][0], 'scores': retrieved[j][1], 'row': rows[i][j]})
            results_data.append(sample_res)
        
        output_name = '/'.join(cfg.model_file.split('/')[:-1]) + f'/{cfg.out_file}/table_chunks_to_passages_shard{shard_id}_of_{cfg.num_shards}.json'
        if not os.path.exists('/'.join(cfg.model_file.split('/')[:-1]) + f'/{cfg.out_file}'):
            os.makedirs('/'.join(cfg.model_file.split('/')[:-1]) + f'/{cfg.out_file}')
        with open(output_name, 'w') as f:
            json.dump(results_data, f, indent=4)

def span_proposal(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)
    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)
    cfg.encoder.encoder_model_type = 'hf_cos'
    sequence_length = 512
    cfg.encoder.sequence_length = sequence_length
    cfg.encoder.pretrained_file=None
    cfg.encoder.pretrained_model_cfg = 'bert-base-uncased'

    tensorizer, encoder, _ = init_biencoder_components(
        cfg.encoder.encoder_model_type, cfg, inference_only=True
    )
    encoder, _ = setup_for_distributed_mode(
        encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16
    )
    encoder.eval()
    
    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")
    if 'question_model.encoder.embeddings.position_ids' not in saved_state.model_dict:
        if 'question_model.encoder.embeddings.position_ids' in model_to_load.state_dict():
            saved_state.model_dict['question_model.encoder.embeddings.position_ids'] = model_to_load.state_dict()['question_model.encoder.embeddings.position_ids']
            saved_state.model_dict['ctx_model.encoder.embeddings.position_ids'] = model_to_load.state_dict()['ctx_model.encoder.embeddings.position_ids']
    model_to_load.load_state_dict(saved_state.model_dict, strict=True)
    expert_id = 5
    data, table_chunks, table_chunk_ids, row_start, row_indices = prepare_all_table_chunks(cfg.qa_dataset, tensorizer.tokenizer)
    found_cells = contrastive_generate_grounding(encoder, tensorizer, table_chunks, row_start, row_indices, cfg.batch_size, expert_id=expert_id)
    #full_word_start, full_word_end, span, row_id (몇번째 row인지)
    for i in tqdm(range(len(found_cells))):
        data[i]['grounding'] = found_cells[i]
    output_name = '/'.join(cfg.model_file.split('/')[:-1]) + '/all_table_chunks_span_prediction.json'
    json.dump(data, open(output_name, 'w'), indent=4)

def process_ott_beams_new(beams, original_sample, all_table_chunks, all_passages, tokenizer, limit):
    exist = {}
    all_included = []
    for beam in beams:
        table_id = beam['chunk_id']
        if table_id not in exist:
            exist[table_id] = 1
            content = all_table_chunks[table_id]
            content_text = content['text']
            content_title = content['title']
            if 'answers' in original_sample:
                ctx_has_answer = has_answer(original_sample['answers'], content_text, tokenizer, 'string')
            else:
                ctx_has_answer = False
            all_included.append({'id':table_id, 'title': content_title, 'text': content_text, 'has_answer': ctx_has_answer, 'hop1 score': beam['hop1 score'], 'chunk_is_gold': beam['chunk_is_gold'], 'row_idx': beam['row_idx'], 'row_is_gold': beam['row_is_gold'], 'row_rank_score': beam['row_rank_score']})
            if len(all_included) == limit:
                break
        if 'pasg_title' in beam:
            if beam['pasg_title'] in exist:
                continue
            exist[beam['pasg_title']] = 1
            text = all_passages[beam['pasg_title']][0]
            #grounded_cell = beam[3]
            #table_part = beam['q'].split('[SEP]')[1].strip()
            content = all_table_chunks[table_id]
            content_text = content['text']
            content_title = content['title']
            rows = content['text'].split('\n')[1:]
            header = content['text'].split('\n')[0]
            full_text = header + '\n' + rows[beam['row_idx']] + '\n' + beam['pasg_title'] + ' ' + text
            if 'answers' in original_sample:
                pasg_has_answer = has_answer(original_sample['answers'], full_text, tokenizer, 'string')
            else:
                pasg_has_answer = False
            all_included.append({'id':table_id, 'title': content_title, 'text': full_text, 'has_answer': pasg_has_answer, 'hop1 score': beam['hop1 score'], 'chunk_is_gold': beam['chunk_is_gold'], 'row_idx': beam['row_idx'], 'row_is_gold': beam['row_is_gold'], 'row_rank_score': beam['row_rank_score'], 'grounding': beam['grounding'] if 'grounding' in beam else None, 'hop2 score': beam['hop2 score'], 'hop2 source': beam['hop2 source'], 'path score': beam['path score']})
            if len(all_included) == limit:
                break
            #else:
            #    failed += 1
    return all_included

def chain_of_skills(cfg: DictConfig):

    with open('/home/shpark/mnt_sdc/shpark/cos/cos/OTT-QA/ott_dev_q_to_tables_with_bm25neg.json', 'r') as file:
        dev_set = json.load(file)
    
    table_chunks = json.load(open(cfg.ctx_datatsets[0], 'r'))
    all_table_chunks = {}
    for chunk in table_chunks:
        all_table_chunks[chunk['chunk_id']] = chunk

    # passage embedding으로 변경
    cfg.encoded_ctx_files[0] = cfg.encoded_ctx_files[0].replace('ott_table_original', 'ott_wiki_linker')
    # reload the index
    encoder, tensorizer, gpu_index_flat, doc_ids = set_up_encoder(cfg, sequence_length=512)  
    logger.info(f"Setting expert_id={cfg.hop2_expert}")
    expert_id = cfg.hop2_expert
    expanded_query_retrieval_result_list = []
    
    for si, sample in enumerate(tqdm(dev_set)):
        expanded_query_list = []
        question = sample['question']
        retrieved_result_dict = {}
        
        for positive_ctx in sample['positive_ctxs']:
            gold_table_chunk_id = positive_ctx['chunk_id']
            table_titile = all_table_chunks[gold_table_chunk_id]['title']
            table_text = all_table_chunks[gold_table_chunk_id]['text']
            column_names = table_text.split('\n')[0]
            
            for row in table_text.split('\n')[1:]:
                expanded_query_list.append(question + ' [SEP] ' + table_titile + ' ' + column_names + '\n' + row)

            # vector를 확장한다. (row 단위로)
            q_vecs = generate_question_vectors(encoder, tensorizer, expanded_query_list, cfg.batch_size, expert_id=expert_id, silence=True)
            
            # 확장된 vector로 10개의 관련 passage를 검색한다.
            D, I = gpu_index_flat.search(q_vecs.numpy(), 10)  # actual search
            passage_list = []
            score_list = []
            for ri in range(len(I)):
                retrieved_score = D[ri].tolist()
                row_list = []
                
                for idx in I[ri]:
                    retrieved_title = doc_ids[idx].replace('ott-wiki:_', '').strip()
                    row_list.append(retrieved_title)
                
                passage_list.append(row_list)
                score_list.append(retrieved_score)
            
            retrieved_result_dict[gold_table_chunk_id] = {'passage_list': passage_list, 'score_list': score_list}

        expanded_query_retrieval_result_list.append({'id':sample['id'], 'question': question, 'retrieved_result': retrieved_result_dict})

    with open('/home/shpark/COS/error_analysis/expanded_query_retrieval_result.json', 'w') as file:
        json.dump(expanded_query_retrieval_result_list, file, indent=4)

@hydra.main(config_path="conf", config_name="dense_retriever")
def main(cfg: DictConfig):
    if cfg.do_link:
        main_realistic_all_table_chunks(cfg)
    elif cfg.do_span:
        span_proposal(cfg)
    elif cfg.do_cos:
        chain_of_skills(cfg)
    else:
        raise ValueError('Please specify the task to do')

if __name__ == '__main__':
    main()