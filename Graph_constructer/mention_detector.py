import hydra
import logging
import torch
import json
import os
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
logger = logging.getLogger()
setup_logger(logger)

class MentionDetector:
    def __init__(self, cfg, mongodb):
        self.cfg = cfg
        self.mongodb = mongodb
        
    def span_proposal(self, all_tables = None, all_passages = None):
        cfg = setup_cfg_gpu(self.cfg)
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
        expert_id = self.cfg.expert_id
        
        if all_tables is not None:
            data = all_tables
            if os.path.exists(cfg.table_chunk):
                with open(cfg.table_chunk, 'r') as file:
                    chunk_dict = json.load(file)
                table_chunks = chunk_dict['table_chunks']
                table_chunk_ids = chunk_dict['table_chunk_ids']
                row_start = chunk_dict['row_start']
                row_indices = chunk_dict['row_indices']
            else:
                chunk_dict = {}
                table_chunks, table_chunk_ids, row_start, row_indices = self.prepare_all_table_chunks(data, tensorizer.tokenizer)
                chunk_dict['table_chunks'] = table_chunks
                chunk_dict['table_chunk_ids'] = table_chunk_ids
                chunk_dict['row_start'] = row_start
                chunk_dict['row_indices'] = row_indices
                json.dump(chunk_dict, open(cfg.table_chunk, 'w'), indent=4)
            _, found_mentions = self.contrastive_generate_grounding(encoder, tensorizer, table_chunks, row_start, row_indices, cfg.batch_size, expert_id=expert_id, max_length=cfg.max_mention_context_length)
            output_name = self.cfg.table_mention_path
            span_prediction_results = []
            for i in tqdm(range(len(found_mentions))):
                span_dict = {}
                span_dict['chunk_id'] = data[i]['chunk_id']
                span_dict['node_id'] = i
                span_dict['grounding'] = found_mentions[i]
                span_prediction_results.append(span_dict)
            json.dump(span_prediction_results, open(output_name, 'w'), indent=4)
            collection_name = os.path.basename(output_name).split('.')[0]
            collection = self.mongodb[collection_name]
            collection.insert_many(span_prediction_results)
            
        if all_passages is not None:
            data = all_passages
            if os.path.exists(cfg.passage_chunk):
                with open(cfg.passage_chunk, 'r') as file:
                    chunk_dict = json.load(file)
                passage_chunks = chunk_dict['passage_chunks']
                passage_chunk_ids = chunk_dict['passage_chunk_ids']
            else:
                chunk_dict = {}
                passage_chunks, passage_chunk_ids = self.prepare_all_passage_chunks(data)
                chunk_dict['passage_chunks'] = passage_chunks
                chunk_dict['passage_chunk_ids'] = passage_chunk_ids
                json.dump(chunk_dict, open(cfg.passage_chunk, 'w'), indent=4)
            _, found_mentions = self.contrastive_generate_grounding(encoder, tensorizer, passage_chunks, None, None, cfg.batch_size, expert_id=expert_id, max_length=cfg.max_mention_context_length)
            output_name = self.cfg.passage_mention_path
            span_prediction_results = []
            for i in tqdm(range(len(found_mentions))):
                span_dict = {}
                span_dict['chunk_id'] = data[i]['chunk_id']
                span_dict['node_id'] = i
                span_dict['grounding'] = found_mentions[i]
                span_prediction_results.append(span_dict)
            json.dump(span_prediction_results, open(output_name, 'w'), indent=4)
            collection_name = os.path.basename(output_name).split('.')[0]
            collection = self.mongodb[collection_name]
            collection.insert_many(span_prediction_results)
            
    def prepare_all_table_chunks(self, data, tokenizer):
        chunk_dict = {}
        table_chunks = []
        table_chunk_ids = []
        row_start = []
        row_indices = []
        for chunk in tqdm(data):
            table_chunks.append(chunk['title'] + ' [SEP] ' + chunk['text'])
            table_chunk_ids.append(chunk['chunk_id'])
            table_row_indices = get_row_indices(chunk['title'] + ' [SEP] ' + chunk['text'], tokenizer)
            row_start.append(table_row_indices[0])
            row_indices.append(table_row_indices[1:])
        return table_chunks, table_chunk_ids, row_start, row_indices

    def prepare_all_passage_chunks(self, data):
        passage_chunks = []
        passage_chunk_ids = []
        for chunk in tqdm(data):
            passage_chunks.append(chunk['title'] + ' [SEP] ' + chunk['text'])
            passage_chunk_ids.append(chunk['chunk_id'])
        return passage_chunks, passage_chunk_ids

    def contrastive_generate_grounding(
        self,
        encoder: torch.nn.Module,
        tensorizer: Tensorizer,
        questions: List[str],
        row_start: List[int],
        row_indices: List[List[int]],
        bsz: int, sep_id=1, expert_id=5, max_length = 128
    ):
        n = len(questions)
        found_cells = []
        found_mentions = []
        breaking = 0
        accepted = 0
        rejected = 0
        thresholds = []
        with torch.no_grad():
            for batch_start in tqdm(range(0, n, bsz)):
                node_id_list = [*range(batch_start, batch_start + bsz)]
                batch_questions = questions[batch_start : batch_start + bsz]
                batch_token_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]
                if row_start is not None:
                    batch_row_start = row_start[batch_start : batch_start + bsz]
                else:
                    batch_row_start = []
                    for i, token_tensor in enumerate(batch_token_tensors):
                        if '[SEP]' in batch_questions[i]:
                            batch_row_start.append((token_tensor == tensorizer.tokenizer.sep_token_id).nonzero()[0][0]+1)
                        else:
                            batch_row_start.append(1)
                if row_indices is not None:
                    batch_row_indices = row_indices[batch_start : batch_start + bsz]
                else:
                    batch_row_indices = None

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

                start_positions = []
                end_positions = []
                num_spans = []
                for i in range(len(batch_row_start)):
                    padding_start = (batch_token_tensors[i] == tensorizer.tokenizer.sep_token_id).nonzero()[sep_id][0]+1
                    spans = [(i, j) for i in range(batch_row_start[i], padding_start) for j in range(i, min(i+10, padding_start))]#table/column name에서는 span을 prediction하지 않는다.
                    start_positions.append([s[0] for s in spans])
                    end_positions.append([s[1] for s in spans])
                    num_spans.append(len(spans))
                batch_start_positions = torch.zeros((len(start_positions), max(num_spans)), dtype=torch.long, device=q_ids_batch.device)
                batch_end_positions = torch.zeros((len(end_positions), max(num_spans)), dtype=torch.long, device=q_ids_batch.device)
                for i in range(len(start_positions)):
                    batch_start_positions[i][:len(start_positions[i])] = torch.tensor(start_positions[i])
                    batch_end_positions[i][:len(end_positions[i])] = torch.tensor(end_positions[i])

                q_outputs = encoder.question_model(q_ids_batch, q_seg_batch, q_attn_mask, expert_id=expert_id)
                start_vecs = []
                end_vecs = []
                for i in range(q_outputs[0].shape[0]):
                    start_vecs.append(torch.index_select(q_outputs[0][i], 0, batch_start_positions[i]))
                    end_vecs.append(torch.index_select(q_outputs[0][i], 0, batch_end_positions[i]))
                start_vecs = torch.stack(start_vecs, dim=0)
                end_vecs = torch.stack(end_vecs, dim=0)
                span_vecs = torch.cat([start_vecs, end_vecs], dim=-1)
                span_vecs = torch.tanh(encoder.span_proj(span_vecs))
                cells_need_link = encoder.span_query(span_vecs).squeeze(-1)
                q_pooled = encoder.span_query(q_outputs[1])
                invalid_spans = batch_start_positions == 0
                cells_need_link[invalid_spans] = -1e10
                cells_need_link = cells_need_link - q_pooled 
                sorted_score, indices = torch.sort(cells_need_link, dim=-1, descending=True)
                for i in range(len(batch_row_start)):
                    accepted_index = (sorted_score[i] < 0).nonzero()[0][0]
                    sorted_start = batch_start_positions[i][indices[i]][:accepted_index]
                    sorted_end = batch_end_positions[i][indices[i]][:accepted_index]
                    cut_off = 0
                    patient = 0
                    cells = []
                    mentions = []
                    for j in range(accepted_index):
                        start = sorted_start[j].item()
                        end = sorted_end[j].item()
                        if any([start <= cell[1] and end >= cell[0] for cell in cells]):
                            continue
                        if batch_row_indices is not None:
                            # 같은 row내에 있는지 확인
                            mid = check_across_row(start, end, batch_row_indices[i])
                        else:
                            mid = False
                        if mid:
                            breaking += 1
                            full_word_start = start
                            full_word_end = end
                            span = tensorizer.tokenizer.decode(batch_token_tensors[i][full_word_start:full_word_end+1])
                            row_id = locate_row(full_word_start, full_word_end, batch_row_indices[i])
                            if span.isnumeric():
                                continue 
                            cells.append((full_word_start, full_word_end, span, row_id))
                            full_word_start = start
                            full_word_end = end
                            span = tensorizer.tokenizer.decode(batch_token_tensors[i][full_word_start:full_word_end+1])
                            row_id = locate_row(full_word_start, full_word_end, batch_row_indices[i])
                            if span.isnumeric():
                                continue 
                            cells.append((full_word_start, full_word_end, span, row_id))
                        else:
                            full_word_start = start
                            full_word_end = end
                            span = tensorizer.tokenizer.decode(batch_token_tensors[i][full_word_start:full_word_end+1])
                            if batch_row_indices is not None:
                                row_id = locate_row(full_word_start, full_word_end, batch_row_indices[i])
                            else:
                                row_id = None
                            if span.isnumeric():
                                continue 
                            cells.append((full_word_start, full_word_end, span, row_id))
                        #get the left and right context of the mentions
                        mention_dict = {}
                        if row_start is not None:
                            mention_dict['full_word_start'] = full_word_start
                            mention_dict['full_word_end'] = full_word_end
                            
                            title_column_name = tensorizer.tokenizer.decode(batch_token_tensors[i][1:row_start[i]]).strip()
                            if mid:
                                same_row_left = tensorizer.tokenizer.decode(batch_token_tensors[i][batch_row_indices[i][row_id-2]:full_word_start]).strip()
                            else:
                                same_row_left = tensorizer.tokenizer.decode(batch_token_tensors[i][batch_row_indices[i][row_id-1]:full_word_start]).strip()
                            mention_context_left = title_column_name + ' [SEP] ' + same_row_left
                            if len(mention_context_left.split(" ")) > max_length:
                                mention_context_left = ' '.join(mention_context_left.split(" ")[max(0, -max_length):])
                            mention_dict['context_left'] = mention_context_left
                            
                            same_row_right = tensorizer.tokenizer.decode(batch_token_tensors[i][full_word_end+1:batch_row_indices[i][row_id]]).strip()
                            mention_context_right = same_row_right
                            if len(mention_context_right.split(" ")) > max_length:
                                mention_context_right = ' '.join(mention_context_right.split(" ")[:min(len(mention_context_right.split(" ")), max_length)])
                            mention_dict['context_right'] = mention_context_right.replace(' [PAD]', '')
                            
                            mention_dict['row_id'] = row_id
                            mention_dict['node_id'] = node_id_list[i]
                            mention_dict['mention'] = span
                            mention_dict['across_row'] = mid
                            mention_dict['mention_id'] = len(mentions)
                            mentions.append(mention_dict)
                        else:
                            mention_dict['full_word_start'] = full_word_start
                            mention_dict['full_word_end'] = full_word_end
                            
                            mention_context_left = tensorizer.tokenizer.decode(batch_token_tensors[i][1:full_word_start]).strip()
                            if len(mention_context_left.split(" ")) > max_length:
                                mention_context_left = ' '.join(mention_context_left.split(" ")[max(0, -max_length):])
                            mention_dict['context_left'] = mention_context_left
                            
                            mention_context_right = tensorizer.tokenizer.decode(batch_token_tensors[i][full_word_end+1:]).strip()
                            if len(mention_context_right.split(" ")) > max_length:
                                mention_context_right = ' '.join(mention_context_right.split(" ")[:min(len(mention_context_right.split(" ")), max_length)])
                            mention_dict['context_right'] = mention_context_right.replace(' [PAD]', '')
                            
                            mention_dict['node_id'] = node_id_list[i]
                            mention_dict['mention_id'] = len(mentions)
                            mention_dict['mention'] = span
                            mentions.append(mention_dict)

                    thresholds.append(cut_off)
                    found_cells.append(cells)
                    found_mentions.append(mentions)
                torch.cuda.empty_cache()
        print ('breaking', breaking)
        print ('accepted', accepted, 'rejected', rejected, 'threshold', np.mean(thresholds))
        return found_cells, found_mentions