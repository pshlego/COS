import hydra
from omegaconf import DictConfig
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

def find_span_indices(text, span, tokenizer, row_start_idx, row_end_idx):
    # Tokenize the text and the span
    tokens = tokenizer.encode(text)
    span_tokens = tokenizer.encode(span, add_special_tokens=False)

    # Initialize indices
    start_index = None
    end_index = None

    # Search for the start token of the span in the tokenized text
    for i, token in enumerate(tokens):
        # If the current token matches the first token of the span
        # and the subsequent tokens in the text match the rest of the span tokens
        if tokens[i:i+len(span_tokens)] == span_tokens and (i >= row_start_idx and i <= row_end_idx):
            start_index = i
            end_index = i + len(span_tokens)
            break  # Exit the loop once the span is found

    if start_index is not None and end_index is not None:
        return start_index, end_index
    else:
        return None, None

class GoldDataConstructor:
    def __init__(self, cfg, target_table_id_to_table_info, wiki_passage_title_to_ott_id):
        self.cfg = cfg
        self.target_table_id_to_table_info = target_table_id_to_table_info
        
        self.wiki_passage_title_to_ott_id = wiki_passage_title_to_ott_id
        
    def generate(self, output_path = "", max_length = 128):
        with open(output_path, 'w') as file:
            # use single GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.cfg.device_id}"
            
            tokenizer = self.setup_tokenizer()
            
            invalid_passage_cnt = 0
            
            for _, table_info in tqdm(self.target_table_id_to_table_info.items()):
                
                text = ''
                max_chunk_num = len(table_info['chunk_num_to_chunk'])
                for chunk_num in range(max_chunk_num):
                    chunk = table_info['chunk_num_to_chunk'][chunk_num]
                    
                    if chunk_num == 0:
                        text += chunk['title'] + ' [SEP] '
                        text += chunk['text']
                    else:
                        text += '\n'.join(chunk['text'].split('\n')[1:])
                    
                tokenized_text = tokenizer.encode(text)
                row_indices = get_row_indices(text, tokenizer)
                title_column_name = text.split('\n')[0]
                
                # [
                #    # per row
                #    [
                #       (mention_1, linked_passage_1),
                #       ....,
                #       (mention_n, linked_passage_n)
                #    ],
                #    ...,
                # ]
                #
                hyperlinks = table_info['hyperlinks']
                
                for row_id, hyperlinks_per_row in enumerate(hyperlinks):
                    row_start_idx = row_indices[row_id]
                    row_end_idx = row_indices[row_id+1] - 1
                    
                    for mention, linked_passage_title in hyperlinks_per_row:
                        
                        if linked_passage_title in self.wiki_passage_title_to_ott_id.keys():
                            ott_id = int(self.wiki_passage_title_to_ott_id[linked_passage_title])
                        else:
                            invalid_passage_cnt += 1
                            continue
                        
                        start_idx, end_idx = find_span_indices(text, mention, tokenizer, row_start_idx, row_end_idx)
                        same_row_left = tokenizer.decode(tokenized_text[row_indices[row_id]:start_idx]).strip()
                        mention_context_left = title_column_name + ' [SEP] ' + same_row_left
                        
                        if len(mention_context_left.split(" ")) > max_length:
                            mention_context_left = ' '.join(mention_context_left.split(" ")[max(0, -max_length):])
                            
                        mention_context_right = tokenizer.decode(tokenized_text[end_idx:row_indices[row_id+1]]).strip()
                        
                        if len(mention_context_right.split(" ")) > max_length:
                            mention_context_right = ' '.join(mention_context_right.split(" ")[:min(len(mention_context_right.split(" ")), max_length)])
                        
                        mention_context_right = mention_context_right.replace(' [PAD]', '')
                        
                        json_line = json.dumps({'context_left': mention_context_left, 'context_right': mention_context_right, 'mention': mention, 'label_id': ott_id})
                        file.write(json_line + '\n')        
            
        print("Number of invalid passages: ", invalid_passage_cnt)

    
    def setup_tokenizer(self, ):
        cfg = setup_cfg_gpu(self.cfg)
        cfg.n_gpu = 1
        
        cfg.encoder.encoder_model_type = 'hf_cos'
        sequence_length = 512
        cfg.encoder.sequence_length = sequence_length
        cfg.encoder.pretrained_file=None
        cfg.encoder.pretrained_model_cfg = 'bert-base-uncased'
        
        tensorizer, _, _ = init_biencoder_components(
            cfg.encoder.encoder_model_type, cfg, inference_only=True
        )
        
        return tensorizer.tokenizer

@hydra.main(config_path="conf", config_name="mention_detector")
def main(cfg: DictConfig):
    # Path to the output directory
    output_dir_path = '/mnt/sdf/shpark/MVD_OTT'
    
    # Path to the train/dev/test table ids
    train_dev_test_table_ids_path = '/home/shpark/OTT-QA/released_data/train_dev_test_table_ids.json'
    
    # Corpus paths
    ## Table
    ott_table_chunks_path = '/mnt/sdf/shpark/mnt_sdc/shpark/cos/cos/knowledge/ott_table_chunks_original.json'
    hyperlink_path = '/mnt/sdf/shpark/wiki_hyperlink/wiki_hyperlink.json'
    
    ## Passage
    ott_wiki_passages_path = '/mnt/sdf/shpark/mnt_sdc/shpark/cos/cos/knowledge/ott_wiki_passages.json'
    
    with open(hyperlink_path) as f:
        wiki_hyperlinks = json.load(f)
    
    with open(train_dev_test_table_ids_path) as f:
        train_dev_test_table_ids = json.load(f)
    
    with open(ott_table_chunks_path) as f:
        ott_table_chunks_original = json.load(f)
    
    with open(ott_wiki_passages_path) as f:
        ott_wiki_passages = json.load(f)
    
    target_split_types = [
        'train', 
        'dev', 
        # 'test'
        ]
    
    for target_split_type in target_split_types:
        
        print(f"Processing {target_split_type} split")
        target_table_ids = train_dev_test_table_ids[target_split_type]
        output_path = os.path.join(output_dir_path, target_split_type, f"{target_split_type}.jsonl")

        # Get the linking data for the target split
        target_table_id_to_table_info = {}
        
        for ott_table_chunk in ott_table_chunks_original:
            chunk_id = ott_table_chunk['chunk_id']
            table_id = '_'.join(chunk_id.split('_')[:-1])
            
            if table_id not in target_table_ids:
                continue
            
            chunk_num = int(chunk_id.split('_')[-1])
            
            if table_id not in target_table_id_to_table_info:
                target_table_id_to_table_info[table_id] = {
                    'hyperlinks': wiki_hyperlinks[table_id]
                }
                target_table_id_to_table_info[table_id]['chunk_num_to_chunk'] = {}

            target_table_id_to_table_info[table_id]['chunk_num_to_chunk'][chunk_num] = ott_table_chunk
        
        # Create a dictionary for the passage titles
        wiki_passage_title_to_ott_id = {}
        for ott_id, wiki_passage in enumerate(ott_wiki_passages):
            wiki_passage_title = wiki_passage['title']
            wiki_passage_title_to_ott_id[wiki_passage_title] = ott_id

        gold_data_constructor = GoldDataConstructor(cfg, target_table_id_to_table_info, wiki_passage_title_to_ott_id)
        gold_data_constructor.generate(output_path = output_path)

if __name__ == "__main__":
    main()