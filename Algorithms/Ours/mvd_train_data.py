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

class MVDTrainData:
    def __init__(self, cfg, ott_train_linking_list, ott_wiki_passage_dict):
        self.cfg = cfg
        self.ott_train_linking_list = ott_train_linking_list
        self.ott_wiki_passage_dict = ott_wiki_passage_dict
        
    def generate(self, train_data_path = "", max_length = 128):
        with open(train_data_path, 'w') as file:
            # use single GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.cfg.device_id}"
            
            tensorizer = self.setup_encoder()
            tokenizer = tensorizer.tokenizer
            
            cnt = 0
            
            for ott_train_linking in tqdm(self.ott_train_linking_list):
                text = ott_train_linking['question']
                tokenized_text = tokenizer.encode(text)
                row_indices = get_row_indices(text, tokenizer)
                title_column_name = text.split('\n')[0]
                
                for mention_info in ott_train_linking['positive_ctxs']:
                    passage_title = mention_info['title']
                    
                    if passage_title in self.ott_wiki_passage_dict.keys():
                        passage_id = int(self.ott_wiki_passage_dict[passage_title])
                    else:
                        cnt += 1
                        continue
                    
                    
                    mention = mention_info['grounding']
                    
                    start_id = mention_info['bert_start']
                    end_id = mention_info['bert_end']
                    row_id = locate_row(start_id, end_id, row_indices)
                    
                    same_row_left = tokenizer.decode(tokenized_text[row_indices[row_id-1]:start_id]).strip()
                    mention_context_left = title_column_name + ' [SEP] ' + same_row_left
                    if len(mention_context_left.split(" ")) > max_length:
                        mention_context_left = ' '.join(mention_context_left.split(" ")[max(0, -max_length):])
                        
                    mention_context_right = tensorizer.tokenizer.decode(tokenized_text[end_id+1:row_indices[row_id]]).strip()
                    if len(mention_context_right.split(" ")) > max_length:
                        mention_context_right = ' '.join(mention_context_right.split(" ")[:min(len(mention_context_right.split(" ")), max_length)])
                    mention_context_right = mention_context_right.replace(' [PAD]', '')
                    
                    json_line = json.dumps({'context_left': mention_context_left, 'context_right': mention_context_right, 'mention': mention, 'label_id': passage_id})
                    file.write(json_line + '\n')
            
            print("Number of missing passages: ", cnt)

    
    def setup_encoder(self, ):
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
        
        return tensorizer

@hydra.main(config_path="conf", config_name="mention_detector")
def main(cfg: DictConfig):
    # with open('/mnt/sdf/shpark/mnt_sdc/shpark/cos/cos/OTT-QA/ott_train_linking.json') as f:
    #     ott_train_linking_list = json.load(f)
    with open('/mnt/sdf/shpark/mnt_sdc/shpark/cos/cos/OTT-QA/ott_dev_linking.json') as f:
        ott_train_linking_list = json.load(f)
    with open('/mnt/sdf/shpark/mnt_sdc/shpark/cos/cos/knowledge/ott_wiki_passages.json') as f:
        ott_wiki_passages = json.load(f)
    
    ott_wiki_passage_dict = {}
    
    for passage_id, ott_wiki_passage in enumerate(ott_wiki_passages):
        ott_wiki_passage_dict[ott_wiki_passage['title']] = passage_id

    train_data_generator = MVDTrainData(cfg, ott_train_linking_list, ott_wiki_passage_dict)
    train_data_generator.generate(train_data_path = "/mnt/sdf/shpark/MVD_OTT/dev.jsonl")

if __name__ == "__main__":
    main()