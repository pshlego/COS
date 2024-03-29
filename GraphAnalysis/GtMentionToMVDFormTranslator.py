import json
from tqdm import tqdm

import hydra
from omegaconf import DictConfig
import json
import os
from typing import List


import sys
sys.path.insert(1, '/root/COS/Graph_constructer')
from dpr.options import setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
from dpr.utils.data_utils import Tensorizer
from dpr.models import init_biencoder_components
from utils.utils import locate_row, get_row_indices






ground_truth_hyperlink_path = "/mnt/sdf/shpark/mnt_sdc/shpark/cos/cos/OTT-QA/ott_dev_linking.json"
cos_recognized_hyperlink_path = "/mnt/sdf/shpark/mnt_sdc/shpark/graph/graph/for_test/all_table_chunks_span_prediction.json"


@hydra.main(config_path = "/root/COS/Graph_constructer/conf", config_name = "mention_detector")
def main(cfg: DictConfig):

    


    # 1. Parsing
    gt_hyperlinks = getGroundTruthHyperlinks()
    # with open('/mnt/sdf/shpark/mnt_sdc/shpark/cos/cos/knowledge/ott_wiki_passages.json') as f:
    #     ott_wiki_passages = json.load(f)
    

    # 2. Translate gt to mvd-form
    translated_objs = translateGtToMvdForm(gt_hyperlinks, cfg)


    # 3. Print file out
    with open("gt_dev_entities.json", 'w') as f:
        json.dump(translated_objs, f)



######################################################
# Translate                                          #
######################################################




def translateGtToMvdForm(gt_hyperlinks, config_for_tokenizer):
    
    def setup_encoder():
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{config_for_tokenizer.device_id}"
        
        cfg = setup_cfg_gpu(config_for_tokenizer)
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
    
    def get_left_right_contexts(tokenizer, tokenized_text, title_columns, row_indices, gt_entity, max_length):
        
        start_id = gt_entity["bert_start"]
        end_id = gt_entity["bert_end"]
        row_id = locate_row(start_id, end_id, row_indices)
        
        same_row_left = tokenizer.decode(tokenized_text[row_indices[row_id - 1] : start_id]).strip()
        mention_context_left = title_columns + ' [SEP] ' + same_row_left
        if len(mention_context_left.split(" ")) > max_length:
            mention_context_left = ' '.join(mention_context_left.split(" ")[max(0, -max_length):])
            
        mention_context_right = tokenizer.decode(tokenized_text[end_id+1:row_indices[row_id]]).strip()
        if len(mention_context_right.split(" ")) > max_length:
            mention_context_right = ' '.join(mention_context_right.split(" ")[:min(len(mention_context_right.split(" ")), max_length)])
        mention_context_right = mention_context_right.replace(' [PAD]', '')
        
        return mention_context_left, mention_context_right
    
    max_length = 128
    tokenizer = setup_encoder()
    
    translated_objs = []
    for i, chunk_info in enumerate(tqdm(gt_hyperlinks)):
    
        translated_obj = {}
        translated_obj["node_id"] = i
        translated_obj["chunk_id"] = chunk_info["chunk_id"]
    
        # 1. Get entities
        entities = chunk_info["positive_ctxs"]
    
        # 2. Fill in properties except 'grounding'
        
        translated_obj["title"] = chunk_info["question"].split(" [SEP] ")[0]
        text = chunk_info["question"].split(" [SEP] ")[1]
        translated_obj["text"] =  text
        
        tokenized_text = tokenizer.encode(text)
        row_indices = get_row_indices(text, tokenizer)
        title_columns = chunk_info["question"].split('\n')[0]
        
        # 3. Fill in 'grounding'
        translated_entities = []
        for j, gt_entity in enumerate(entities):
            translated_entity = {}
            translated_entity["mention_id"] = j
            translated_entity["full_word_start"] = gt_entity["bert_start"]
            translated_entity["full_word_end"] = gt_entity["bert_end"]
            
            context_left, context_right = get_left_right_contexts(
                tokenizer,
                tokenized_text, 
                title_columns,
                row_indices, 
                gt_entity, 
                max_length
            )
            translated_entity["context_left"] =  context_left
            translated_entity["context_right"] = context_right
            translated_entity["row_id"] = gt_entity["cell_id"].split("_")[-3]
            translated_entity["node_id"] = i
            translated_entity["mention"] = gt_entity["grounding"]
            translated_entity["across_row"] = False
            
            translated_entities.append(translated_entity)
            
        translated_obj["grounding"] = translated_entities
        
        translated_objs.append(translated_obj)

    return translated_objs

# class MVDTrainData:
#     def __init__(self, cfg, ott_train_linking_list, ott_wiki_passage_dict):
#         self.cfg = cfg
#         self.ott_train_linking_list = ott_train_linking_list
#         # self.ott_wiki_passage_dict = ott_wiki_passage_dict
        
#     def generate(self, train_data_path = "", max_length = 128):
#         with open(train_data_path, 'w') as file:
#             # use single GPU
#             os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.cfg.device_id}"
            
#             tensorizer = self.setup_encoder()
#             tokenizer = tensorizer.tokenizer
                        
#             for ott_train_linking in tqdm(self.ott_train_linking_list):
#                 text = ott_train_linking['question']
#                 tokenized_text = tokenizer.encode(text)
#                 row_indices = get_row_indices(text, tokenizer)
#                 title_column_name = text.split('\n')[0]
                
#                 for mention_info in ott_train_linking['positive_ctxs']:
#                     passage_title = mention_info['title']
                    
#                     # if passage_title in self.ott_wiki_passage_dict.keys():
#                     #     passage_id = int(self.ott_wiki_passage_dict[passage_title])
#                     # else: continue
                    
#                     mention = mention_info['grounding']
                    
#                     start_id = mention_info['bert_start']
#                     end_id = mention_info['bert_end']
#                     row_id = locate_row(start_id, end_id, row_indices)
                    
#                     same_row_left = tokenizer.decode(tokenized_text[row_indices[row_id-1]:start_id]).strip()
#                     mention_context_left = title_column_name + ' [SEP] ' + same_row_left
#                     if len(mention_context_left.split(" ")) > max_length:
#                         mention_context_left = ' '.join(mention_context_left.split(" ")[max(0, -max_length):])
                        
#                     mention_context_right = tensorizer.tokenizer.decode(tokenized_text[end_id+1:row_indices[row_id]]).strip()
#                     if len(mention_context_right.split(" ")) > max_length:
#                         mention_context_right = ' '.join(mention_context_right.split(" ")[:min(len(mention_context_right.split(" ")), max_length)])
#                     mention_context_right = mention_context_right.replace(' [PAD]', '')
                    
#                     json_line = json.dumps({'context_left': mention_context_left, 'context_right': mention_context_right, 'mention': mention, 'label_id': passage_id})
#                     file.write(json_line + '\n')
            
#             print("Number of missing passages: ", cnt)

    
#     def setup_encoder(self, ):
#         cfg = setup_cfg_gpu(self.cfg)
#         cfg.n_gpu = 1
        
#         cfg.encoder.encoder_model_type = 'hf_cos'
#         sequence_length = 512
#         cfg.encoder.sequence_length = sequence_length
#         cfg.encoder.pretrained_file=None
#         cfg.encoder.pretrained_model_cfg = 'bert-base-uncased'
        
#         tensorizer, _, _ = init_biencoder_components(
#             cfg.encoder.encoder_model_type, cfg, inference_only=True
#         )
        
#         return tensorizer











######################################################
# Parsing                                            #
######################################################


def getGroundTruthHyperlinks():
    with open(ground_truth_hyperlink_path, 'r') as f:
        data = json.load(f)
    return data


# def parseGroundTruthHyperlinks():
#     gt_object = getGroundTruthHyperlinks()
#     chunk_to_entities = {}
    
#     for table_hyperlinks in gt_object:
#         chunk_id = table_hyperlinks['chunk_id']
#         entities = []
#         for positive_ctx in table_hyperlinks['positive_ctxs']:
#             entity = (positive_ctx["bert_start"], positive_ctx["bert_end"])
#             entities.append(entity)
        
#         chunk_to_entities[chunk_id] = entities
    
#     return chunk_to_entities







if __name__ == "__main__":
    main()

