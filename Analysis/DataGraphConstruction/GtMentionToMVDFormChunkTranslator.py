import json
from tqdm import tqdm

import hydra
from omegaconf import DictConfig
import json
import os
from typing import List


import sys
sys.path.insert(1, '/root/COS/Graph_constructer')
from DPR.dpr.options import setup_cfg_gpu, set_cfg_params_from_state
from DPR.dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
from DPR.dpr.utils.data_utils import Tensorizer
from DPR.dpr.models import init_biencoder_components
from Graph_constructer.utils.utils import locate_row, get_row_indices






ground_truth_hyperlink_path = '/mnt/sdf/shpark/MVD_OTT/valid.jsonl'
cos_recognized_hyperlink_path = "/mnt/sdf/shpark/mnt_sdc/shpark/graph/graph/for_test/all_table_chunks_span_prediction.json"


@hydra.main(config_path = "/home/shpark/COS/Graph_constructer/conf", config_name = "mention_detector")
def main(cfg: DictConfig):

    


    # 1. Parsing
    gt_hyperlinks = getGroundTruthHyperlinks()

    with open('/mnt/sde/shpark/graph_constructer/mention_detector/gold/target_table_id_to_table_info.json') as f:
        target_table_id_to_table_info = json.load(f)

    with open('/mnt/sdf/Corpus/raw_tables.json') as f:
        raw_tables = json.load(f)
    
    raw_tables_dict = {}
    
    for raw_table in raw_tables:
        raw_tables_dict[raw_table['table_id']] = raw_table

    # 2. Translate gt to mvd-form
    translated_objs = translateGtToMvdForm(gt_hyperlinks, raw_tables_dict, target_table_id_to_table_info, cfg)


    # 3. Print file out
    with open("/mnt/sde/shpark/graph_constructer/mention_detector/gold/gt_dev_entities_chunks.json", 'w') as f:
        json.dump(translated_objs, f)



######################################################
# Translate                                          #
######################################################




def translateGtToMvdForm(gt_hyperlinks, raw_tables_dict, target_table_id_to_table_info, config_for_tokenizer):
    
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
    
    tokenizer = setup_encoder()
    ###### Stolpersteine_in_Plzeň_Region_1_0라는 table의 마지막 column이 text가 너무 많은데, COS에서는 해당 column을 제거한듯.
    translated_objs_dict = {}
    converted_tables = set()
    chunk_mention_count = {}
    for i, chunk_info in enumerate(tqdm(gt_hyperlinks)):
        if chunk_info["table_id"] not in converted_tables:
            table_info = raw_tables_dict[chunk_info["table_id"]]
            chunk_num_to_chunk = target_table_id_to_table_info[chunk_info["table_id"]]['chunk_num_to_chunk']
            chunk_nums = len(chunk_num_to_chunk)
            row_num = 0
            total_text = table_info['title'] + ' [SEP] ' + table_info['text']
            row_indices = get_row_indices(total_text, tokenizer)
            chunk_section_list = []
            for chunk_num in range(chunk_nums):
                chunk_id = chunk_info["table_id"] + '_' + str(chunk_num)
                chunk = chunk_num_to_chunk[str(chunk_num)]
                title = table_info['title']
                text = table_info['text'].split('\n')[0] + '\n' + '\n'.join(table_info['text'].split('\n')[row_num + 1:row_num + chunk['row_num']+1]) + '\n'
                row_section = (row_num, row_num+chunk['row_num'])
                chunk_section_list.append(row_section)
                row_num += chunk['row_num']
                translated_objs_dict[chunk_id] = {}
                translated_objs_dict[chunk_id]['title'] = title
                translated_objs_dict[chunk_id]['text'] = text
                translated_objs_dict[chunk_id]['grounding'] = []
            converted_tables.add(chunk_info["table_id"])
        
        start_id = chunk_info['start_idx']
        end_id = chunk_info['end_idx'] - 1
        row_id = locate_row(start_id, end_id, row_indices) - 1
        for chunk_num, row_section in enumerate(chunk_section_list):
            if row_id >= row_section[0] and row_id < row_section[1]:
                chunk_id = chunk_info["table_id"] + '_' + str(chunk_num)
                if row_section[0] != 0:
                    start_id -= len(tokenizer.tokenize('\n'.join(table_info['text'].split('\n')[1:1+row_section[0]])))
                    end_id -= len(tokenizer.tokenize('\n'.join(table_info['text'].split('\n')[1:1+row_section[0]])))
                    row_id -= row_section[0]
                break
        if chunk_id not in chunk_mention_count:
            chunk_mention_count[chunk_id] = 0
        else:
            chunk_mention_count[chunk_id] += 1
        translated_objs_dict[chunk_id]['grounding'].append({'mention':chunk_info['mention'], 'full_word_start':start_id, 'full_word_end':end_id, 'context_left':chunk_info['context_left'], 'context_right':chunk_info['context_right'], 'row_id':row_id, 'node_id':len(translated_objs_dict.keys()), 'across_row':False, 'mention_id':chunk_mention_count[chunk_id]})
    
    translated_objs = []
    for node_id, (chunk_id, translated_obj) in enumerate(translated_objs_dict.items()):
        translated_obj['chunk_id'] = chunk_id
        translated_obj['node_id'] = node_id
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
    data = []
    with open(ground_truth_hyperlink_path) as f:
        for line in f:
            data.append(json.loads(line))
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

