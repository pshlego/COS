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
    
    
    ott_table_chunks_path = '/mnt/sdf/shpark/mnt_sdc/shpark/cos/cos/knowledge/ott_table_chunks_original.json'
    hyperlink_path = '/mnt/sdf/shpark/wiki_hyperlink/wiki_hyperlink.json'
    train_dev_test_table_ids_path = '/home/shpark/OTT-QA/released_data/train_dev_test_table_ids.json'
    with open(train_dev_test_table_ids_path) as f:
        train_dev_test_table_ids = json.load(f)
    with open(ott_table_chunks_path) as f:
        ott_table_chunks_original = json.load(f)
    with open(hyperlink_path) as f:
        wiki_hyperlinks = json.load(f)
        
    with open('/mnt/sdf/Corpus/raw_tables.json') as f:
            raw_tables = json.load(f)
    
    raw_tables_dict = {}
    
    for raw_table in raw_tables:
        raw_tables_dict[raw_table['table_id']] = raw_table        

    target_table_ids = train_dev_test_table_ids['dev']
    
    # Get the linking data for the target split
    target_table_id_to_table_info = {}
    
    for ott_table_chunk in tqdm(ott_table_chunks_original):
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
        ott_table_chunk['row_num'] = len(ott_table_chunk['text'].split('\n')[1:-1])
        target_table_id_to_table_info[table_id]['chunk_num_to_chunk'][chunk_num] = ott_table_chunk

    # write the target_table_id_to_table_info to a file
    with open('/mnt/sde/shpark/graph_constructer/mention_detector/gold/target_table_id_to_table_info.json', 'w') as f:
        json.dump(target_table_id_to_table_info, f)
    
if __name__ == "__main__":
    main()