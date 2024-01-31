import hydra
import os
import json
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from mention_detector import MentionDetector
from entity_linker import COSEntityLinker, MVDEntityLinker
from view_generator import ViewGenerator
from index_builder import IndexBuilder
from transformers import BertTokenizer
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from utils.utils import process_mention, MentionInfo, prepare_datasource
import numpy as np
import pickle
from pymongo import MongoClient

@hydra.main(config_path="conf", config_name="graph_construction")
def main(cfg: DictConfig):
    # set mongoDB
    client = MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password))
    mongodb = client[cfg.dbname]

    data_sources = {}
    
    # mention detecting  
    if cfg.do_mention_detection:
        mentions = {}
        mention_detector_cfg = hydra.compose(config_name='mention_detector')
        mention_detector = MentionDetector(mention_detector_cfg, mongodb)
        source_types = ['table', 'passage'] if cfg.mention_detection['source_type'] == 'both' else [cfg.mention_detection['source_type']]
        
        collection_names = mongodb.list_collection_names()
        for source_type in source_types:
            mention_path = mention_detector_cfg[source_type]['result_path']
            mention_collection_name = os.path.basename(mention_path).split('.')[0]
            if not os.path.exists(mention_path) and mention_collection_name not in collection_names:
                os.makedirs(os.path.dirname(mention_path), exist_ok=True)
                data = prepare_datasource(cfg, mongodb, source_type)
                data_sources[source_type] = data
                mentions[source_type] = mention_detector.detect(data, source_type)
            else:
                collection = mongodb[mention_collection_name]
                total_num = collection.count_documents({})
                mentions[source_type] = [doc for doc in tqdm(collection.find(), total = total_num)]

    # entity linking
    if cfg.do_entity_linking:
        linking_type = cfg.entity_linking['linking_type']
        if linking_type == 'cos':
            
            entity_linking_cfg = hydra.compose(config_name=f'{linking_type}')
            entity_linker = COSEntityLinker(entity_linking_cfg, mongodb)
            
            source_types = ['table', 'passage'] if cfg.entity_linking['source_type'] == 'both' else [cfg.entity_linking['source_type']]
            dest_type = cfg.entity_linking['dest_type']
            
            for source_type in source_types:
                entity_linking_result_path = entity_linking_cfg['result_path'].split('.')[0] + '_' + source_type + '_' + dest_type +'.json'
                
                if not os.path.exists(entity_linking_result_path):
                    os.makedirs(os.path.dirname(entity_linking_result_path), exist_ok=True)
                    entity_linker.cfg.result_path = entity_linking_result_path
                    entity_linker.link(source_type, dest_type)
                    
        elif linking_type == 'mvd':
            
            entity_linking_cfg = hydra.compose(config_name=f'{linking_type}')
            
            source_types = ['table', 'passage'] if cfg.entity_linking['source_type'] == 'both' else [cfg.entity_linking['source_type']]
            dest_types = ['table', 'passage'] if cfg.entity_linking['dest_type'] == 'both' else [cfg.entity_linking['dest_type']]
            
            # view generating
            if entity_linking_cfg.do_view_generation:
                view_generator_cfg = entity_linking_cfg.view_generator
                view_generator = ViewGenerator(view_generator_cfg, mongodb)
                views = {}
                for dest_type in dest_types:
                    view_path = view_generator_cfg[dest_type]['view_path']
                    
                    if not os.path.exists(view_path):
                        os.makedirs(os.path.dirname(view_path), exist_ok=True)
                        
                        if dest_type in data_sources:
                            data = data_sources[dest_type]
                        else:
                            data = prepare_datasource(cfg, mongodb, dest_type)
                            data_sources[dest_type] = data
                        
                        views[dest_type] = view_generator.generate(data, dest_type)
                    else:
                        views[dest_type] = json.load(open(view_path, 'r'))
            else:
                views = None
            
            # index building
            index_builder_cfg = entity_linking_cfg.index_builder
            index_builder = IndexBuilder(index_builder_cfg, views)
            indicies, view2entity = index_builder.build(cfg.entity_linking['dest_type'])
            
            print('index building finished')
            entity_linker = MVDEntityLinker(entity_linking_cfg, indicies[cfg.entity_linking['dest_type']], view2entity[cfg.entity_linking['dest_type']], index_builder.embedder, mongodb)
            for source_type in source_types:
                
                entity_linking_result_path = entity_linking_cfg['result_path'].split('.')[0] + '_' + source_type + '_' + cfg.entity_linking['dest_type'] +'.json'
                
                if not os.path.exists(entity_linking_result_path):
                    os.makedirs(os.path.dirname(entity_linking_result_path), exist_ok=True)
                    entity_linker.cfg.result_path = entity_linking_result_path
                    detected_mentions = mentions[source_type]
                    entity_linker.link(source_type, detected_mentions)

if __name__ == "__main__":
    main()