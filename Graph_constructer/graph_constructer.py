import hydra
import os
import json
import torch
from omegaconf import DictConfig
from mention_detector import MentionDetector
from view_generator import ViewGenerator
from index_builder import IndexBuilder

@hydra.main(config_path="conf", config_name="graph_constructer")
def main(cfg: DictConfig):
    # Set up device
    device = torch.device(
            "cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    
    # Prepare data
    all_tables_path = cfg.ctx_sources[cfg.ctx_src_table]['file']
    all_passages_path = cfg.ctx_sources[cfg.ctx_src_passage]['file']
    all_tables = None
    all_passages = None
    
    # Preprocess data
    ## Mention detection (Named Entity Recognition)
    mention_detector_cfg = hydra.compose(config_name='mention_detector')
    table_mention_path = mention_detector_cfg.table_mention_path
    os.makedirs(os.path.dirname(table_mention_path), exist_ok=True)
    passage_mention_path = mention_detector_cfg.passage_mention_path
    os.makedirs(os.path.dirname(passage_mention_path), exist_ok=True)
    
    if cfg.do_mention_detection:
        if all_tables is None:
            all_tables = json.load(open(all_tables_path, 'r'))
        if all_passages is None:
            all_passages = json.load(open(all_passages_path, 'r'))

        mention_detector = MentionDetector(mention_detector_cfg, all_tables, all_passages)
        if not os.path.exists(table_mention_path):
            table_mention_path = mention_detector.span_proposal('table')
        
        if not os.path.exists(passage_mention_path):
            passage_mention_path = mention_detector.span_proposal('passage')
    
    ## Decompose descriptions into Hierarchical views
    view_generator_cfg = hydra.compose(config_name='view_generator')
    table_view_path = view_generator_cfg.table_view_path
    os.makedirs(os.path.dirname(table_view_path), exist_ok=True)
    passage_view_path = view_generator_cfg.passage_view_path
    os.makedirs(os.path.dirname(passage_view_path), exist_ok=True)
    
    if cfg.do_view_generation:
        if all_tables is None:
            all_tables = json.load(open(all_tables_path, 'r'))
        if all_passages is None:
            all_passages = json.load(open(all_passages_path, 'r'))

        view_generator = ViewGenerator(view_generator_cfg, all_tables, all_passages)
        if not os.path.exists(table_view_path):
            table_view_path = view_generator.generate('table')
            
        if not os.path.exists(passage_view_path):
            passage_view_path = view_generator.generate('passage')

    with open(table_view_path, 'r') as file:
        table_views = json.load(file)

    with open(passage_view_path, 'r') as file:
        passage_views = json.load(file)

    # Construct graph
    ## Build index for all views
    if cfg.do_index_building:
        index_builder_cfg = hydra.compose(config_name='index_builder')
        index_builder = IndexBuilder(index_builder_cfg, table_views, passage_views, device, n_gpu)
        indicies = index_builder.build()
        
    ## Link mentions to views with index
    with open(table_mention_path, 'r') as file:
        table_mentions = json.load(file)

    with open(passage_mention_path, 'r') as file:
        passage_mentions = json.load(file)
        
    table_index = indicies['table']
    passage_index = indicies['passage']
    
if __name__ == "__main__":
    main()