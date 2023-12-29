import hydra
import os
import json
import torch
from omegaconf import DictConfig
from mention_detector import MentionDetector
from view_generator import ViewGenerator
from index_builder import IndexBuilder
from model import RetrievalModel,TeacherModel,MVD
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

def load_embedder(cfg):
    men_bert = BertModel.from_pretrained(cfg.bert_model)
    ent_bert = BertModel.from_pretrained(cfg.bert_model)
    tech_bert = BertModel.from_pretrained(cfg.bert_model)
    retriever = RetrievalModel(men_bert,ent_bert)
    teacher = TeacherModel(tech_bert)
    if cfg.pretrain_retriever:
        retriever.load_state_dict(torch.load(cfg.pretrain_retriever,map_location='cpu'),strict=False)
    if cfg.pretrain_teacher:
        teacher.load_state_dict(torch.load(cfg.pretrain_teacher,map_location='cpu'),strict=False)
    if cfg.task_name == 'mvd':
        model = MVD(retriever=retriever,teacher=teacher)
        config = retriever.mention_encoder.config
    if cfg.task_name == 'retriever':
        model = retriever
        config = retriever.mention_encoder.config
    elif cfg.task_name == 'teacher':
        model = teacher
        config = teacher.rank_encoder.config
    return model,config

@hydra.main(config_path="conf", config_name="graph_constructer")
def main(cfg: DictConfig):
    device = torch.device(
            "cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    
    # Load data
    # all_tables_dict = {}
    # ctx_src_table = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src_table])
    # ctx_src_table.load_data_to(all_tables_dict)
    # all_tables = [(k, v) for k, v in all_tables_dict.items()]
    
    # all_passages_dict = {}
    # ctx_src_passage = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src_passage])
    # ctx_src_passage.load_data_to(all_passages_dict)
    # all_passages = [(k, v) for k, v in all_passages_dict.items()]
    
    # Preprocess data
        ## Mention detection (Named Entity Recognition)
    mention_detector_cfg = hydra.compose(config_name='mention_detector')
    table_mention_path = '/'.join(mention_detector_cfg.model_file.split('/')[:-1]) + '/all_table_chunks_span_prediction.json'
    passage_mention_path = '/'.join(mention_detector_cfg.model_file.split('/')[:-1]) + '/all_passage_chunks_span_prediction.json'
    if cfg.do_mention_detection:
        mention_detector = MentionDetector(mention_detector_cfg, cfg.ctx_sources[cfg.ctx_src_table]['file'], cfg.ctx_sources[cfg.ctx_src_passage]['file'])
        if not os.path.exists(table_mention_path):
            table_mention_path = mention_detector.span_proposal('table')
        
        if not os.path.exists(passage_mention_path):
            passage_mention_path = mention_detector.span_proposal('passage')

    # with open(table_mention_path, 'r') as file:
    #     table_mentions = json.load(file)

    # with open(passage_mention_path, 'r') as file:
    #     passage_mentions = json.load(file)
    
    ## Decompose descriptions into Hierarchical views
    view_generator_cfg = hydra.compose(config_name='view_generator')
    table_view_path = view_generator_cfg.table_view_path
    passage_view_path = view_generator_cfg.passage_view_path
    
    if cfg.do_view_generation:
        view_generator = ViewGenerator(view_generator_cfg, cfg.ctx_sources[cfg.ctx_src_table]['file'], cfg.ctx_sources[cfg.ctx_src_passage]['file'])
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
        embedder, embedder_config = load_embedder(index_builder_cfg)
        embedder.to(device)
        if n_gpu > 1:
            embedder = torch.nn.DataParallel(embedder)
        index_builder = IndexBuilder(index_builder_cfg, table_views, passage_views, embedder)
        indicies = index_builder.build()
    
    ## Link mentions to views with index
    
if __name__ == "__main__":
    main()