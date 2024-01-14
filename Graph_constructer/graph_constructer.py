import hydra
import os
import json
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from mention_detector import MentionDetector
from view_generator import ViewGenerator
from index_builder import IndexBuilder
from transformers import BertTokenizer
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from utils.utils import process_mention, MentionInfo, prepare_datasource
import numpy as np
import pickle
from pymongo import MongoClient

class GraphConstructer:
    def __init__(self, cfg, table_mentions_cursor, passage_mentions_cursor, indicies, view2entity, embedder, device):
        # TODO: Change the cfg according to the below code.
        self.cfg = cfg
        self.table_mentions = table_mentions_cursor
        self.passage_mentions = passage_mentions_cursor
        self.index = indicies['passage']
        self.view2entity = view2entity['passage']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.mention_embedder = embedder
        self.device = device

    def construct(self, data_type='table'):
        if data_type == 'table':
            table_mention_queries = self.prepate_mention_queries('table')
            mention_queries = table_mention_queries
        else:
            passage_mention_queries = self.prepate_mention_queries('passage')
            mention_queries = passage_mention_queries

        all_mention_ids = torch.tensor(
            [f.mention_ids for f in mention_queries], dtype=torch.long)
        all_node_ids = torch.tensor(
            [f.node_id for f in mention_queries], dtype=torch.long)
        query_data = TensorDataset(all_mention_ids,all_node_ids)
        query_sampler = SequentialSampler(query_data)
        dataloader = DataLoader(query_data, sampler=query_sampler, batch_size=self.cfg.batch_size)
        graph = self.entity_linking(dataloader)
        return graph

    def prepate_mention_queries(self, data_type='table'):
        if data_type == 'table':
            mention_queries_path = self.cfg.table_mention_queries_path
            os.makedirs(os.path.dirname(mention_queries_path), exist_ok=True)
            if not os.path.exists(mention_queries_path):
                mention_queries = self.prepare_queries(data_type)
                with open(mention_queries_path, 'wb') as handle:
                    pickle.dump(mention_queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(mention_queries_path, "rb") as file:
                    mention_queries = pickle.load(file)
        else:
            mention_queries_path_1 = self.cfg.passage_mention_queries_path.replace('.pkl','_1.pkl')
            mention_queries_path_2 = self.cfg.passage_mention_queries_path.replace('.pkl','_2.pkl')
            os.makedirs(os.path.dirname(mention_queries_path_1), exist_ok=True)
            if not os.path.exists(mention_queries_path_1):
                mention_queries = self.prepare_queries(data_type)
                half_index = int(len(mention_queries)/2)
                with open(mention_queries_path_1, 'wb') as handle:
                    pickle.dump(mention_queries[:half_index], handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(mention_queries_path_2, 'wb') as handle:
                    pickle.dump(mention_queries[half_index:], handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(mention_queries_path_1, "rb") as file:
                    mention_queries_1 = pickle.load(file)
                with open(mention_queries_path_2, "rb") as file:
                    mention_queries_2 = pickle.load(file)
                mention_queries = mention_queries_1 + mention_queries_2
        return mention_queries
    
    def entity_linking(self, dataloader):
        mention_embeds, node_ids, cand_entities_list = list(),list(),list()
        for mention_ids, node_id in tqdm(dataloader):
            mention_ids = mention_ids.to(self.device)
            with torch.no_grad():
                mention_embed = self.mention_embedder(mention_ids=mention_ids)
                mention_embed = mention_embed.detach().cpu().numpy().astype('float32')
                mention_embeds.append(mention_embed)
                top_k = self.cfg.top_k * 30 + 1
                _, closest_entities = self.index.search(mention_embed, top_k)
                cand_entities = self.get_distinct_entities(closest_entities, 'topk')
                cand_entities_list.extend(cand_entities)
                node_ids.extend(node_id.tolist())
                
        # mention_embeds = np.concatenate(mention_embeds, axis=0)
        # cand_entities_list = np.concatenate(cand_entities_list, axis=0)
        entity_linking_result  = []
        node_list = []
        entity_linking_dict = {}
        for node_id, cand_idx in tqdm(zip(node_ids, cand_entities_list), total=len(node_ids)):
            if node_id not in node_list:
                if int(node_id) != 0:
                    entity_linking_result.append(entity_linking_dict)
                node_list.append(node_id)
                entity_linking_dict = {}
                entity_linking_dict['node_id'] = node_id
                entity_linking_dict['linked_entities'] = []
            mention_dict = {}
            mention_dict['mention_id'] = len(entity_linking_dict['linked_entities'])
            mention_dict['linked_entity'] = cand_idx
            entity_linking_dict['linked_entities'].append(mention_dict)
        return entity_linking_result

    def get_distinct_entities(self, closest_entities, type):
        mention_num = len(closest_entities)
        pred_entity_idxs = list()
        for i in range(mention_num):
            pred_entity_idx = [eidx for eidx in closest_entities[i][1:]]
            if self.view2entity is not None:
                pred_entity_idx = [self.view2entity[str(eidx)] for eidx in closest_entities[i][1:]]
            new_pred_entity_idx = list()
            for item in pred_entity_idx:
                if type == 'topk':
                    if item not in new_pred_entity_idx and len(new_pred_entity_idx) < self.cfg.top_k:
                        new_pred_entity_idx.append(item)
                else:
                    if item not in new_pred_entity_idx:
                        new_pred_entity_idx.append(item)
            pred_entity_idxs.append(new_pred_entity_idx)
        
        return pred_entity_idxs

    def prepare_queries(self, data_type):
        mention_queries = []
        if data_type == 'table':
            data_mention = self.table_mentions
        else:
            data_mention = self.passage_mentions
        for datum_mention in tqdm(data_mention, desc="Preparing queries"):
            mentions = datum_mention['grounding']
            for id, mention_dict in enumerate(mentions):
                if 'row_id' in mention_dict.keys():
                    data_type = 'table'
                else:
                    data_type = 'passage'
                mention_ids, mention_tokens = process_mention(self.tokenizer, mention_dict, self.cfg.max_seq_length)
                mention_queries.append(MentionInfo(
                                datum_mention=mention_tokens,
                                node_id=mention_dict['node_id'],
                                row_id= mention_dict['row_id'] if data_type == 'table' else None,
                                mention_ids=mention_ids,
                                mention_id=mention_dict['mention_id'],
                                mention_tokens = mention_tokens,
                                data_type = data_type)
                                )
        return mention_queries

@hydra.main(config_path="conf", config_name="graph_constructer")
def main(cfg: DictConfig):
    # Set up device
    device = torch.device(
            "cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    
    # Set MongoDB
    client = MongoClient(f"mongodb://localhost:{cfg.port}/", username=cfg.username, password=str(cfg.password))
    mongodb = client[cfg.dbname]
    
    # Preprocess data
    ## Mention detection
    mention_detector_cfg = hydra.compose(config_name='mention_detector')
    
    table_mention_path = mention_detector_cfg.table_mention_path
    table_mention_collection_name = os.path.basename(table_mention_path).split('.')[0]
    os.makedirs(os.path.dirname(table_mention_path), exist_ok=True)
    
    passage_mention_path = mention_detector_cfg.passage_mention_path
    passage_mention_collection_name = os.path.basename(passage_mention_path).split('.')[0]
    os.makedirs(os.path.dirname(passage_mention_path), exist_ok=True)
    
    if cfg.do_mention_detection:
        mention_detector = MentionDetector(mention_detector_cfg, mongodb)
        if not os.path.exists(table_mention_path):
            all_tables = prepare_datasource(cfg, mongodb, 'table')
            mention_detector.span_proposal(all_tables=all_tables)
        if not os.path.exists(passage_mention_path):
            all_passages = prepare_datasource(cfg, mongodb, 'passage')
            mention_detector.span_proposal(all_passages=all_passages)
    
    ## Decompose descriptions into hierarchical views
    view_generator_cfg = hydra.compose(config_name='view_generator')
    
    table_view_path = view_generator_cfg.table_view_path
    table_view_collection_name = os.path.basename(table_view_path).split('.')[0]
    os.makedirs(os.path.dirname(table_view_path), exist_ok=True)
    
    passage_view_path = view_generator_cfg.passage_view_path
    passage_view_collection_name = os.path.basename(passage_view_path).split('.')[0]
    os.makedirs(os.path.dirname(passage_view_path), exist_ok=True)
    
    if cfg.do_view_generation:
        view_generator = ViewGenerator(view_generator_cfg, mongodb)
        if not os.path.exists(table_view_path):
            if all_tables is None:
                all_tables = prepare_datasource(cfg, mongodb, 'table')
            view_generator.generate(all_tables=all_tables)
        if not os.path.exists(passage_view_path):
            if all_passages is None:
                all_passages = prepare_datasource(cfg, mongodb, 'passage')
            view_generator.generate(all_passages=all_passages)

    # Construct graph
    ## Build index for all views
    index_builder_cfg = hydra.compose(config_name='index_builder')
    table_views_cursor = mongodb[table_view_collection_name].find()
    passage_views_cursor = mongodb[passage_view_collection_name].find()
    index_builder = IndexBuilder(index_builder_cfg, table_views_cursor, passage_views_cursor, device, n_gpu)
    indicies, view2entity = index_builder.build()
    ## Link mentions to views with index
    
    table_mentions_cursor = mongodb[table_mention_collection_name].find()
    passage_mentions_cursor = mongodb[passage_mention_collection_name].find()
    graph_constructer = GraphConstructer(cfg, table_mentions_cursor, passage_mentions_cursor, indicies, view2entity, index_builder.embedder, device)
    table_graph = graph_constructer.construct('table')
    if not os.path.exists(cfg.table_graph_path):
        with open(cfg.table_graph_path, 'w') as fout:
            json.dump(table_graph, fout, indent=4)

    # passage_graph = graph_constructer.construct('passage')
    # if not os.path.exists(cfg.passage_graph_path):
    #     with open(cfg.passage_graph_path, 'w') as fout:
    #         json.dump(passage_graph, fout, indent=4)

if __name__ == "__main__":
    main()