import torch
from utils.utils import EntityDataset
from torch.utils.data import DataLoader
from transformers import BertModel
from tqdm import tqdm
import os
import gc
import pickle
import faiss
import numpy as np
from utils.model import RetrievalModel,TeacherModel,MVD

class IndexBuilder:
    def __init__(self, cfg, table_views, passage_views, device, n_gpu):
        self.cfg = cfg
        self.table_views = table_views
        self.passage_views = passage_views
        self.local_views = dict()
        self.global_views = dict()
        self.local_views['table'] = list()
        self.global_views['table'] = list()
        self.local_views['passage'] = list()
        self.global_views['passage'] = list()
        
        for view in self.table_views['doc_list']:
            self.local_views['table'].append(view['local_ids'])
            self.global_views['table'].append(view['global_ids'])

        for view in self.passage_views['doc_list']:
            self.local_views['passage'].append(view['local_ids'])
            self.global_views['passage'].append(view['global_ids'])

        embedder, _ = self.load_embedder(cfg)
        embedder.to(device)
        if n_gpu > 1:
            embedder = torch.nn.DataParallel(embedder)
        self.embedder = embedder

    def build(self,):
        entity_embedding, entity_embedding2id, data_type_idx, view2entity = self.get_entity_embedding()
        # Begin entity_embedding reorder
        new_entity_embedding = entity_embedding.copy()
        for i in range(entity_embedding.shape[0]):
            new_entity_embedding[entity_embedding2id[i]] = entity_embedding[i]
        del entity_embedding,entity_embedding2id
        gc.collect()
        entity_embedding = new_entity_embedding
        dim = entity_embedding.shape[1]
        
        # Build ANN Index
        indicies={}
        for data_type in ['table', 'passage']:
            new_embedding = entity_embedding[data_type_idx[data_type][0]:data_type_idx[data_type][1]]
            indicies[data_type] = faiss.IndexFlatIP(dim)
            indicies[data_type].add(new_embedding.astype(np.float32))
        
        return indicies
    
    def get_entity_embedding(self,):
        entity_embedding_path = self.cfg.entity_embedding_path
        os.makedirs(os.path.dirname(entity_embedding_path), exist_ok=True)
        entity_embedding_idx_path = self.cfg.entity_embedding_idx_path
        os.makedirs(os.path.dirname(entity_embedding_idx_path), exist_ok=True)
        
        if not (os.path.exists(entity_embedding_path) and os.path.exists(entity_embedding_idx_path)):
            entity_idxs,entity_embeds,data_type_idx,view2entity = self.embed_entities()

        if not os.path.exists(entity_embedding_path):
            with open(entity_embedding_path, 'wb') as handle:
                pickle.dump(entity_embeds, handle, protocol=4)
        
        if not os.path.exists(entity_embedding_idx_path):
            with open(entity_embedding_idx_path, 'wb') as handle:
                pickle.dump(entity_idxs, handle, protocol=4)

        return entity_embeds, entity_idxs, data_type_idx, view2entity
        
    def embed_entities(self, ):
        data_type_idx = dict()
        view2entity = dict()
        local_view_embeds = list()
        global_view_embeds = list()
        
        num = 0
        for data_type in ['table', 'passage']:
            view2entity[data_type] = dict()
            view_idx= 0
            entity_idx = 0
            start_idx = num
            for local_ids,global_ids in zip(self.local_views[data_type], self.global_views[data_type]):
                entity_ids = [global_ids] + local_ids
                view_num = len(entity_ids)
                for i in range(view_num):
                    if i == 0:
                        global_view_embeds.append((entity_ids[i],num))
                    else:
                        local_view_embeds.append((entity_ids[i],num))
                    
                    view2entity[data_type][view_idx] = entity_idx

                    num += 1
                    view_idx += 1
                entity_idx += 1
            end_idx = num
            data_type_idx[data_type] = [start_idx, end_idx]

        entity_idxs = list()
        entity_embeds = list()
        datasets = list()
        
        with torch.no_grad():
            dataloader_list = list()
            if len(global_view_embeds) > 0:
                global_view_dataset = EntityDataset(global_view_embeds,view_type="global")
                global_view_dataloader = DataLoader(global_view_dataset, batch_size=self.cfg.global_batch_size)
                dataloader_list.append(global_view_dataloader)
            
            if len(local_view_embeds) > 0:
                local_view_dataset = EntityDataset(local_view_embeds,view_type="local")
                local_view_dataloader = DataLoader(local_view_dataset, batch_size=self.cfg.local_batch_size)
                dataloader_list.append(local_view_dataloader)
            
            for infer_dataloader in dataloader_list:
                for batch in tqdm(infer_dataloader):
                    entity_ids,entity_idx = batch
                    entity_ids = entity_ids.cuda()
                    entity_emd = self.embedder(entity_ids=entity_ids)
                    entity_emd = entity_emd.detach().cpu()
                    
                    entity_idxs.append(entity_idx)
                    entity_embeds.append(entity_emd)
                    
        entity_embeds = torch.cat(entity_embeds, dim=0).numpy()
        entity_idxs = torch.cat(entity_idxs, dim=0).numpy()
        
        return entity_idxs, entity_embeds, data_type_idx, view2entity
    
    def load_embedder(self, cfg):
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