import torch
from utils import EntityDataset
from torch.utils.data import Sampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import os
import gc
import pickle
import faiss
import numpy as np

class IndexBuilder:
    def __init__(self, cfg, table_views, passage_views, embedder):
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
        output_dir = self.cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)
        entity_embedding_path = os.path.join(output_dir,"entity_embedding_data_obj.pb")
        entity_embedding_idx_path = os.path.join(output_dir,"entity_embedding_idx_data_obj.pb")
        
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
            if len(local_view_embeds) > 0:
                datasets.append(EntityDataset(local_view_embeds,view_type="local"))
            if len(global_view_embeds) > 0:
                datasets.append(EntityDataset(global_view_embeds,view_type="global"))
                
            for dataset in datasets:
                infer_dataloader = DataLoader(dataset, batch_size=self.cfg.batch_size)
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