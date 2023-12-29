from torch.utils.data import Dataset
import torch

class EntityDataset(Dataset):
    
    def __init__(self, entities,view_type="local"):
        self.len = len(entities)
        self.entities = entities
        self.view_type = view_type

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def free(self):
        self.inputs = None

    def __getitem__(self, index, max_ent_length=512):
        
        entity_ids = self.entities[index][0]
        # global-view
        if self.view_type != "local":
            entity_ids = [101] + entity_ids[1:-2][:max_ent_length-2] + [102]
            entity_ids += [0] * (max_ent_length-len(entity_ids))
        entity_ids = torch.LongTensor(entity_ids)
        entity_idx = self.entities[index][1]
        res = [entity_ids,entity_idx]
        return res

def check_across_row(start, end, row_indices):
    for i in range(len(row_indices)):
        if start < row_indices[i] and end > row_indices[i]:
            return row_indices[i]
    return False

def locate_row(start, end, row_indices):
    for i in range(len(row_indices)):
        if end <= row_indices[i]:
            return i
    return -1

def get_row_indices(question, tokenizer):
    original_input = tokenizer.tokenize(question)
    rows = question.split('\n')
    indices = []
    tokens = []
    for row in rows:
        tokens.extend(tokenizer.tokenize(row))
        indices.append(len(tokens)+1)
    assert tokens == original_input
    return indices