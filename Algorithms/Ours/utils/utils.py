from torch.utils.data import Dataset
import sys
import os
import torch
from tqdm import tqdm
from typing import List,Optional, Union, Optional
import json

class MentionInfo(object):
    """Constructs train/dev InputFeatures."""
    def __init__(
            self,
            mention_ids:Optional[List[int]]=None,
            datum_mention:dict=None,
            mention_tokens:List[str]=None,
            data_type:str='all',
            node_id:Optional[int]=None,
            row_id:Optional[int]=None,
            mention_id:Optional[int]=None,
            original_cell:Optional[str]=None,
            chunk_id: Optional[str]=None,
            question: Optional[str]=None
            ):
        self.chunk_id = chunk_id
        self.original_cell = original_cell
        self.datum_mention = datum_mention
        self.node_id = node_id
        self.row_id = row_id
        self.mention_ids = mention_ids
        self.mention_tokens = mention_tokens
        self.data_type = data_type
        self.mention_id = mention_id
        self.question = question

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
        indices.append(len(tokens) + 1)
    assert tokens == original_input
    return indices

def process_mention(tokenizer, mention_dict, max_seq_length):
    # process mention format: [CLS] context_left [STR] mention [END] context_right [SEP]
    mention = mention_dict['mention']
    mention_left = mention_dict['context_left']
    mention_right = mention_dict['context_right']
    mention_tokens = ['[unused0]'] + tokenizer.tokenize(mention) + ['[unused1]']
    context_left = tokenizer.tokenize(mention_left)
    context_right = tokenizer.tokenize(mention_right)

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    mention_tokens = (
        context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )
    if len(mention_tokens) > max_seq_length - 2:
        mention_tokens = mention_tokens[:(max_seq_length - 2)]
    mention_tokens = ['[CLS]'] + mention_tokens + ['[SEP]']
    mention_ids = tokenizer.convert_tokens_to_ids(mention_tokens)
    mention_padding = [0] * (max_seq_length - len(mention_ids))
    mention_ids += mention_padding
    return mention_ids,mention_tokens

def prepare_datasource(cfg, mongodb, source_type = 'table'):
    collection_list = mongodb.list_collection_names()
    collection_name = cfg.data_sources[source_type]['collection_name']
    
    if collection_name in collection_list:
        collection = mongodb[collection_name]
        total_num = collection.count_documents({})
        return [doc for doc in tqdm(collection.find(), total = total_num)]
    else:
        data = json.load(open(cfg.data_sources[source_type]['path'], 'r'))
        collection = mongodb[collection_name]
        collection.insert_many(data)
        return data

def read_jsonl(file_path, key = None, num = None):
    if key is None:
        data = []
    else:
        data = {}

    with open(file_path, 'r') as file:
        if num is None:
            loader = tqdm(file)
        else:
            loader = tqdm(file, total = num)

        for line in loader:
            if key is None:
                data.append(json.loads(line))
            else:
                json_data = json.loads(line)
                data[json_data[key]] = json_data

    return data

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__