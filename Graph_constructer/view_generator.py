from transformers import BertTokenizer
import json
from tqdm import tqdm
import nltk
import os
from utils.utils import get_row_indices

class ViewGenerator:
    def __init__(self, cfg, mongodb):
        self.cfg = cfg
        self.mongodb = mongodb
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def generate(self, all_tables=None, all_passages=None):
        doc_list = []
        if all_tables is not None:
            output_name = self.cfg.table_view_path
            data = all_tables
            for chunk in tqdm(data):
                field = {}
                window = self.get_entity_windows(chunk, 'table')
                field = window
                field['node_id'] = len(doc_list)
                doc_list.append(field)
            json.dump(doc_list, open(output_name, 'w'), indent=4)
            collection_name = os.path.basename(output_name).split('.')[0]
            collection = self.mongodb[collection_name]
            collection.insert_many(doc_list)
        if all_passages is not None:
            output_name = self.cfg.passage_view_path
            data = all_passages
            for chunk in tqdm(data):
                field = {}
                window = self.get_entity_windows(chunk, 'passage')
                field = window
                field['node_id'] = len(doc_list)
                doc_list.append(field)
            json.dump(doc_list, open(output_name, 'w'), indent=4)
            collection_name = os.path.basename(output_name).split('.')[0]
            collection = self.mongodb[collection_name]
            collection.insert_many(doc_list)

    def get_entity_windows(self, chunk, data_type):
        CLS,ENT,SEP = '[CLS]','[SEP]','[SEP]'
        title = chunk['title'].strip()
        text = chunk['text'].strip()
        title_tokens = self.tokenizer.tokenize(title)
        text_tokens = self.tokenizer.tokenize(text) 
        if data_type == 'passage': 
            window = (title_tokens + [ENT] + text_tokens)[:self.cfg.max_global_view_len-2]
            window = [CLS] + window + [SEP]
            global_ids = self.tokenizer.convert_tokens_to_ids(window)
            local_ids = self.tokenize_split_description(title, text, data_type)
        else:
            table_row_indices = get_row_indices(title + ' [SEP] ' + text, self.tokenizer)
            row_start = table_row_indices[0]
            row_indicies = table_row_indices[1:]
            title_column_name_tokens = self.tokenizer.tokenize(title + ' [SEP] ' + text)[:row_start-1]
            
            if len(title_column_name_tokens + [ENT] + self.tokenizer.tokenize(title + ' [SEP] ' + text)[row_start-1:])> self.cfg.max_global_view_len-2:
                window = (title_column_name_tokens + [ENT] + self.tokenizer.tokenize(title + ' [SEP] ' + text)[row_start-1:])[:self.cfg.max_global_view_len-2]
            else:
                window = title_column_name_tokens + [ENT] + self.tokenizer.tokenize(title + ' [SEP] ' + text)[row_start-1:]
            window = [CLS] + window + [SEP]
            global_ids = self.tokenizer.convert_tokens_to_ids(window)
            if len(global_ids) < self.cfg.max_global_view_len:
                global_ids += [0] * (self.cfg.max_global_view_len - len(global_ids))
            local_ids = self.tokenize_split_description(title, text, data_type, row_start, table_row_indices)
            
        window = {}
        window['local_ids'] = local_ids
        window['global_ids'] = global_ids
        return window

    def tokenize_split_description(self, title, text, data_type, row_start=None, table_row_indices=None):
        ENTITY_TAG = '[SEP]'
        CLS_TAG = '[CLS]'
        SEP_TAG = '[SEP]'
        
        multi_sent = []
        pre_text = []
        if data_type == 'passage':
            title_tokens = self.tokenizer.tokenize(title)
            title_text = title_tokens + [ENTITY_TAG]
            for sent in nltk.sent_tokenize(text.replace(' .', '.')):
                text = self.tokenizer.tokenize(sent)
                pre_text += text
                if len(pre_text) <= 5:
                    continue
            
                whole_text = title_text + pre_text
                whole_text = [CLS_TAG] + whole_text[:self.cfg.max_local_view_len - 2] + [SEP_TAG]
                tokens = self.tokenizer.convert_tokens_to_ids(whole_text)
                pre_text = []
                #padding
                if len(tokens) < self.cfg.max_local_view_len:
                    tokens += [0] * (self.cfg.max_local_view_len - len(tokens))
                assert len(tokens) == self.cfg.max_local_view_len
                multi_sent.append(tokens)
        else:
            whole_tokens = self.tokenizer.tokenize(title + ' [SEP] ' + text)
            title_column_name_tokens = self.tokenizer.tokenize(title + ' [SEP] ' + text)[:row_start-1]
            for id in range(len(table_row_indices)):
                try:
                    single_row = whole_tokens[table_row_indices[id]-1:table_row_indices[id+1]-1]
                except:
                    single_row = whole_tokens[table_row_indices[id]-1:]
                pre_text += single_row
                if len(pre_text) <= 5:
                    continue
                whole_text = title_column_name_tokens + [SEP_TAG] + pre_text
                whole_text = [CLS_TAG] + whole_text[:self.cfg.max_local_view_len - 2] + [SEP_TAG]
                tokens = self.tokenizer.convert_tokens_to_ids(whole_text)
                pre_text = []
                if len(tokens) < self.cfg.max_local_view_len:
                    tokens += [0] * (self.cfg.max_local_view_len - len(tokens))
                assert len(tokens) == self.cfg.max_local_view_len
                multi_sent.append(tokens)
        return multi_sent
        