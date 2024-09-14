import hydra
from omegaconf import DictConfig
import logging
import json
import os
from typing import List
from dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from dpr.models import init_biencoder_components
from tqdm import tqdm
from utils.utils import check_across_row, locate_row, get_row_indices

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

@hydra.main(config_path = "conf", config_name = "mention_detector")
def main(cfg: DictConfig):
    # Path to the output directory
    output_dir_path = '/mnt/sdf/shpark/MVD_OTT'
    output_file_name = "cos_dev_predictions.json"
    
    # Path to the train/dev/test table ids
    # train_dev_test_table_ids_path = '/mnt/sdf/shpark/train_dev_test_table_ids.json'
    
    # Corpus paths
    ## Table
    ott_table_chunks_path = '/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json'
    cos_format_mentions_path = '/mnt/sdf/shpark/mnt_sdc/shpark/graph/graph/for_test/all_table_chunks_span_prediction.json'
    
    ## Passage
    ott_wiki_passages_path = '/mnt/sdf/shpark/mnt_sdc/shpark/cos/cos/knowledge/ott_wiki_passages.json'
    
    print("[1. Reading files...]")
    
    # print("  1.1) Reading train-dev test table ids")
    # with open(train_dev_test_table_ids_path) as f:
    #     train_dev_test_table_ids = json.load(f)
    
    print("  1.2) Reading OTT table chunks")
    with open(ott_table_chunks_path) as f:
        ott_table_chunks_original = json.load(f)
    
    # print("  1.3) Reading COS-found mentions")
    # with open(cos_format_mentions_path) as f:
    #     cos_format_mention_objects = json.load(f)
    
    # print("  1.4) Reading original passages")
    # with open(ott_wiki_passages_path) as f:
    #     ott_wiki_passages = json.load(f)
    
    print("[1. Reading files - Complete]")
    
    
    target_split_types = [
        # 'train', 
        'dev', 
        # 'test'
        ]
    
    print()
    print("[2. Processing...]")
    for target_split_type in target_split_types:
        
        print(f"Processing {target_split_type} split")
        # target_table_ids = train_dev_test_table_ids[target_split_type]
        
        # output_path = os.path.join(output_dir_path, output_file_name)

        # Get the linking data for the target split
        target_table_id_to_table_info = {}
        
        # Get raw table text
        print("  2.1. Getting original table chunks")
        for ott_table_chunk in tqdm(ott_table_chunks_original):
            chunk_id = ott_table_chunk['chunk_id']
            table_id = '_'.join(chunk_id.split('_')[:-1])
            chunk_num = int(chunk_id.split('_')[-1])
            
            # if table_id not in target_table_ids:
            #     continue
            
            if table_id not in target_table_id_to_table_info:
                target_table_id_to_table_info[table_id] = {}
                target_table_id_to_table_info[table_id]['chunk_num_to_chunk'] = {}

            target_table_id_to_table_info[table_id]['chunk_num_to_chunk'][chunk_num] = ott_table_chunk
        
        # table id to global row id to chunk id and local row id
        table_id_to_global_row_id_to_chunk_id_and_local_row_id = {}
        for table_id, table_info in target_table_id_to_table_info.items():
            chunk_num_to_chunk = table_info['chunk_num_to_chunk']
            if table_id not in table_id_to_global_row_id_to_chunk_id_and_local_row_id:
                table_id_to_global_row_id_to_chunk_id_and_local_row_id[table_id] = {}
            global_row_id = 0
            for chunk_num, chunk in chunk_num_to_chunk.items():
                row_text_list = chunk['text'].split('\n')[1:]
                row_text_list = [row_text for row_text in row_text_list if row_text != '']
                for row_id, _ in enumerate(row_text_list):
                    chunk_info = target_table_id_to_table_info[table_id]['chunk_num_to_chunk'][chunk_num]
                    table_id_to_global_row_id_to_chunk_id_and_local_row_id[table_id][global_row_id] = {'chunk_id': chunk_info['chunk_id'], 'local_row_id': row_id}
                    global_row_id += 1
        
        # dump table_id_to_global_row_id_to_chunk_id_and_local_row_id
        with open('/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Evaluation_Dataset/table_to_chunk.json', 'w') as f:
            json.dump(table_id_to_global_row_id_to_chunk_id_and_local_row_id, f, indent=4)
        
        # print("  2.2. Getting COS' mentions")
        # data = read_jsonl('/mnt/sde/OTT-QAMountSpace/OTTeR/Embeddings/indexed_embeddings/dev_output_k100_table_corpus_blink.jsonl')
        # for datum in data:
        #     star_graph_info_list = datum['top_100']
        #     for star_graph_info in star_graph_info_list:
        #         table_id = star_graph_info['table_id']
        #         global_row_id = star_graph_info['row_id']
        #         row_text = ', '.join(star_graph_info['table'][1][0])
        #         chunk_num, local_row_id = table_id_to_global_row_id_to_chunk_id_and_local_row_id[table_id][global_row_id]
        #         corresponding_row_text = target_table_id_to_table_info[table_id]['chunk_num_to_chunk'][chunk_num]['text'].split('\n')[1+local_row_id]
        #         if table_id not in target_table_id_to_table_info:
        #             continue
        #         else:
        #             not_in = True
        #             # mapping row_id into chunk_num and row_id in the chunk
        #             for chunk_num, chunk in target_table_id_to_table_info[table_id]['chunk_num_to_chunk'].items():
        #                 if row_text in chunk['text']:
        #                     not_in = False
        #                     break
                        

                    
                
            
        # # Get COS' mentions
        # for cos_format_mention_object in tqdm(cos_format_mention_objects):
        #     chunk_id = cos_format_mention_object['chunk_id']
        #     table_id = '_'.join(chunk_id.split('_')[:-1])
        #     chunk_num = int(chunk_id.split('_')[-1])
            
        #     if table_id not in target_table_ids:
        #         continue
            
        #     if 'chunk_num_to_mentions' not in target_table_id_to_table_info[table_id]:
        #         target_table_id_to_table_info[table_id]['chunk_num_to_mentions'] = {}
            
        #     target_table_id_to_table_info[table_id]['chunk_num_to_mentions'][chunk_num] = cos_format_mention_object["grounding"]
        # # Create a dictionary for the passage titles
        # wiki_passage_title_to_ott_id = {}
        # print("  2.3. Conducting passage title to id map")
        # for ott_id, wiki_passage in enumerate(tqdm(ott_wiki_passages)):
        #     wiki_passage_title = wiki_passage['title']
        #     wiki_passage_title_to_ott_id[wiki_passage_title] = ott_id

        # gold_data_constructor = MentionAccuracyCalculator(cfg, target_table_id_to_table_info, wiki_passage_title_to_ott_id)
        # gold_data_constructor.generate(output_path = output_path)
















class MentionAccuracyCalculator:
    
    def __init__(self, cfg, target_table_id_to_table_info, wiki_passage_title_to_ott_id):
        self.cfg = cfg
        self.target_table_id_to_table_info = target_table_id_to_table_info
        
        self.wiki_passage_title_to_ott_id = wiki_passage_title_to_ott_id
        

    def setup_tokenizer(self, ):
        cfg = setup_cfg_gpu(self.cfg)
        cfg.n_gpu = 1
        
        cfg.encoder.encoder_model_type = 'hf_cos'
        sequence_length = 512
        cfg.encoder.sequence_length = sequence_length
        cfg.encoder.pretrained_file=None
        cfg.encoder.pretrained_model_cfg = 'bert-base-uncased'
        
        tensorizer, _, _ = init_biencoder_components(
            cfg.encoder.encoder_model_type, cfg, inference_only=True
        )
        
        return tensorizer.tokenizer
        
        
    def generate(self, output_path = "", max_length = 128):
        with open(output_path, 'w') as file:
            
            invalid_passage_cnt = 0
            
            # Setup tokenizer
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.cfg.device_id}"
            tokenizer = self.setup_tokenizer()
            
            for _, table_info in tqdm(self.target_table_id_to_table_info.items()):
                
                # 1. Aggregate raw table text from chunk texts
                raw_table_text = ''
                max_chunk_num = len(table_info['chunk_num_to_chunk'])
                
                accumulated_tokens_num_per_chunknum = [0]
                for chunk_num in range(max_chunk_num):
                    chunk = table_info['chunk_num_to_chunk'][chunk_num]
                    
                    if chunk_num == 0:
                        raw_table_text += chunk['title'] + ' [SEP] '
                        raw_table_text += chunk['text']
                        accumulated_tokens_num_per_chunknum.append(len(tokenizer.encode(raw_table_text)) - 1)
                    else:
                        raw_table_text += '\n'.join(chunk['text'].split('\n')[1:])
                    
                tokenized_text = tokenizer.encode(raw_table_text)
                row_indices = get_row_indices(raw_table_text, tokenizer)
                title_column_name = raw_table_text.split('\n')[0]
                
                
                                
                # 2. Iterate over each row to check hyperlinks
                # for row_id, hyperlinks_per_row in enumerate(hyperlinks):
                accumulated_tokens_num = 0
                for chunk_num in range(max_chunk_num):
                        
                    for mention_obj in chunk_num_to_mentions[chunk_num]:
                        start_idx = mention_obj["full_word_start"]
                        end_idx   = mention_obj["full_word_end"]
                        
                    if chunk_num != 0:
                        
                        
                        start_idx = chunk_num_to_mentions[chunk_num]
                    
                    
                    if chunk_num == 0:
                        accumulated_tokens_num += table_info["chunk_num_to_chunk"][chunk_num]
                        
                        
                                           
                    # # 1. Find the start and end indices of the row
                    # row_start_idx = row_indices[row_id]
                    # row_end_idx = row_indices[row_id + 1] - 1 
                     
                    # # 2. Iterate over each hyperlink in a row 
                    # for mention, linked_passage_title in hyperlinks_per_row:
                        
                    #     if linked_passage_title in self.wiki_passage_title_to_ott_id.keys():
                    #         ott_id = int(self.wiki_passage_title_to_ott_id[linked_passage_title])
                    #     else:
                    #         invalid_passage_cnt += 1
                    #         continue
                        
                    #     start_idx, end_idx = find_span_indices(raw_table_text, mention, tokenizer, row_start_idx, row_end_idx)
                    #     same_row_left = tokenizer.decode(tokenized_text[row_indices[row_id]:start_idx]).strip()
                    #     mention_context_left = title_column_name + ' [SEP] ' + same_row_left
                        
                    #     if len(mention_context_left.split(" ")) > max_length:
                    #         mention_context_left = ' '.join(mention_context_left.split(" ")[max(0, -max_length):])
                            
                    #     mention_context_right = tokenizer.decode(tokenized_text[end_idx:row_indices[row_id+1]]).strip()
                        
                    #     if len(mention_context_right.split(" ")) > max_length:
                    #         mention_context_right = ' '.join(mention_context_right.split(" ")[:min(len(mention_context_right.split(" ")), max_length)])
                        
                    #     mention_context_right = mention_context_right.replace(' [PAD]', '')
                        
                    #     json_line = json.dumps({'mention': mention, 'label_id': ott_id, 'context_left': mention_context_left, 'context_right': mention_context_right})
                    #     file.write(json_line + '\n')        
            
        print("Number of invalid passages: ", invalid_passage_cnt)

    
    






















def find_span_indices(text, span, tokenizer, row_start_idx, row_end_idx):
    
    # Tokenize the text and the span
    tokens = tokenizer.encode(text)
    span_tokens = tokenizer.encode(span, add_special_tokens=False)

    # Initialize indices
    start_index = None
    end_index = None

    # Search for the start token of the span in the tokenized text
    for i, token in enumerate(tokens):
        # If the current token matches the first token of the span
        # and the subsequent tokens in the text match the rest of the span tokens
        if tokens[i:i+len(span_tokens)] == span_tokens and (i >= row_start_idx and i <= row_end_idx):
            start_index = i
            end_index = i + len(span_tokens)
            break  # Exit the loop once the span is found

    if start_index is not None and end_index is not None:
        return start_index, end_index
    else:
        return None, None



















logger = logging.getLogger()
setup_logger(logger)

if __name__ == "__main__":
    main()