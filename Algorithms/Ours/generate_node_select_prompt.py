# Copyright 2024 The Chain-of-Table authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import copy
import re
from pymongo import MongoClient
from tqdm import tqdm
import numpy as np
from utils.helper import NoIndent, MyEncoder


def twoD_list_transpose(arr, keep_num_rows=3):
    arr = arr[: keep_num_rows + 1] if keep_num_rows + 1 <= len(arr) else arr
    return [[arr[i][j] for i in range(len(arr))] for j in range(len(arr[0]))]


def select_column_build_prompt(table_title, column_w_value_list, linked_passages_list, question):
    dic = {
        "table_caption": table_title,
        "table_column_priority": [NoIndent(column_w_value) for column_w_value in column_w_value_list],
    }
    linear_table_dic = json.dumps(
        dic, cls=MyEncoder, ensure_ascii=False, sort_keys=False, indent=2
    )
    dic = {
        "passage_titles": [linked_passages[1] for linked_passages in linked_passages_list],
        "linked_passage_context": [NoIndent(linked_passages[1:]) for linked_passages in linked_passages_list],
    }
    linear_passage_dic = json.dumps(
        dic, cls=MyEncoder, ensure_ascii=False, sort_keys=False, indent=2
    )
    prompt = "/*\ntable segment = " + linear_table_dic + "\n*/\n"
    prompt += "/*\nlinked passages = " + linear_passage_dic + "\n*/\n"
    prompt += "question : " + question + ".\n"
    prompt += "explain : "
    prompt += "The answer is : f_passage([])"
    return prompt

def main():
    
    username = "root"
    password = "1234"
    client = MongoClient("mongodb://localhost:27017/", username=username, password=password)
    db = client["mydatabase"]  # 데이터베이스 선택
    print("MongoDB Connected")
    table_chunks_to_passages_gt_data_collection = db["table_chunks_to_passages_gt_data"]
    total_dev = table_chunks_to_passages_gt_data_collection.count_documents({})
    print("1. Loading data graph...")
    table_chunk_id_to_linked_passages = {doc['table_chunk_id']: doc for doc in tqdm(table_chunks_to_passages_gt_data_collection.find(), total=total_dev)}
    print("finish loading dev set")

    with open("/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_train_q_to_tables_with_bm25neg.json", 'r') as file:
        data = json.load(file)
    # 4. Load passages
    print("4. Loading passages...")
    passage_key_to_content = {}
    passage_contents = json.load(open("/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json"))
    PASSAGES_NUM = len(passage_contents)
    for passage_content in tqdm(passage_contents):
        passage_key_to_content[passage_content['title']] = passage_content
    print("4. Loaded " + str(PASSAGES_NUM) + " passages!", end = "\n\n")
    
    prompt_total = """Using f_passage() api to select relevant passages in the given row and linked passages that support or oppose the question. \n\n"""
    for datum in data[:8]:
        positive_ctx = datum['positive_ctxs'][0]
        table_title = positive_ctx['title'].lower()
        column_list = positive_ctx['text'].split('\n')[0].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ').split(' | ')
        column_list = [col.lower() for col in column_list]
        row_id = positive_ctx['rows'].index(positive_ctx['answer_node'][0][1][0])
        gold_col_id = positive_ctx['answer_node'][0][1][1]
        row_value_list = positive_ctx['text'].split('\n')[1+row_id].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ').split(' | ')
        row_value_list = [row_value.lower() for row_value in row_value_list]
        column_w_value_list = []
        for col_id, col in enumerate(column_list):
            #if gold_col_id == col_id:
            column_w_value_list.append([col, row_value_list[col_id]])
            
        question = datum['question']
        linked_passages_list = []
        entity_linking_results = table_chunk_id_to_linked_passages[positive_ctx['chunk_id']]['results']
        for entity_linking_result in entity_linking_results:
            if int(row_id) == int(entity_linking_result['row']):
                mention = entity_linking_result['original_cell']
                if entity_linking_result['retrieved'][0] not in passage_key_to_content:
                    continue
                passage_content = passage_key_to_content[entity_linking_result['retrieved'][0]]
                passage_title = passage_content['title']
                passage_text = passage_content['text']
                linked_passages_list.append([mention, passage_title, f"{passage_text}"])

        prompt = select_column_build_prompt(table_title, column_w_value_list, linked_passages_list, question)
        prompt_total += prompt + '\n\n'
    
    with open("/home/shpark/OTT_QA_Workspace/select_passages_prompt.txt", 'w') as file:
        file.write(prompt_total)
        

if __name__ == "__main__":
    main()