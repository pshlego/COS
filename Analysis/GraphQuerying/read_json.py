import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
path = "/home/shpark/OTT_QA_Workspace/error_case/both_error_cases_reranking_last_not_in_data_graph_baai_rerank_full_layer_wo_table_retrieval_error.json"

with open(path, "r") as f:
    data = json.load(f)

count = 0
rank_dict = {}
qid_list = []
for qid, datum in data.items():
    qid_list.append(qid)
    positive_table_chunk = set('_'.join(positive_table_segment.split('_')[:-1]) for positive_table_segment in datum['positive_table_segments'])
    retrieved_table_chunk = set([node['chunk_id'] for node_id, node in datum['retrieved_graph'].items() if node['type'] == 'table segment'])
    if len(positive_table_chunk.intersection(retrieved_table_chunk)) == 0:
        count += 1
    else:
        sorted_retrieved_graph = sorted(datum['retrieved_graph'].items(), key=lambda x: x[1]['score'], reverse=True)
        node_list = []
        for node_id, node_info in sorted_retrieved_graph:
            if node_info['type'] == 'table segment':
                node_list.append(node_info['chunk_id'])
            else:
                node_list.append(node_id)

        rank = node_list.index(list(positive_table_chunk.intersection(retrieved_table_chunk))[0]) - len(datum['sorted_retrieved_graph'])
        if rank not in rank_dict:
            rank_dict[rank] = 1
        else:
            rank_dict[rank] += 1
    
    # #'List_of_Bomberman_video_games_8_2' in [node['chunk_id'] for node_id, node in datum['retrieved_graph'].items() if node['type'] == 'table segment']
    # print(datum['positive_ctxs'][0]['text'])
    
print(count)
print(dict(sorted(rank_dict.items())))
print(len(data))