import json
from tqdm import tqdm
from pymongo import MongoClient
username = "root"
password = "1234"
client = MongoClient("mongodb://localhost:27017/", username=username, password=password)
db = client["mydatabase"]  # 데이터베이스 선택
print("MongoDB Connected")
ott_wiki_passages = db["ott_wiki_passages"]
total_dev = ott_wiki_passages.count_documents({})
print(f"Loading {total_dev} instances...")
ott_wiki_passages_dict = {doc['chunk_id']: doc for doc in tqdm(ott_wiki_passages.find(), total=total_dev)}
print("finish loading passages")
with open('/home/shpark/mnt_sdc/shpark/cos/cos/models/ott_dev_core_reader_hop1keep200_shard0_of_1_check_both.json', 'r') as file:
    cos_results = json.load(file)

answer_source_dict_list = []
num = 0
for cos_result in cos_results:
    # module_dict = {'entity_linking':0, 'expanded_query':0, 'table': 0}
    module_dict = {'entity_linking':0, 'expanded_query':0}
    has_answer = False
    gold_passage_set = {gold_passage for positive_ctx in cos_result['positive_ctxs'] for gold_passage in positive_ctx['target_pasg_titles'] if gold_passage is not None}
    if len(list(gold_passage_set)) > 0:
        for i, ctx in enumerate(cos_result['ctxs']):
            if ctx['has_answer']:
                # if 'path score' not in ctx:
                #     module_dict['table'] = 1
                #     has_answer = True
                # else:
                has_gold_passage = False
                for gold_passage in gold_passage_set:
                    try:
                        if ott_wiki_passages_dict[gold_passage]['text'] in ctx['text']:
                            has_gold_passage = True
                            break
                    except:
                        print(gold_passage)
                
                if has_gold_passage:
                    hop2_source = ctx['hop2 source']
                    if ctx['both'] == 1:
                        hop2_source = 'both'
                    if hop2_source == 'r2':
                        module_dict['expanded_query'] = 1
                        has_answer = True
                    elif hop2_source == 'pl':
                        module_dict['entity_linking'] = 1
                        has_answer = True
                    elif hop2_source == 'both':
                        module_dict['expanded_query'] = 1
                        module_dict['entity_linking'] = 1
                        has_answer = True
                        
        if has_answer:
            answer_source_dict_list.append(module_dict)
        num+=1
# statistics of answer source
# for module_dict in answer_source_dict_list:
#     print(module_dict)
print(f"Total Answer at passage: {num}")
print(f"Total: {len(answer_source_dict_list)}")
# print(f"Table only: {sum([module_dict['table'] and not module_dict['expanded_query'] and not module_dict['entity_linking'] for module_dict in answer_source_dict_list])}")
print(f"Entity Linking only: {sum([module_dict['entity_linking'] and not module_dict['expanded_query'] for module_dict in answer_source_dict_list])}")
print(f"Expanded Query only: {sum([module_dict['expanded_query'] and not module_dict['entity_linking'] for module_dict in answer_source_dict_list])}")
# print(f"Table and Entity Linking: {sum([module_dict['table'] and module_dict['entity_linking'] and not module_dict['expanded_query'] for module_dict in answer_source_dict_list])}")
# print(f"Table and Expanded Query: {sum([module_dict['table'] and module_dict['expanded_query'] and not module_dict['entity_linking'] for module_dict in answer_source_dict_list])}")
print(f"Entity Linking and Expanded Query: {sum([module_dict['entity_linking'] and module_dict['expanded_query'] for module_dict in answer_source_dict_list])}")
# print(f"Table, Entity Linking and Expanded Query: {sum([module_dict['table'] and module_dict['entity_linking'] and module_dict['expanded_query'] for module_dict in answer_source_dict_list])}")
           