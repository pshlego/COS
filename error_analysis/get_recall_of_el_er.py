import json
from tqdm import tqdm
from pymongo import MongoClient
username = "root"
password = "1234"
client = MongoClient("mongodb://localhost:27017/", username=username, password=password)
db = client["mydatabase"]  # 데이터베이스 선택
print("MongoDB Connected")
dev_collection = db["ott_dev_q_to_tables_with_bm25neg"]
total_dev = dev_collection.count_documents({})
print(f"Loading {total_dev} instances...")
dev_instances = [doc for doc in tqdm(dev_collection.find(), total=total_dev)]
print("finish loading dev set")

graph_collection = db["table_chunks_to_passages_cos_table_passage"]
total_graph = graph_collection.count_documents({})
print(f"Loading {total_dev} instances...")
entity_linking_results = {graph['table_chunk_id']: graph for graph in tqdm(graph_collection.find(), total=total_graph)}
print("finish loading dev set")

at_least_one_tag = True

with open('/home/shpark/COS/error_analysis/expanded_query_retrieval_result.json', 'r') as file:
    expanded_query_retrieval_results = json.load(file)

for topk in [1,2,3,4,5]:
    entity_linking_pasg_title_list = []
    expanded_query_retrieval_pasg_title_list = []
    for qid, dev_instance in enumerate(dev_instances):
        
        for positive_ctx in dev_instance['positive_ctxs']:
            chunk_id = positive_ctx['chunk_id']
            target_pasg_titles = positive_ctx['target_pasg_titles']
            passage_list = expanded_query_retrieval_results[qid]['retrieved_result'][chunk_id]["passage_list"]
            
            expanded_query_retrieval_passages = []
            for passage in passage_list:
                expanded_query_retrieval_passages.extend(passage)
            
            entity_linking_passages = []
            for entity_linking_result in entity_linking_results[chunk_id]['results']:
                entity_linking_passages.extend(entity_linking_result['retrieved'][:topk])
            
            # expanded_query_retrieval_exist = False
            if at_least_one_tag:
                expanded_query_retrieval_passage_exist = False
                entity_linking_passage_exist = False
                for pasg_title in target_pasg_titles:
                    if pasg_title in expanded_query_retrieval_passages:
                        expanded_query_retrieval_passage_exist = True
                    
                    if pasg_title in entity_linking_passages:
                        entity_linking_passage_exist = True
                        
                if expanded_query_retrieval_passage_exist:
                    expanded_query_retrieval_pasg_title_list.append(1)
                else:
                    expanded_query_retrieval_pasg_title_list.append(0)
                
                if entity_linking_passage_exist:
                    entity_linking_pasg_title_list.append(1)
                else:
                    entity_linking_pasg_title_list.append(0)
                
            else:
                for pasg_title in target_pasg_titles:
                    if pasg_title in expanded_query_retrieval_passages:
                        expanded_query_retrieval_pasg_title_list.append(1)
                    else:
                        expanded_query_retrieval_pasg_title_list.append(0)
                    
                    if pasg_title in entity_linking_passages:
                        entity_linking_pasg_title_list.append(1)
                    else:
                        entity_linking_pasg_title_list.append(0)

    # print the recall of the whole passage retrieval
    if topk == 1:
        print('expanded_query_retrieval')
        print(sum(expanded_query_retrieval_pasg_title_list)/len(expanded_query_retrieval_pasg_title_list))
        print(len(expanded_query_retrieval_pasg_title_list))

    print('entity_linking')
    print(f"topk: {topk}")
    print(sum(entity_linking_pasg_title_list)/len(entity_linking_pasg_title_list))
    print(len(entity_linking_pasg_title_list))
    