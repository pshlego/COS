import os
import sys
import json
from tqdm import tqdm
from pymongo import MongoClient
from ColBERT.colbert import Searcher
from ColBERT.colbert.infra import ColBERTConfig

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def get_topk_retrieved_docs(retriever, query, id_to_key, topk=10000):
    doc_ids = []
    scores = []
    retrieved_info = retriever.search(query, topk)
    retrieved_id_list = retrieved_info[0]
    retrieved_score_list = retrieved_info[2]
    
    retrieved_results = []
    for doc_id, score in zip(retrieved_id_list, retrieved_score_list):
        if str(doc_id) not in id_to_key:
            continue
        doc_key = id_to_key[str(doc_id)]
        retrieved_results.append((doc_key, score))

    return retrieved_results

def main():
    username = "root"
    password = "1234"
    client = MongoClient("mongodb://localhost:27017/", username=username, password=password)
    db = client["mydatabase"]  # 데이터베이스 선택
    
    # passage_collection = db["ott_wiki_passages"]
    # total_passage = passage_collection.count_documents({})
    # print(f"Loading {total_passage} instances...")
    # passage_key_to_content = {doc['chunk_id']:doc for doc in tqdm(passage_collection.find(), total=total_passage)}
    # print("finish loading passage")
    
    table_collection = db["ott_table"]
    total_table = table_collection.count_documents({})
    print(f"Loading {total_table} instances...")
    table_list = [doc for doc in tqdm(table_collection.find(), total=total_table)]
    table_key_to_content = {doc['chunk_id']:doc for doc in tqdm(table_list)}
    print("finish loading table")
    
    star_collection = db["preprocess_table_graph_cos_apply_topk_star"]
    total_star = star_collection.count_documents({})
    print(f"Loading {total_star} instances...")
    star_key_to_content = {doc['chunk_id']:doc for doc in tqdm(star_collection.find(), total=total_star)}
    print("finish loading star")
    
    edge_collection = db["preprocess_table_graph_cos_apply_topk_edge_1"]
    total_edge = edge_collection.count_documents({})
    print(f"Loading {total_edge} instances...")
    edge_key_to_content = {doc['chunk_id']:doc for doc in tqdm(edge_collection.find(), total=total_edge)}
    print("finish loading table")
    
    qa_dataset_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_dev_q_to_tables_with_bm25neg.json"
    index_root_path = "/mnt/sdc/shpark/OTT-QAMountSpace/Embeddings"
    
    star_ids_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/star/index_to_chunk_id_star_topk_1.json"
    edge_ids_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/edge/top_1/index_to_chunk_id_edge_topk_1.json"
    table_ids_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/table_cos_version/index_to_chunk_id.json"
    passage_ids_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/passage_cos_version/index_to_chunk_id.json"
    
    collection_star_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/star/star_topk_1.tsv"
    collection_edge_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/edge/top_1/edge_topk_1.tsv"
    collection_table_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/table_cos_version/collection.tsv"
    collection_passage_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/passage_cos_version/collection.tsv"
    
    star_index_name = "top1_star_embeddings_v2_trained_1_epoch_bsize_512.nbits2"
    edge_index_name = "top1_edge_embeddings_v2_trained_1_epoch_bsize_512.nbits2"
    table_index_name = "table_segment_embeddings.nbits2"
    passage_index_name = "passage_embeddings.nbits2"
    
    star_checkpoint_path = "/mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/ColBERT/edge_trained_v2"
    edge_checkpoint_path = "/mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/ColBERT/edge_trained_v2"
    table_checkpoint_path = "/mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/ColBERT/original"
    passage_checkpoint_path = "/mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/ColBERT/original"
    
    print("Hello World")
    # 1. Load Retrievers
    print("1. Loading retrievers...")
    print("1.1. Loading id mappings...")
    id_to_star_key = json.load(open(star_ids_path))
    id_to_edge_key = json.load(open(edge_ids_path))
    id_to_table_key = json.load(open(table_ids_path))
    id_to_passage_key = json.load(open(passage_ids_path))
    print("1.1. Loaded id mappings!")
    
    print("1.2. Loading index...")
    print("1.2 (1/4). Loading star index...")
    disablePrint()
    colbert_star_retriever = Searcher(index=star_index_name, config=ColBERTConfig(), collection=collection_star_path, index_root=index_root_path, checkpoint=star_checkpoint_path)
    enablePrint()
    print("1.2 (1/4). Loaded star index complete!")
    print("1.2 (2/4). Loading edge index...")
    disablePrint()
    colbert_edge_retriever = Searcher(index=edge_index_name, config=ColBERTConfig(), collection=collection_edge_path, index_root=index_root_path, checkpoint=edge_checkpoint_path)
    enablePrint()
    print("1.2 (2/4). Loaded edge index complete!")
    print("1.2 (3/4). Loading table index...")
    disablePrint()
    colbert_table_retriever = Searcher(index=table_index_name, config=ColBERTConfig(), collection=collection_table_path, index_root=index_root_path, checkpoint=table_checkpoint_path)
    enablePrint()
    print("1.2 (3/4). Loaded table index complete!")
    print("1.2 (4/4). Loading passage index...")
    disablePrint()
    colbert_passage_retriever = Searcher(index=passage_index_name, config=ColBERTConfig(), collection=collection_passage_path, index_root=index_root_path, checkpoint=passage_checkpoint_path)
    enablePrint()
    print("1.2 (4/4). Loaded passage index complete!")
    print("1.2. Loaded index complete!")
    
    qa_dataset = json.load(open(qa_dataset_path))
    
    retrieved_results_list = []
    
    for qa_datum in tqdm(qa_dataset):
        question = qa_datum["question"]

        gold_table_segments = [
            f"{ctx['chunk_id']}_{ctx['rows'].index(node[1][0])}"
            for ctx in qa_datum['positive_ctxs'][:1]
            for node in ctx['answer_node'][:1]
            if isinstance(ctx, dict) and 'chunk_id' in ctx and 'answer_node' in ctx
        ]
        
        gold_passages = [
            ans[2].replace('/wiki/', '').replace('_', ' ')
            for ctx in qa_datum['positive_ctxs'][:1]
            for ans in ctx['answer_node'][:1]
            if isinstance(ctx, dict) and 'answer_node' in ctx and ans[3] == 'passage'
        ]
        
        star_retrieved_results = get_topk_retrieved_docs(colbert_star_retriever, question, id_to_star_key)
        
        star_node_list = []
        for star_key, score in star_retrieved_results:
            table_id = int(star_key.split('_')[0])
            table_chunk_id = table_list[table_id]['chunk_id']
            star_node_list.append(f"{table_chunk_id}_{star_key.split('_')[-1]}")
            star_key = star_key
            star_content = star_key_to_content[star_key]
            if 'mentions_in_row_info_dict' not in star_content:
                continue
            mentions_in_row_info_dict = star_content['mentions_in_row_info_dict']
            for mention_id, mention_info in mentions_in_row_info_dict.items():
                linked_passage_title = mention_info['mention_linked_entity_id_list'][0]
                star_node_list.append(linked_passage_title)
        
        edge_retrieved_results = get_topk_retrieved_docs(colbert_edge_retriever, question, id_to_edge_key)
        
        edge_node_list = []
        for edge_key, score in edge_retrieved_results:
            table_id = int(edge_key.split('_')[0])
            table_chunk_id = table_list[table_id]['chunk_id']
            edge_node_list.append(f"{table_chunk_id}_{edge_key.split('_')[-1]}")
            edge_content = edge_key_to_content[edge_key]

            if 'linked_entity_id' not in edge_content:
                continue

            edge_node_list.append(edge_content['linked_entity_id'])
        
        table_retrieved_results = get_topk_retrieved_docs(colbert_table_retriever, question, id_to_table_key)
        passage_retrieved_results = get_topk_retrieved_docs(colbert_passage_retriever, question, id_to_passage_key)
        
        table_chunk_id_list = [[f"{table_list[int(table_retrieved_result[0].split('_')[0])]['chunk_id']}_{table_retrieved_result[0].split('_')[-1]}", table_retrieved_result[1]] for table_retrieved_result in table_retrieved_results]
        passage_chunk_id_list = [[passage_retrieved_result[0], passage_retrieved_result[1]] for passage_retrieved_result in passage_retrieved_results]
        sorted_combined_retrieved_results = sorted(table_chunk_id_list + passage_chunk_id_list, key=lambda x: x[1], reverse=True)
        
        node_node_list = []
        for node_key, score in sorted_combined_retrieved_results:
            node_node_list.append(node_key)
        
        retrieved_results_list.append({
            "question": question,
            "star": star_node_list,
            "edge": edge_node_list,
            "node": node_node_list,
            "gold_table_segments": gold_table_segments,
            "gold_passages": gold_passages
        })
    
    star_recall_list = []
    edge_recall_list = []
    node_recall_list = []
    topk_list = [2, 5, 10, 20, 50, 100, 200, 500]

    for retrieved_results in retrieved_results_list:
        
        gold_table_segments = retrieved_results["gold_table_segments"]
        gold_passages = retrieved_results["gold_passages"]
        gold_nodes = set(gold_table_segments + gold_passages)
        
        star_node_list = list(set(retrieved_results["star"]))
        edge_node_list = list(set(retrieved_results["edge"]))
        node_node_list = list(set(retrieved_results["node"]))
        
        star_recall = []
        edge_recall = []
        node_recall = []
        
        for topk in topk_list:
            star_recall.append(len(set(star_node_list[:topk]) & gold_nodes) / len(gold_nodes))
            edge_recall.append(len(set(edge_node_list[:topk]) & gold_nodes) / len(gold_nodes))
            node_recall.append(len(set(node_node_list[:topk]) & gold_nodes) / len(gold_nodes))
    
        star_recall_list.append(star_recall)
        edge_recall_list.append(edge_recall)
        node_recall_list.append(node_recall)
    
    # Save the results as plots
    import matplotlib.pyplot as plt
    import numpy as np
    star_recall_list = np.array(star_recall_list)
    edge_recall_list = np.array(edge_recall_list)
    node_recall_list = np.array(node_recall_list)
    
    star_recall_mean = np.mean(star_recall_list, axis=0)
    edge_recall_mean = np.mean(edge_recall_list, axis=0)
    node_recall_mean = np.mean(node_recall_list, axis=0)
    
    star_recall_std = np.std(star_recall_list, axis=0)
    edge_recall_std = np.std(edge_recall_list, axis=0)
    node_recall_std = np.std(node_recall_list, axis=0)
    
    plt.errorbar(topk_list, star_recall_mean, yerr=star_recall_std, fmt='-o', label='Star')
    plt.errorbar(topk_list, edge_recall_mean, yerr=edge_recall_std, fmt='-o', label='Edge')
    plt.errorbar(topk_list, node_recall_mean, yerr=node_recall_std, fmt='-o', label='Node')
    
    plt.xlabel('Top-k')
    plt.ylabel('Recall')
    plt.title('Recall@k')
    plt.legend()
    #save the plot as a png file
    plt.savefig('/home/shpark/OTT_QA_Workspace/retrieval_accuracy_experiment.png')
    
        


if __name__ == "__main__":
    main()