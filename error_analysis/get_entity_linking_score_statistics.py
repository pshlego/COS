import json
from tqdm import tqdm
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
username = "root"
password = "1234"
client = MongoClient("mongodb://localhost:27017/", username=username, password=password)
db = client["mydatabase"]  # 데이터베이스 선택
print("MongoDB Connected")
# dev_collection = db["ott_dev_q_to_tables_with_bm25neg"]
# total_dev = dev_collection.count_documents({})
# print(f"Loading {total_dev} instances...")
# dev_instances = [doc for doc in tqdm(dev_collection.find(), total=total_dev)]
# print("finish loading dev set")

# ott_wiki_passages = db["ott_wiki_passages"]
# total_passage = ott_wiki_passages.count_documents({})
# print(f"Loading {total_dev} instances...")
# passages = {passage['chunk_id']: passage for passage in tqdm(ott_wiki_passages.find(), total=total_passage)}
# print("finish loading dev set")

graph_collection = db["table_chunks_to_passages_cos_table_passage"]
total_graph = graph_collection.count_documents({})
print(f"Loading {total_graph} instances...")
entity_linking_results = [graph for graph in tqdm(graph_collection.find(), total=total_graph)]
print("finish loading dev set")
score_list = []
for entity_linking_result in entity_linking_results:
    for result in entity_linking_result['results']:
        score_list.extend(result['scores'])
score_list = np.array(score_list)

# 예시 Entity Linking Score 리스트 생성 (정규 분포를 따르는 랜덤 데이터 사용)
scores = score_list

def plot_distribution(scores):
    plt.hist(scores, bins=30, alpha=0.7, color='blue')
    plt.title('Entity Linking Score Distribution')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.savefig('/home/shpark/COS/error_analysis/entity_linking_score_distribution.png')
    plt.close()

def calculate_mean_median(scores):
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    return mean_score, median_score

def calculate_quantiles(scores, quantiles):
    df_scores = pd.DataFrame(scores, columns=['Scores'])
    thresholds = {}
    for quantile in quantiles:
        threshold = df_scores['Scores'].quantile(quantile)
        thresholds[f'{int(quantile*100)}%'] = threshold
    return thresholds

# 분포 그래프 표시
plot_distribution(scores)

# 평균과 중앙값 계산
mean_score, median_score = calculate_mean_median(scores)
print(f'Mean: {mean_score}, Median: {median_score}')

# Quantile 값 계산
quantiles = [q/10 for q in range(1, 10)]
thresholds = calculate_quantiles(scores, quantiles)
print(thresholds)