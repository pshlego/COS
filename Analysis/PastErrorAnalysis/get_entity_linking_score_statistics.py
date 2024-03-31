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
def min_max_normalize(scores):
    min_val = np.min(scores)
    max_val = np.max(scores)
    if min_val == max_val:
        return np.ones(len(scores))
    normalized = (scores - min_val) / (max_val - min_val)
    return normalized.squeeze().tolist()

graph_collection = db["table_chunks_to_passages_cos_table_passage"]
total_graph = graph_collection.count_documents({})
print(f"Loading {total_graph} instances...")
entity_linking_results = [graph for graph in tqdm(graph_collection.find(), total=total_graph)]
print("finish loading dev set")
score_list = []
for entity_linking_result in entity_linking_results:
    for result in entity_linking_result['results']:
        score_list.extend(min_max_normalize(result['scores']))
score_list = np.array(score_list)

scores = score_list

def plot_distribution(scores):
    plt.hist(scores, bins=50, alpha=0.7, color='blue')
    plt.title('Entity Linking Score Distribution')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.savefig('/home/shpark/COS/error_analysis/entity_linking_score_distribution_2.png')
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

plot_distribution(scores)

mean_score, median_score = calculate_mean_median(scores)
print(f'Mean: {mean_score}, Median: {median_score}')

quantiles = [q/10 for q in range(1, 10)]
thresholds = calculate_quantiles(scores, quantiles)
print(thresholds)