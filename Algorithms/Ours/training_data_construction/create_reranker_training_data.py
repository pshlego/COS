import json
from tqdm import tqdm

def read_jsonl_file(file_path):
    """
    Reads a JSONL file and returns a list of JSON objects.
    
    :param file_path: Path to the JSONL file.
    :return: List of JSON objects.
    """
    json_objects = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line.strip())
            json_objects.append(json_object)

    return json_objects

if __name__ == "__main__":
    positive_num = 1
    negative_num = 15
    file_path = '/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Training_Dataset/edge/triples.jsonl'  # Replace with your JSONL file path
    data = read_jsonl_file(file_path)
    prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    
    corpus = {}
    collection_filepath = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Training_Dataset/edge/collection.tsv"
    with open(collection_filepath, "r", encoding="utf8") as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            corpus[pid] = passage


    ### Read the train queries, store in queries dict
    queries = {}
    queries_filepath = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Training_Dataset/edge/queries.tsv"
    with open(queries_filepath, "r", encoding="utf8") as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            queries[qid] = query

    query_dict = {}
    for item in tqdm(data):
        qid = item[0]
        positive_pid = item[1]
        negative_pid = item[2]

        if qid not in query_dict:
            query_dict[qid] = {'pos':[], 'neg':[]}

        if positive_pid not in query_dict[qid]['pos']:
            query_dict[qid]['pos'].append(positive_pid)
        
        if negative_pid not in query_dict[qid]['neg']:
            query_dict[qid]['neg'].append(negative_pid)
    
    training_data = []
    for qid, query_info in tqdm(query_dict.items()):
        training_datum = {}
        training_datum['query'] = queries[str(qid)]
        training_datum['pos'] = [corpus[str(pid)] for pid in query_info['pos'][:positive_num]]
        training_datum['neg'] = [corpus[str(pid)] for pid in query_info['neg'][:negative_num]]
        # training_datum['prompt'] = prompt
        training_data.append(training_datum)
    
    #write a training data as jsonl file
    with open('/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Training_Dataset/edge/reranking_edge_15_negatives.jsonl', 'w', encoding='utf-8') as file:
        for item in training_data:
            file.write(json.dumps(item) + '\n')
    