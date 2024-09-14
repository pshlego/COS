import json
import re
import copy
import unicodedata
from tqdm import tqdm
from Algorithms.Ours.dpr.utils.tokenizers import SimpleTokenizer
from Algorithms.Ours.dpr.data.qa_validation import has_answer
import argparse

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def remove_accents_and_non_ascii(text):
    # Normalize the text to decompose combined characters
    normalized_text = unicodedata.normalize('NFD', text)
    # Remove all characters that are not ASCII
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining non-letter characters
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,!.?\-]', '', ascii_text)
    return cleaned_text

def evaluate(star_graph_info_list, qa_data, table_to_chunk, table_key_to_content, passage_key_to_content, tokenizer):
    edge_count = 0
    answers = qa_data['answers']
    context = ""
    retrieved_table_set = set()
    retrieved_passage_set = set()
    
    for star_graph_info in star_graph_info_list:
        global_table_id = star_graph_info['table_id']
        global_row_id = star_graph_info['row_id']
        try:
            chunk_info = table_to_chunk[str(global_table_id)][str(global_row_id)]
            table_id = chunk_info['chunk_id']
            row_id = int(chunk_info['local_row_id'])
        except:
            global_row_text = ', '.join(star_graph_info['table'][1][0])
            chunk_info_list = table_to_chunk[str(global_table_id)]
            table_chunk_id_list = list(set([chunk_info['chunk_id'] for row_id, chunk_info in chunk_info_list.items()]))
            is_find = False
            for table_chunk_id in table_chunk_id_list:
                row_text_list = table_key_to_content[table_chunk_id]['text'].split('\n')
                if global_row_text in row_text_list:
                    row_id = row_text_list.index(global_row_text)
                    table_id = table_chunk_id
                    is_find = True
                    break
            if not is_find:
                continue

        table = table_key_to_content[table_id]

        if table_id not in retrieved_table_set:
            retrieved_table_set.add(table_id)
            
            # if edge_count == final_max_edge_count:
            #     continue
            
            context += table['text']
            edge_count += 1

        for raw_passage_id in star_graph_info['passages_id']['all_index']:
            passage_id = raw_passage_id.replace('/wiki/','').replace('_', ' ')

            # if passage_id in retrieved_passage_set:
            #     continue
            
            retrieved_passage_set.add(passage_id)
            passage_content_text = passage_key_to_content[raw_passage_id]
            passage_text = passage_id + ' ' + passage_content_text

            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            table_segment_text = column_name + '\n' + row_values
            
            edge_text = table_segment_text + '\n' + passage_text
            
            # if edge_count == final_max_edge_count:
            #     continue
            
            edge_count += 1
            context += edge_text

    normalized_answers = [remove_accents_and_non_ascii(answer) for answer in answers]
    normalized_context = remove_accents_and_non_ascii(context)
    is_has_answer = has_answer(normalized_answers, normalized_context, tokenizer, 'string', max_length=4096)
    
    if is_has_answer:
        recall = 1
        error_analysis = copy.deepcopy(qa_data)
        error_analysis['retrieved_graph'] = retrieved_graph
        if  "hard_negative_ctxs" in error_analysis:
            del error_analysis["hard_negative_ctxs"]
    else:
        recall = 0
        error_analysis = copy.deepcopy(qa_data)
        error_analysis['retrieved_graph'] = retrieved_graph
        if  "hard_negative_ctxs" in error_analysis:
            del error_analysis["hard_negative_ctxs"]

    return recall, error_analysis

if __name__ == "__main__":
    retrieved_results_path = "/mnt/sde/OTT-QAMountSpace/DoTTeR/Embeddings/indexed_embeddings/dev_output_k100_table_corpus_blink.jsonl"
    retrieved_results = read_jsonl(retrieved_results_path)
    table_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    passage_data_path = "/mnt/sde/OTT-QAMountSpace/OTT-QA/data/all_passages.json"
    passage_ids_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/ColBERT_Embedding_Dataset/passage_cos_version/index_to_chunk_id.json"
    qa_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_dev_q_to_tables_with_bm25neg.json"
    table_to_chunk_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Evaluation_Dataset/table_to_chunk.json"
    print(f"Loading corpus...")
    
    print("3. Loading tables...")
    table_key_to_content = {}
    table_contents = json.load(open(table_data_path))
    print("3. Loaded " + str(len(table_contents)) + " tables!")
    print("3. Processing tables...")
    for table_key, table_content in tqdm(enumerate(table_contents)):
        table_key_to_content[table_content['chunk_id']] = table_content
    print("3. Processing tables complete!", end = "\n\n")
    
    # Load passages
    print("4. Loading passages...")
    passage_key_to_content = {}
    passage_key_to_content = json.load(open(passage_data_path))

    qa_data = json.load(open(qa_data_path))
    table_to_chunk = json.load(open(table_to_chunk_path))
    qid_to_qa_datum = {qa_datum['id']: qa_datum for qa_datum in qa_data}
    
    tokenizer = SimpleTokenizer()
    error_case_list = []
    recall_list = []
    for retrieved_result in tqdm(retrieved_results):
        qid = retrieved_result["question_id"]
        qa_datum = qid_to_qa_datum[qid]
        retrieved_graph = retrieved_result["top_100"]
        recall, error_case = evaluate(retrieved_graph, qa_datum, table_to_chunk, table_key_to_content, passage_key_to_content, tokenizer)
        recall_list.append(recall)
        error_case_list.append(error_case)
            
    print("Min: ", sum(recall_list) / len(recall_list))
    print('len: ', len(recall_list))
        