import re
import json
import copy
import unicodedata
from tqdm import tqdm
from Ours.dpr.data.qa_validation import has_answer
from Ours.dpr.utils.tokenizers import SimpleTokenizer
import argparse

def dump_jsonl(data, path):
    """
    Dumps a list of dictionaries to a JSON Lines file.

    :param data: List of dictionaries to be dumped into JSONL.
    :param path: Path where the JSONL file will be saved.
    """
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')
        print(f"Data successfully written to {path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def evaluate(star_graph_info_list, qa_data, table_to_chunk, table_key_to_content, passage_key_to_content, final_max_edge_count):
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
            
            if edge_count == final_max_edge_count:
                continue
            
            context += table['text']
            edge_count += 1

        for raw_passage_id in star_graph_info['passages_id']['all_index']:
            passage_id = raw_passage_id.replace('/wiki/','').replace('_', ' ')

            if passage_id in retrieved_passage_set:
                continue
            
            retrieved_passage_set.add(passage_id)
            passage_content_text = passage_key_to_content[raw_passage_id]
            passage_text = passage_id + ' ' + passage_content_text

            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            table_segment_text = column_name + '\n' + row_values
            
            edge_text = table_segment_text + '\n' + passage_text
            
            if edge_count == final_max_edge_count:
                continue
            
            edge_count += 1
            context += edge_text

    normalized_context = remove_accents_and_non_ascii(context)
    normalized_answers = [remove_accents_and_non_ascii(answer) for answer in answers]
    is_has_answer = has_answer(normalized_answers, normalized_context, SimpleTokenizer(), 'string')
    
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

def remove_accents_and_non_ascii(text):
    # Normalize the text to decompose combined characters
    normalized_text = unicodedata.normalize('NFD', text)
    # Remove all characters that are not ASCII
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining non-letter characters
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,!.?\-]', '', ascii_text)
    return cleaned_text

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description="Evaluate graph queries")
    parser.add_argument('--edge_num', type=int, default=50, help="Maximum number of edges to consider")
    
    # Parse the arguments
    args = parser.parse_args()

    # Paths to files
    results_path = "/mnt/sde/OTT-QAMountSpace/DoTTeR/Embeddings/indexed_embeddings/dev_output_k100_table_corpus_blink.json"#"/mnt/sde/OTT-QAMountSpace/OTTeR/Embeddings/indexed_embeddings/dev_output_k100_table_corpus_blink.jsonl"
    qa_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_dev_q_to_tables_with_bm25neg.json"
    table_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    passage_data_path = "/mnt/sde/OTT-QAMountSpace/OTT-QA/data/all_passages.json"
    table_to_chunk_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Evaluation_Dataset/table_to_chunk.json"

    # Load tables
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

    # Read retrieved results and QA data
    retrieved_results = read_jsonl(results_path)
    recall_list = []
    error_case_list = [] 
    qa_data = json.load(open(qa_data_path))
    table_to_chunk = json.load(open(table_to_chunk_path))
    qid_to_qa_datum = {qa_datum['id']: qa_datum for qa_datum in qa_data}

    # Evaluate each retrieved result
    for retrieved_result in tqdm(retrieved_results):
        qid = retrieved_result["question_id"]
        qa_datum = qid_to_qa_datum[qid]
        retrieved_graph = retrieved_result["top_100"]
        recall, error_case = evaluate(retrieved_graph, qa_datum, table_to_chunk, table_key_to_content, passage_key_to_content, final_max_edge_count=args.edge_num)
        recall_list.append(recall)
        error_case_list.append(error_case)
            
    print("Min: ", sum(recall_list) / len(recall_list))
    print('len: ', len(recall_list))
    print('k: ', args.edge_num)

    # Uncomment to dump error cases if needed
    # dump_jsonl(error_case_list, "/mnt/sdf/OTT-QAMountSpace/AnalysisResults/Ours/GraphQuerier/error_analysis/300_150_llm_70B_128_w_llm_150_full.jsonl")
