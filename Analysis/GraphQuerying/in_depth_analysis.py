import re
import json
import unicodedata
from Algorithms.Ours.dpr.data.qa_validation import has_answer
from Algorithms.Ours.dpr.utils.tokenizers import SimpleTokenizer

def get_context(retrieved_graph, table_key_to_content, passage_key_to_content, tokenizer, query_topk, augment_topk, filtered_retrieval_type):
    two_node_graph_count = 0
    context = ""
    retrieved_table_set = set()
    retrieved_passage_set = set()
    new_sorted_retrieved_graph = []
    sorted_retrieved_graph = sorted(retrieved_graph.items(), key=lambda x: x[1]['score'], reverse=True)
    final_node_rank = 0
    
    for node_rank, (node_id, node_info) in enumerate(sorted_retrieved_graph):
        node_is_added = False
        if node_info['type'] == 'table segment':
            linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation' and (x[3] < query_topk) and (x[4] < augment_topk)) or x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[4] < 2) and (x[3] < 2)) or x[2] == 'two_node_graph_retrieval']
            if len(linked_nodes) == 0:
                continue
            else:
                node_info['linked_nodes'] = linked_nodes
            table_id = node_id.split('_')[0]
            table = table_key_to_content[table_id]
            node_info['chunk_id'] = table['chunk_id']
            if table_id not in retrieved_table_set:
                if two_node_graph_count == 50:
                    break
                
                if not node_is_added:
                    node_is_added = True
                    new_sorted_retrieved_graph.append((node_id, node_info))
                    final_node_rank += 1
                
                retrieved_table_set.add(table_id)
                context += table['text']
                two_node_graph_count += 1
            
            if two_node_graph_count == 50:
                break
            
            max_linked_node_id, max_score, _, _, _ = max(linked_nodes, key=lambda x: x[1], default=(None, 0, 'two_node_graph_retrieval', 0, 0))

            if max_linked_node_id in retrieved_passage_set:
                if not node_is_added:
                    node_is_added = True
                    new_sorted_retrieved_graph.append((node_id, node_info))
                    final_node_rank += 1
                continue
            
            retrieved_passage_set.add(max_linked_node_id)
            passage_content = passage_key_to_content[max_linked_node_id]
            passage_text = passage_content['title'] + ' ' + passage_content['text']
            
            row_id = int(node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            table_segment_text = column_name + '\n' + row_values
            
            two_node_graph_text = table_segment_text + '\n' + passage_text
            context += two_node_graph_text
            two_node_graph_count += 1

            if not node_is_added:
                node_is_added = True
                new_sorted_retrieved_graph.append((node_id, node_info))
                final_node_rank += 1

        elif node_info['type'] == 'passage':
            linked_nodes = [x for x in node_info['linked_nodes'] if x[2] in filtered_retrieval_type and (x[2] == 'table_segment_node_augmentation' and (x[3] < 2) and (x[4] < 2)) or x[2] in filtered_retrieval_type and (x[2] == 'passage_node_augmentation' and (x[4] < query_topk) and (x[3] < augment_topk)) or x[2] == 'two_node_graph_retrieval']

            if len(linked_nodes) == 0:
                continue
            else:
                node_info['linked_nodes'] = linked_nodes


            if node_id in retrieved_passage_set:
                if not node_is_added:
                    node_is_added = True
                    new_sorted_retrieved_graph.append((node_id, node_info))
                    final_node_rank += 1
                continue

            if two_node_graph_count == 50:
                break            

            max_linked_node_id, max_score, _, _, _ = max(linked_nodes, key=lambda x: x[1], default=(None, 0, 'two_node_graph_retrieval', 0, 0))
            table_id = max_linked_node_id.split('_')[0]
            table = table_key_to_content[table_id]
            
            if table_id not in retrieved_table_set:
                retrieved_table_set.add(table_id)
                context += table['text']
                two_node_graph_count += 1

            if two_node_graph_count == 50:
                break

            row_id = int(max_linked_node_id.split('_')[1])
            table_rows = table['text'].split('\n')
            column_name = table_rows[0]
            row_values = table_rows[row_id+1]
            table_segment_text = column_name + '\n' + row_values

            retrieved_passage_set.add(node_id)
            passage_content = passage_key_to_content[node_id]
            passage_text = passage_content['title'] + ' ' + passage_content['text']

            two_node_graph_text = table_segment_text + '\n' + passage_text
            context += two_node_graph_text
            two_node_graph_count += 1
            
            if not node_is_added:
                node_is_added = True
                new_sorted_retrieved_graph.append((node_id, node_info))
                final_node_rank += 1

    return remove_accents_and_non_ascii(context), new_sorted_retrieved_graph, final_node_rank

def remove_accents_and_non_ascii(text):
    # Normalize the text to decompose combined characters
    normalized_text = unicodedata.normalize('NFD', text)
    # Remove all characters that are not ASCII
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    # Remove any remaining non-letter characters
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,!.?\-]', '', ascii_text)
    return cleaned_text

short_path = '/home/shpark/OTT_QA_Workspace/both_error_cases_passage_10_2_short.json'
original_path = '/mnt/sdd/shpark/output/integrated_graph_augmented_passage_10_2_v15_20_fix_scoring.json'
qa_dataset_path=  "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_dev_q_to_tables_with_bm25neg.json"
table_data_path= "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
passage_data_path= "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json"
short_data = json.load(open(short_path))
original_data = json.load(open(original_path))
qa_dataset = json.load(open(qa_dataset_path))

table_key_to_content = {}
table_contents = json.load(open(table_data_path))
for table_key, table_content in enumerate(table_contents):
    table_key_to_content[str(table_key)] = table_content

passage_key_to_content = {}
passage_contents = json.load(open(passage_data_path))
for passage_content in passage_contents:
    passage_key_to_content[passage_content['title']] = passage_content

qid_list = ['dd6a935fc3ca3446', 'a70cfc60e541827f', '1d7f52d9c59e6fc5', '08d4e37cbc7bb2c5', 'b0fe5731a19a8fb9', '960022483a9d97c4', '90b0d5dcf0eaf6b5', 'd8338761374ef6a8', 'e46aefefda60a34d', '622783be803c181c', '822fa10c6b80eb78', '1f1484f82a7625dd']

tokenizer = SimpleTokenizer()
count = 0
for original_datum, qa_datum in zip(original_data, qa_dataset):
    if qa_datum['id'] in qid_list:
        answers = qa_datum['answers']
        context = ""
        # get sorted retrieved graph
        all_included = []
        retrieved_table_set = set()
        retrieved_passage_set = set()
        two_node_graph_count = 0
        retrieved_graph = original_datum
        normalized_context, sorted_retrieved_graph, final_node_rank = get_context(retrieved_graph, table_key_to_content, passage_key_to_content, tokenizer, 10, 2, ['two_node_graph_retrieval', 'passage_node_augmentation'])
        normalized_answers = [remove_accents_and_non_ascii(answer) for answer in answers]
        is_has_answer = has_answer(normalized_answers, normalized_context, tokenizer, 'string')
        
        if is_has_answer:
            print(is_has_answer)
        else:
            count += 1
            print(is_has_answer)

print(count)