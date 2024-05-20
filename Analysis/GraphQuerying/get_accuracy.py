import json
from tqdm import tqdm
from Algorithms.Ours.dpr.utils.tokenizers import SimpleTokenizer
from Algorithms.Ours.dpr.data.qa_validation import has_answer
if __name__ == "__main__":
    retrieved_graphs_path = "/mnt/sdd/shpark/output/integrated_graph_passage_augmented_only_20_1_v5.json"
    table_data_path= "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    passage_data_path_1= "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json"
    passage_ids_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/ColBERT_Embedding_Dataset/passage_cos_version/index_to_chunk_id.json"
    qa_dataset_path=  "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_dev_q_to_tables_with_bm25neg.json"
    qa_dataset = json.load(open(qa_dataset_path))
    print(f"Loading corpus...")
    table_key_to_content = json.load(open(table_data_path))
    id_to_passage_key = json.load(open(passage_ids_path))
    
    passage_key_to_content = {}
    raw_passages = json.load(open(passage_data_path_1))
    for i, passage in enumerate(raw_passages):
        passage_key_to_content[passage['title']] = passage

    with open(retrieved_graphs_path, 'r') as f:
        retrieved_graphs = json.load(f)
    
    recall_list = []

    tokenizer = SimpleTokenizer()
    
    for retrieved_graph, qa_datum in tqdm(zip(retrieved_graphs, qa_dataset), total=len(qa_dataset)):
        answers = qa_datum['answers']
        context = ""
        # get sorted retrieved graph
        all_included = []
        retrieved_table_set = set()
        retrieved_passage_set = set()
        for node_id, node_info in retrieved_graph.items():
            if 'score' not in node_info:
                node_info['score'] = max([x[1] for x in node_info['linked_table_segment_nodes']])
        
        sorted_retrieved_graph = sorted(retrieved_graph.items(), key=lambda x: x[1]['score'], reverse=True)
        for node_id, node_info in sorted_retrieved_graph:
            if node_info['type'] == 'table segment':
                table_id = int(node_id.split('_')[0])
                table = table_key_to_content[table_id]
                if table_id not in retrieved_table_set:
                    retrieved_table_set.add(table_id)
                    context += table['text']
                    all_included.append(table['text'])
                
                node_info['linked_passage_nodes'] = [(x[0], x[1]) for x in node_info['linked_passage_nodes'] if x[2] != 'passage_node_augmentation']
                #node_info['linked_passage_nodes'] = [(x[0], x[1]) for x in node_info['linked_passage_nodes']]
                max_linked_node_id, max_score = max(node_info['linked_passage_nodes'], key=lambda x: x[1], default=(None, 0))
                if isinstance(max_linked_node_id, int):
                    max_linked_node_id = id_to_passage_key[str(max_linked_node_id)].replace('/wiki/', '').replace('_', ' ')
                
                if max_linked_node_id in retrieved_passage_set:
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
                all_included.append(two_node_graph_text)
                
            elif node_info['type'] == 'passage':
                if node_id in retrieved_passage_set:
                    continue
                retrieved_passage_set.add(node_id)
                try:
                    passage_content = passage_key_to_content[node_id]
                except:
                    continue
                passage_text = passage_content['title'] + ' ' + passage_content['text']
                
                node_info['linked_table_segment_nodes'] = [(x[0], x[1]) for x in node_info['linked_table_segment_nodes'] if x[2] != 'passage_node_augmentation']
                
                if len(node_info['linked_table_segment_nodes']) == 0:
                    continue
                
                max_linked_node_id, max_score = max(node_info['linked_table_segment_nodes'], key=lambda x: x[1], default=(None, 0))
                table_id = int(max_linked_node_id.split('_')[0])
                table = table_key_to_content[table_id]
                
                if table_id not in retrieved_table_set:
                    retrieved_table_set.add(table_id)
                    context += table['text']
                    all_included.append(table['text'])

                row_id = int(max_linked_node_id.split('_')[1])
                table_rows = table['text'].split('\n')
                column_name = table_rows[0]
                row_values = table_rows[row_id+1]
                table_segment_text = column_name + '\n' + row_values
                two_node_graph_text = table_segment_text + '\n' + passage_text
                context += two_node_graph_text
                all_included.append(two_node_graph_text)
        is_has_answer = has_answer(answers, context, tokenizer, 'string', max_length=4096)
        if is_has_answer:
            recall_list.append(1)
        else:
            recall_list.append(0)

    print(f"recall: {sum(recall_list) / len(recall_list)}")
            
        