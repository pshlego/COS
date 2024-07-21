import ast
import json
import vllm
from tqdm import tqdm
from transformers import set_seed
from prompts import select_table_segment_prompt
from pymongo import MongoClient
# VLLM Parameters
COK_VLLM_TENSOR_PARALLEL_SIZE = 2 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
COK_VLLM_GPU_MEMORY_UTILIZATION = 1.0 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
set_seed(0)

class LlamaTraverler:
    def __init__(self, model_path):
        self.llm = vllm.LLM(
            model_path,
            worker_use_ray=True,
            tensor_parallel_size=COK_VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=COK_VLLM_GPU_MEMORY_UTILIZATION, 
            trust_remote_code=True,
            dtype="half", # note: bfloat16 is not supported on nvidia-T4 GPUs
            # max_model_len=4096, # input length + output length
            enforce_eager=True,
        )
        self.prompt_template = select_table_segment_prompt
        self.max_passage_length = 128
        self.tokenizer = self.llm.get_tokenizer()
    
    def select(self, question, graph):
        prompt = self.get_prompt(question, graph)
        
        responses = self.llm.generate(
                [prompt],
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.5,  # randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=128,  # Maximum number of tokens to generate per output sequence.
                    logprobs=1
                ),
                use_tqdm = False
            )

        selected_node = responses[0].outputs[0].text
        
        return selected_node
    
    def get_prompt(self, question, table_and_linked_passages):
        prompt = self.prompt_template.format(question = question, table_and_linked_passages = table_and_linked_passages)
        return prompt

def convert_table_to_text(table_info, row_id_to_linked_passages, tokenizer):
    table_and_linked_passages = ""
    table_and_linked_passages += f"Table Name: {table_info['title']}\n"
    table_and_linked_passages += f"Column Name: {table_info['column_name'].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ')}\n\n"
    for row_id, row_content in enumerate(table_info['rows']):
        table_and_linked_passages += f"Row_{row_id + 1}: {row_content.replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ')}\n"
        if str(row_id) in row_id_to_linked_passages:
            table_and_linked_passages += f"Passages linked to Row_{row_id + 1}:\n"
            for linked_passage in list(set(row_id_to_linked_passages[str(row_id)])):
                tokenized_content = tokenizer.encode(linked_passage)
                trimmed_tokenized_content = tokenized_content[:128]
                trimmed_content = tokenizer.decode(trimmed_tokenized_content)
                table_and_linked_passages += f"- {trimmed_content}\n"
        table_and_linked_passages += "\n\n"

    return table_and_linked_passages

if __name__ == "__main__":
    model_path = "/mnt/sdf/shpark/OTT-QAMountSpace/OTT-QAMountSpace/ModelCheckpoints/Ours/llm/TechxGenus/Meta-Llama-3-70B-Instruct-AWQ"
    table_data_path = "/mnt/sdf/shpark/OTT-QAMountSpace/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    passage_data_path = "/mnt/sdf/shpark/OTT-QAMountSpace/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json"
    generated_data_graph_path = "/mnt/sdf/shpark/OTT-QAMountSpace/OTT-QAMountSpace/AnalysisResults/COS/DataGraphConstructor/table_chunks_to_passages_cos_table_passage.json"
    path = "/root/OTT_QA_Workspace/error_case/error_case_1/table_segment_error_cases_reranking_last_baai_rerank_full_layer_wo_table_retrieval_error.json"
    
    # MongoDB Connection Setup
    username = "root"
    password = "1234"
    client = MongoClient(f"mongodb://{username}:{password}@localhost:27017/")
    db = client["mydatabase"]
    print("MongoDB Connected")

    # Load Graph Data
    graph_collection = db["table_chunks_to_passages_cos_table_passage"]
    generated_data_graphs = [graph for graph in graph_collection.find()]
    print(f"Loaded {len(generated_data_graphs)} graphs.")
    
    table_key_to_content = {}
    table_contents = json.load(open(table_data_path))
    for table_key, table_content in enumerate(table_contents):
        table_key_to_content[table_content['chunk_id']] = table_content
    
    passage_key_to_content = {}
    passage_contents = json.load(open(passage_data_path))
    for passage_content in passage_contents:
        passage_key_to_content[passage_content['title']] = passage_content
            
    with open(path, "r") as f:
        data = json.load(f)

    # with open(generated_data_graph_path, 'r') as f:
    #     generated_data_graphs = json.load(f)

    generated_table_chunk_to_passages = {}
    for generated_data_graph in generated_data_graphs:
        generated_table_chunk_to_passages[generated_data_graph['table_chunk_id']] = generated_data_graph
    
    traveler = LlamaTraverler(model_path)
    result_em_list = []
    result_list = []
    for qid, datum in tqdm(data.items()):
        try:        
            question = datum['question']
            table_chunk_id_list = []
            for table_chunk in datum['positive_ctxs']:
                table_chunk_id_list.append(table_chunk['chunk_id'])
            
            sorted_retrieved_graph = datum['sorted_retrieved_graph']
            # linked_passages = []
            table_chunk_id = None
            for node_id, node_info in sorted_retrieved_graph:
                if node_info['type'] == 'table segment' and node_info['chunk_id'] in table_chunk_id_list:
                    # for linked_node_info in node_info['linked_nodes']:
                    #     if linked_node_info[2] == 'edge_reranking':
                    #         linked_passages.append({'title':linked_node_info[0], 'row': node_id.split('_')[-1]})
                    if table_chunk_id is None:
                        table_chunk_id = node_info['chunk_id']

            rows = datum['positive_ctxs'][0]['rows']
            
            real_row_id_list = []
            for positive_ctx in datum['positive_ctxs']:
                if table_chunk_id == positive_ctx['chunk_id']:
                    for single_answer_node in positive_ctx['answer_node']:
                        row_id = single_answer_node[1][0]
                        real_row_id = rows.index(row_id)
                        real_row_id_list.append(f"Row_{real_row_id+1}")
            
            table_title = table_key_to_content[table_chunk_id]['title']
            table_column_name = table_key_to_content[table_chunk_id]['text'].split('\n')[0]
            table_rows = table_key_to_content[table_chunk_id]['text'].split('\n')[1:]
            
            table_rows = [row for row in table_rows if row != ""]
            
            table_info = {"title": table_title, "column_name": table_column_name, "rows": table_rows}
            
            # row_id_to_linked_passages = {}
            # for linked_passage_info in linked_passages:
            #     if linked_passage_info['row'] not in row_id_to_linked_passages:
            #         row_id_to_linked_passages[linked_passage_info['row']] = []
            #     try:
            #         row_id_to_linked_passages[linked_passage_info['row']].append(passage_key_to_content[linked_passage_info['title']]['text'])
            #     except:
            #         continue

            linked_passages = generated_table_chunk_to_passages[table_chunk_id]['results']
            
            row_id_to_linked_passages = {}
            for linked_passage_info in linked_passages:
                if str(linked_passage_info['row']) not in row_id_to_linked_passages:
                    row_id_to_linked_passages[str(linked_passage_info['row'])] = []
                try:
                    row_id_to_linked_passages[str(linked_passage_info['row'])].append(passage_key_to_content[linked_passage_info['retrieved'][0]]['text'])
                except:
                    continue

            table_and_linked_passages = convert_table_to_text(table_info, row_id_to_linked_passages, traveler.tokenizer)
            
            selected_node = traveler.select(question, table_and_linked_passages)
            
            try:
                result = ast.literal_eval(selected_node)
                result = [string.strip() for string in result]
            except:
                result = [selected_node.strip()]
            
            if len(set(real_row_id_list).intersection(set(result))) != 0:
                result_em_list.append(1)
            else:
                result_em_list.append(0)
            
            datum['selected_node'] = selected_node
            result_list.append(datum)
        
        except:
            continue
    
    print("Total", len(result_em_list))
    print("EM Score", sum(result_em_list) / len(result_em_list))
    
    with open("table_segment_selected_results.json", "w") as f:
        json.dump(result_list, f)