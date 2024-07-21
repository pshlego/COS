import ast
import json
import vllm
from tqdm import tqdm
from transformers import set_seed
from prompts import select_passage_prompt
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
        self.prompt_template = select_passage_prompt
        self.max_passage_length = 32
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
    
    def get_prompt(self, question, graph):
        table_segment = graph['table_segment']
        table_segment_content = f"Table Title: {table_segment['title']}" + "\n" + table_segment['content'].replace(' , ', '[special tag]').replace(', ', ' | ').replace('[special tag]', ' , ')
        
        linked_passages = graph['linked_passages']
        linked_passage_contents = ""
        for linked_passage in linked_passages:
            title = linked_passage['title']
            content = linked_passage['content']
            tokenized_content = self.tokenizer.encode(content)
            trimmed_tokenized_content = tokenized_content[: self.max_passage_length]
            trimmed_content = self.tokenizer.decode(trimmed_tokenized_content)
            linked_passage_contents += f"Title: {title}. Content: {trimmed_content}\n\n"
        
        prompt = self.prompt_template.format(question = question, table_segment = table_segment_content, linked_passages = linked_passage_contents)
        return prompt
    
if __name__ == "__main__":
    model_path = "/mnt/sdf/shpark/OTT-QAMountSpace/OTT-QAMountSpace/ModelCheckpoints/Ours/llm/TechxGenus/Meta-Llama-3-70B-Instruct-AWQ"
    table_data_path = "/mnt/sdf/shpark/OTT-QAMountSpace/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    passage_data_path = "/mnt/sdf/shpark/OTT-QAMountSpace/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json"
    table_chunk_to_linked_passage_path = "/mnt/sde/shpark/graph_constructer/mention_detector/gold/gt_dev_entities_chunks_w_exception_handling_and_linked_passage.json"
    path = "/root/OTT_QA_Workspace/error_case/error_case_1/passage_error_cases_reranking_last_baai_rerank_full_layer_wo_table_retrieval_error.json"
    
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
    
    generated_table_chunk_to_passages = {}
    for generated_data_graph in generated_data_graphs:
        generated_table_chunk_to_passages[generated_data_graph['table_chunk_id']] = generated_data_graph
    
    traveler = LlamaTraverler(model_path)
    result_em_list = []
    result_list = []
    for qid, datum in tqdm(data.items()):
        try:
            question = datum['question']
            table_chunk_id = datum['positive_ctxs'][0]['chunk_id']
            rows = datum['positive_ctxs'][0]['rows']
            row_id = datum['positive_ctxs'][0]['answer_node'][0][1][0]
            real_row_id = rows.index(row_id)
            table_title = table_key_to_content[table_chunk_id]['title']
            table_column_name = table_key_to_content[table_chunk_id]['text'].split('\n')[0]
            table_cell_values = table_key_to_content[table_chunk_id]['text'].split('\n')[1:][real_row_id]
            table_segment_content = {"title": table_title, "content": table_column_name + "\n" + table_cell_values}
            linked_passages = generated_table_chunk_to_passages[table_chunk_id]['results']
            linked_passage_contents = []
            row_id_to_linked_passages = {}
            for linked_passage_info in linked_passages:
                if real_row_id == linked_passage_info['row']:
                    for passage_title in linked_passage_info['retrieved'][:5]:
                        linked_passage_contents.append({"title":linked_passage_info['retrieved'][0],"content":passage_key_to_content[passage_title]['text']})
            
            graph = {"table_segment": table_segment_content, "linked_passages": linked_passage_contents}
            
            selected_node = traveler.select(question, graph)
            try:
                result = ast.literal_eval(selected_node)
            except:
                result = selected_node
            
            if len(set(result).intersection(set(datum['positive_passages']))) != 0:
                result_em_list.append(1)
            else:
                result_em_list.append(0)
            
            datum['selected_node'] = selected_node
            result_list.append(datum)
        
        except:
            continue
    
    print("Total", len(result_em_list))
    print("EM Score", sum(result_em_list) / len(result_em_list))
    
    with open("selected_results.json", "w") as f:
        json.dump(result_list, f)