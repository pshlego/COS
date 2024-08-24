import json
import copy
import vllm
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
from Algorithms.ChainOfSkills.FiE_reader.hotpot_evaluate_v1 import f1_score, exact_match_score
from prompts import end_to_end_qa_prompt, few_shot_qa_prompt, chain_of_thought_qa_prompt
@hydra.main(config_path = "conf", config_name = "llm_reader")
def main(cfg: DictConfig):
    retrieved_results_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_dev_q_to_tables_with_bm25neg.json"#cfg.retrieved_results_path

    with open(retrieved_results_path, "r") as f:
        retrieved_results = json.load(f)
    
    print("2. Loading passages...")
    passage_key_to_content = {}
    passage_contents = json.load(open(cfg.passage_data_path))
    PASSAGES_NUM = len(passage_contents)
    for passage_content in tqdm(passage_contents):
        passage_key_to_content[passage_content['title']] = passage_content
    print("2. Loaded " + str(PASSAGES_NUM) + " passages!", end = "\n\n")
    # llm_reader = LLMReader(cfg)
    # ems, f1s = [], []

    # with open(result_path, "w") as f:
    for retrieved_result in tqdm(retrieved_results):
        retrieved_graph = copy.deepcopy(retrieved_result['positive_ctxs'])
        
        for positive_ctx in retrieved_result['positive_ctxs']:
            if 'answer_node' not in positive_ctx:
                continue
        
            if positive_ctx['answer_node'][0][0] in passage_key_to_content:
                passage_info = passage_key_to_content[positive_ctx['answer_node'][0][0]]
                row_id = positive_ctx['rows'].index(positive_ctx['answer_node'][0][1][0])
            else:
                continue
            column_name = positive_ctx['text'].split('\n')[0]
            row_value = positive_ctx['text'].split('\n')[1+row_id]
            retrieved_graph.append({'chunk_id': positive_ctx['chunk_id'], 'title': positive_ctx['title'], 'text': f"{column_name}\n{row_value}\n{passage_info['title']} {passage_info['text']}"})
        
        retrieved_result['ctxs'] = retrieved_graph

    with open('/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ground_truth_retrieval_results.json', 'w') as f:
        json.dump(retrieved_results, f)
    
class LLMReader:
    def __init__(self, cfg):
        # 1. Load tables
        print("1. Loading tables...")
        self.table_key_to_content = {}
        table_contents = json.load(open(cfg.table_data_path))
        TABLES_NUM = len(table_contents)
        for table_key, table_content in tqdm(enumerate(table_contents), total = TABLES_NUM):
            self.table_key_to_content[str(table_key)] = table_content
        print("1. Loaded " + str(TABLES_NUM) + " tables!", end = "\n\n")
        
        
        # 2. Load passages
        print("2. Loading passages...")
        self.passage_key_to_content = {}
        passage_contents = json.load(open(cfg.passage_data_path))
        PASSAGES_NUM = len(passage_contents)
        for passage_content in tqdm(passage_contents):
            self.passage_key_to_content[passage_content['title']] = passage_content
        print("2. Loaded " + str(PASSAGES_NUM) + " passages!", end = "\n\n")

        print("3. Loading LLM...")
        self.llm = vllm.LLM(
            cfg.llm_checkpoint_path,
            worker_use_ray = True,
            tensor_parallel_size = cfg.tensor_parallel_size, 
            gpu_memory_utilization = cfg.gpu_memory_utilization, 
            trust_remote_code = True,
            dtype = "half", # note: bfloat16 is not supported on nvidia-T4 GPUs
            max_model_len = cfg.mex_model_length, # input length + output length
            enforce_eager = True,
        )
        self.tokenizer = self.llm.get_tokenizer()

        self.table_and_linked_passages_trim_length = cfg.table_and_linked_passages_trim_length
        
        self.prompt_temp_type = cfg.prompt_temp_type
        if self.prompt_temp_type == "end_to_end":
            self.prompt_temp = end_to_end_qa_prompt
        elif self.prompt_temp_type == "few_shot":
            self.prompt_temp = few_shot_qa_prompt
        elif self.prompt_temp_type == "chain_of_thought":
            self.prompt_temp = chain_of_thought_qa_prompt
        
        self.topk = cfg.topk
        
    def generate_answer(self, nl_question, retrieved_graph):
        table_and_passages = self.get_table_and_passages(retrieved_graph)
        prompt = self.get_prompt(nl_question, table_and_passages)

        responses = self.llm.generate(
                                    [prompt],
                                    vllm.SamplingParams(
                                        n = 1,  # Number of output sequences to return for each prompt.
                                        top_p = 0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                                        temperature = 0.5,  # randomness of the sampling
                                        skip_special_tokens = True,  # Whether to skip special tokens in the output.
                                        max_tokens = 64,  # Maximum number of tokens to generate per output sequence.
                                        logprobs = 1
                                    ),
                                    use_tqdm = False
                                )
        if self.prompt_temp_type == "chain_of_thought":
            answer_with_rationales = responses[0].outputs[0].text
            if 'the answer is: ' in answer_with_rationales.lower():
                answer = answer_with_rationales.split('the answer is: ')[1]
            else:
                answer = answer_with_rationales
        else:
            answer = responses[0].outputs[0].text

        return answer

    def get_table_and_passages(self, retrieved_graph):
        table_id_to_table_info = {}
        table_id_row_id_to_linked_passage_ids = {}
        table_and_passages = ""
        retrieved_table_set = set()
        for edge_info in retrieved_graph:
            table_id = edge_info['chunk_id']
            if table_id not in retrieved_table_set:
                retrieved_table_set.add(table_id)
                table_title = edge_info['title']
                table_text = edge_info['text']
                table_id_to_table_info[table_id] = {'title': table_title, 'text': table_text, 'rows': [row for row in table_text.split('\n') if row != '']}
                table_id_row_id_to_linked_passage_ids[table_id] = {}
            else:
                row_text = edge_info['text'].split('\n')[1]
                if row_text not in table_id_to_table_info[table_id]['rows']:
                    row_id = 0
                else:
                    row_id = table_id_to_table_info[table_id]['rows'].index(row_text)
                if str(row_id) not in table_id_row_id_to_linked_passage_ids[table_id]:
                    table_id_row_id_to_linked_passage_ids[table_id][str(row_id)] = []
                
                passage_text = edge_info['text'].split('\n')[2]
                
                table_id_row_id_to_linked_passage_ids[table_id][str(row_id)].append(passage_text)
            
        for table_id, table_info in table_id_to_table_info.items():
            table_and_passages += self.stringify_table_and_linked_passages(table_info, table_id_row_id_to_linked_passage_ids[table_id])
            table_and_passages += "\n\n"

        return table_and_passages

    def stringify_table_and_linked_passages(self, table_info, row_id_to_linked_passage_ids):
        # 1. Stringify table metadata
        table_and_linked_passages = ""
        table_and_linked_passages += f"Title : {table_info['title']}\n"
        table_and_linked_passages += f"col : {table_info['rows'][0].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ')}\n\n"
        
        # 2. Stringify each row + its linked passages
        for row_id, row_content in enumerate(table_info['rows']):
            table_and_linked_passages += f"row {row_id + 1} : {row_content.replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ')}\n"
        
        table_and_linked_passages += "\n\n"
        passage_set = set()
        for row_id, row_content in enumerate(table_info['rows']):                                              
            if str(row_id) in row_id_to_linked_passage_ids:
                table_and_linked_passages += f"Passages linked to row {row_id + 1}:\n"
                
                for linked_passage_content in list(set(row_id_to_linked_passage_ids[str(row_id)])):
                    if linked_passage_content in passage_set: continue
                    
                    passage_set.add(linked_passage_content)
                    tokenized_content = self.tokenizer.encode(linked_passage_content)
                    trimmed_tokenized_content = tokenized_content[ : self.table_and_linked_passages_trim_length]
                    trimmed_content = self.tokenizer.decode(trimmed_tokenized_content)
                    table_and_linked_passages += f"- {trimmed_content}\n"

        return table_and_linked_passages

    def get_prompt(self, nl_question, table_and_passages):
        prompt = self.prompt_temp.format(question=nl_question, table_and_passages=table_and_passages)
        return prompt

# def read_jsonl(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             data.append(json.loads(line.strip()))
#     return data

if __name__ == '__main__':
    main()