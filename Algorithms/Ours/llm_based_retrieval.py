import re
import json
import vllm
import hydra
import requests
from tqdm import tqdm
from omegaconf import DictConfig
from transformers import set_seed
from utils.helper import NoIndent, MyEncoder
from prompt.prompts_v2 import detect_aggregation_query_prompt, select_row_wise_prompt, select_passages_prompt, select_passages_prompt_v2

set_seed(0)

@hydra.main(config_path = "conf", config_name = "graph_candidate_retrieval_v2")
def main(cfg: DictConfig):
    table_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_table_chunks_original.json"
    passage_data_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/COS/ott_wiki_passages.json"

    print("1. Loading tables...")
    table_key_to_content = {}
    table_contents = json.load(open(table_data_path))
    print("1. Loaded " + str(len(table_contents)) + " tables!")
    print("1. Processing tables...")
    for table_key, table_content in tqdm(enumerate(table_contents)):
        table_key_to_content[str(table_key)] = table_content
    print("1. Processing tables complete!", end = "\n\n")
    
    # 2. Load passages
    print("2. Loading passages...")
    passage_key_to_content = {}
    passage_contents = json.load(open(passage_data_path))
    print("2. Loaded " + str(len(passage_contents)) + " passages!")
    print("2. Processing passages...")
    for passage_content in tqdm(passage_contents):
        passage_key_to_content[passage_content['title']] = passage_content
    print("2. Processing passages complete!", end = "\n\n")

    llm_node_selector = LlmNodeSelector(cfg, table_key_to_content, passage_key_to_content)
    
    results_path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/graph_query_algorithm/graph_candidate_retrieval_revised.jsonl"
    
    data = read_jsonl(results_path)
    list_answer = []
    for datum in tqdm(data):
        qa_data = datum["qa data"]
        bipartite_subgraph_candidates = datum["retrieved graph"]
        question = qa_data['question']
    
        bipartite_subgraph_candidate_list, table_id_to_row_id_to_linked_passage_ids = get_bipartite_subgraph_candidate_list(bipartite_subgraph_candidates)
        
        is_aggregate = llm_node_selector.detect_aggregation_query(question)
        if is_aggregate:
            selected_rows = llm_node_selector.select_row_wise(question, table_id_to_row_id_to_linked_passage_ids)
            if len(selected_rows) != 0:
                for table_id, row_id, linked_passage_ids in selected_rows:
                    table_segment_id = f"{table_id}_{row_id}"
                    if table_segment_id not in [bipartite_subgraph_candidate['table_segment_id'] for bipartite_subgraph_candidate in bipartite_subgraph_candidate_list]:
                        bipartite_subgraph_candidate_list.append({"table_segment_id":table_segment_id, "linked_passage_ids": linked_passage_ids})

        table_segment_id_to_passage_id_list = llm_node_selector.select_passage_wise(question, bipartite_subgraph_candidate_list)
        list_answer.append({"qa data": qa_data, "table_segment_id_to_passage_id_list": table_segment_id_to_passage_id_list})
        print()

class LlmNodeSelector:
    def __init__(self, cfg, table_key_to_content, passage_key_to_content):
        self.table_and_linked_passages_trim_length = cfg.table_and_linked_passages_trim_length
        self.passage_trim_length = cfg.passage_trim_length

        self.detect_aggregation_query_prompt = detect_aggregation_query_prompt
        self.select_row_wise_prompt = select_row_wise_prompt
        self.select_passages_prompt = select_passages_prompt_v2#select_passages_prompt

        self.table_key_to_content = table_key_to_content
        self.passage_key_to_content = passage_key_to_content
        
        self.llm_addr = "http://localhost:5004/generate"
        self.trim_addr = "http://localhost:5004/trim"
        
    def detect_aggregation_query(self, query):
        prompt = "" + self.detect_aggregation_query_prompt.rstrip() + "\n\n"
        prompt = self.generate_detect_aggregation_query_prompt(query)
        response_list = requests.post(
                self.llm_addr,
                json={
                    "prompt_list": [prompt],
                    "max_tokens": 5
                },
                timeout=None,
            ).json()["response_list"]

        pattern_col = r"f_agg\(\[(.*?)\]\)"

        try:
            pred = re.findall(pattern_col, response_list[0], re.S)[0].strip()
        except:
            return False
        
        if 'true' == pred.lower():
            return True
        else:
            return False
            
    def select_row_wise(self, query, table_id_to_row_id_to_linked_passage_ids):
        prompt_list = []
        table_id_list = []
        for table_id, row_id_to_linked_passage_ids in table_id_to_row_id_to_linked_passage_ids.items():
            prompt = self.generate_select_row_wise_prompt(query, table_id, row_id_to_linked_passage_ids)
            prompt_list.append(prompt)
            table_id_list.append(table_id)

        response_list = requests.post(
                self.llm_addr,
                json={
                    "prompt_list": prompt_list,
                    "max_tokens": 64
                },
                timeout=None,
            ).json()["response_list"]

        pattern_col = r"f_row\(\[(.*?)\]\)"
        selected_table_id_to_row_id_list = {}
        for table_id, response in zip(table_id_list, response_list):
            try:
                pred = re.findall(pattern_col, response, re.S)[0].strip()
            except Exception:
                continue
            
            selected_row_ids = pred.split(', ')
            selected_table_id_to_row_id_list[table_id] = []
            for i, selected_row_id in enumerate(selected_row_ids):
                try:
                    selected_table_id_to_row_id_list[table_id].append(int(selected_row_id.replace('row', '').strip()))
                except:
                    continue

        selected_rows = []
        for table_id, row_id_list in selected_table_id_to_row_id_list.items():
            for row_id in row_id_list:
                linked_passage_ids = table_id_to_row_id_to_linked_passage_ids[str(table_id)][str(row_id)]
                selected_rows.append([table_id, row_id, linked_passage_ids])
        
        return selected_rows
        
    def select_passage_wise(self, question, bipartite_subgraph_candidate_list):
        prompt_list = []
        for bipartite_subgraph_candidate in bipartite_subgraph_candidate_list:
            table_id = bipartite_subgraph_candidate['table_segment_id'].split('_')[0]
            row_id = bipartite_subgraph_candidate['table_segment_id'].split('_')[1]
            linked_passage_ids = bipartite_subgraph_candidate['linked_passage_ids']
            prompt = self.generate_select_passages_prompt_v2(question, table_id, row_id, linked_passage_ids)
            prompt_list.append(prompt)

        response_list = requests.post(
                self.llm_addr,
                json={
                    "prompt_list": prompt_list,
                    "max_tokens": 64
                },
                timeout=None,
            ).json()["response_list"]

        # pattern_col = r"f_passage\(\[(.*?)\]\)"
        pattern_col = r"\[(.*?)\]"
        table_segment_id_to_passage_id_list = {}
        for bipartite_subgraph_candidate, response in zip(bipartite_subgraph_candidate_list, response_list):
            try:
                pred = re.findall(pattern_col, response, re.S)[0].strip()
            except Exception:
                continue
            
            pred_passage_list = pred.split('", "')
            selected_passage_list = []
            for pred_passage in pred_passage_list:
                if pred_passage.replace('"','') not in self.passage_key_to_content:
                    continue
                selected_passage_list.append(pred_passage.replace('"',''))

            table_segment_id = bipartite_subgraph_candidate['table_segment_id']
            table_segment_id_to_passage_id_list[table_segment_id] = selected_passage_list
        
        return table_segment_id_to_passage_id_list
            

    
    def generate_detect_aggregation_query_prompt(self, query):
        prompt = "" + self.detect_aggregation_query_prompt.rstrip() + "\n\n"
        prompt += f"question : {query}.\n"
        prompt += "The answer is : "
        return prompt
    
    def generate_select_row_wise_prompt(self, query, table_id, row_id_to_linked_passage_ids):
        prompt = "" + self.select_row_wise_prompt.rstrip() + "\n\n"
        table_content = self.table_key_to_content[str(table_id)]
        table_title = table_content['title']
        table_text = table_content['text']
        column_names = table_text.split('\n')[0].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ').split(' | ')
        table_prompt_text = f"table caption : {table_title}\n"
        table_prompt_text += f"col : {column_names}\n"
        linked_passages_prompt_text = ""
        row_text_list = table_text.split('\n')[1:]
        for row_id, linked_passage_ids in row_id_to_linked_passage_ids.items():
            table_prompt_text += f"row {row_id} : {row_text_list[int(row_id)].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ').split(' | ')}\n"
            linked_passages_prompt_text += f"passages linked to row {row_id}\n"
            for linked_passage_id in linked_passage_ids:
                passage_content = self.passage_key_to_content[linked_passage_id]
                linked_passage_text = passage_content['text']
                response = requests.post(
                    self.trim_addr,
                    json={
                        "raw_text": linked_passage_text,
                        "trim_length": self.table_and_linked_passages_trim_length
                    },
                    timeout=None,
                ).json()
                trimmed_text = response["trimmed_text"]
                linked_passages_prompt_text += f"Title: {passage_content['title']}. Content: {trimmed_text}\n"

        prompt += "/*\n" + table_prompt_text + "\n*/\n\n"
        prompt += "/*\n" + linked_passages_prompt_text + "\n*/\n\n"
        prompt += f"question : {query}.\n"
        prompt += "The answer is : "

        return prompt
    
    def generate_select_passages_prompt(self, question, table_id, row_id, linked_passage_ids):
        table_content = self.table_key_to_content[str(table_id)]
        table_title = table_content['title']
        table_text = table_content['text']
        column_list = table_text.split('\n')[0].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ').split(' | ')
        column_list = [col.lower() for col in column_list]
        row_value_list = table_text.split('\n')[1+int(row_id)].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ').split(' | ')
        row_value_list = [row_value.lower() for row_value in row_value_list]
        column_w_value_list = []
        for col_id, col in enumerate(column_list):
            try:
                column_w_value_list.append([col, row_value_list[col_id]])
            except:
                continue


        prompt = "" + self.select_passages_prompt.rstrip() + "\n\n"
        dic = {
            "table_caption": table_title,
            "table_column_priority": [NoIndent(column_w_value) for column_w_value in column_w_value_list],
        }
        linear_table_segment_dic = json.dumps(
            dic, cls=MyEncoder, ensure_ascii=False, sort_keys=False, indent=2
        )
        
        linked_passage_list = []
        for linked_passage_id in linked_passage_ids:
            passage_content = self.passage_key_to_content[linked_passage_id]
            passage_text = passage_content['text']
            tokenized_passage_text = self.tokenizer.encode(passage_text)
            trimmed_tokenized_passage_text = tokenized_passage_text[ : self.passage_trim_length]
            trimmed_text = self.tokenizer.decode(trimmed_tokenized_passage_text)
            linked_passage_list.append([passage_content['title'], trimmed_text])
                
        dic = {
            "linked_passage_title": [linked_passage[0] for linked_passage in linked_passage_list],
            "linked_passage_context": [NoIndent(linked_passage) for linked_passage in linked_passage_list],
        }

        linear_passage_dic = json.dumps(
            dic, cls=MyEncoder, ensure_ascii=False, sort_keys=False, indent=2
        )

        prompt += "/*\ntable segment = " + linear_table_segment_dic + "\n*/\n"
        prompt += "/*\nlinked passages = " + linear_passage_dic + "\n*/\n"
        prompt += "question : " + question + ".\n"
        prompt += "The answer is : "
        return prompt

    def generate_select_passages_prompt_v2(self, question, table_id, row_id, linked_passage_ids):
        table_content = self.table_key_to_content[str(table_id)]
        table_title = table_content['title']
        table_text = table_content['text']
        column_names = table_text.split('\n')[0].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ')
        row_values = table_text.split('\n')[1+int(row_id)].replace(' , ', '[SPECIAL]').replace(', ', ' | ').replace('[SPECIAL]', ' , ')
        
        table_segment_text = f"table caption : {table_title}\n"
        table_segment_text += f"col : {column_names}\n"
        table_segment_text += f"row 1 : {row_values}\n\n"
        
        linked_passages_text = ""

        linked_passage_list = []
        for linked_passage_id in linked_passage_ids:
            passage_content = self.passage_key_to_content[linked_passage_id]
            passage_text = passage_content['text']
            response = requests.post(
                self.trim_addr,
                json={
                    "raw_text": passage_text,
                    "trim_length": self.passage_trim_length
                },
                timeout=None,
            ).json()
            trimmed_text = response["trimmed_text"]
            linked_passages_text += f"Title : {passage_content['title']}. Content: {trimmed_text}\n\n"
        
        prompt = self.select_passages_prompt.format(question=question, table_segment=table_segment_text, linked_passages=linked_passages_text)
        
        return prompt

def get_bipartite_subgraph_candidate_list(bipartite_subgraph_candidates):
    table_segment_id_to_linked_passage_ids = {}
    table_id_to_row_id_to_linked_passage_ids = {}
    bipartite_subgraph_candidate_list = []
    
    sorted_node_list = sorted(bipartite_subgraph_candidates.items(), key=lambda x: max([node_info[1] for node_info in x[1]['linked_nodes']]), reverse=True)
    
    for node_id, node_info in sorted_node_list:
        if node_info['type'] == 'table segment':
            table_segment_id_to_linked_passage_ids[node_id] = list(set([node_info[0] for node_info in node_info['linked_nodes']]))
    
    for table_segment_id, linked_passage_ids in table_segment_id_to_linked_passage_ids.items():
        table_id = table_segment_id.split('_')[0]
        row_id = table_segment_id.split('_')[1]
        
        if table_id not in table_id_to_row_id_to_linked_passage_ids:
            table_id_to_row_id_to_linked_passage_ids[table_id] = {}
        
        if row_id not in table_id_to_row_id_to_linked_passage_ids[table_id]:
            table_id_to_row_id_to_linked_passage_ids[table_id][row_id] = []
            
        table_id_to_row_id_to_linked_passage_ids[table_id][row_id].extend(linked_passage_ids)
        
        bipartite_subgraph_candidate_list.append(
            {
                "table_segment_id": table_segment_id, 
                "linked_passage_ids": list(set(table_id_to_row_id_to_linked_passage_ids[table_id][row_id]))
            }
        )
        
        if len(bipartite_subgraph_candidate_list) >= 3:
            break
        
    return bipartite_subgraph_candidate_list, table_id_to_row_id_to_linked_passage_ids


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

if __name__ == "__main__":
    main()