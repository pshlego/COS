import os
import json
from tqdm import tqdm
def read_json_files(folder_path):
    json_files_list = []
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return json_files_list

    # List all files in the folder
    for filename in tqdm(os.listdir(folder_path)):
        # Check if the file is a JSON file
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as file:
                    # Read the JSON file
                    data_dict = {}
                    data = json.load(file)
                    data_dict['chunk_id'] = filename.split('.')[0]
                    data_dict['title'] = data['title']
                    gold_link_list = []
                    for row_id, row in enumerate(data['data']):
                        for mention in row:
                            if len(mention[1])==0:
                                continue
                            gold_link_info = {}
                            gold_link_info['mention'] = mention[0]
                            gold_link_info['entity'] = mention[1][0][6:].replace('_', ' ')
                            gold_link_info['row'] = row_id
                            gold_link_list.append(gold_link_info)
                    data_dict['gold_link'] = gold_link_list
                    json_files_list.append(data_dict)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    return json_files_list

# Example usage
folder_path = "/mnt/sdd/shpark/data/shpark/preprocessed/tables_tok"  # Replace with your folder path
json_data = read_json_files(folder_path)
with open('/mnt/sdd/shpark/graph/gold_link/gold_link.json', 'w') as outfile:
    json.dump(json_data, outfile)
