import json

def modify_json_key(file_path, old_key, new_key):
    # JSON 파일을 읽음
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 키 값을 변경
    if old_key in data:
        data[new_key] = data.pop(old_key)

    # 변경된 내용을 파일에 다시 저장
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# 사용 예시
file_path = '/mnt/sdd/shpark/cos/knowledge/ott_passage_chunks_original_with_indices.json'
# JSON 파일을 읽음
with open(file_path, 'r') as file:
    data = json.load(file)

# 키 값을 변경
data['chunks'] = data.pop('table_chunks')
data['chunk_ids'] = data.pop('table_chunk_ids')
# 변경된 내용을 파일에 다시 저장
with open(file_path, 'w') as file:
    json.dump(data, file, indent=4)
