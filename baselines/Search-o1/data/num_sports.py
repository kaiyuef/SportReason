"""Data preprocess for NumSports
"""
import json
from tqdm import tqdm

test_path = '/home/siyue/Projects/Search-o1/data/merged_dataset_final.jsonl'
output_path = './num_sports.json'

data_list = []
with open(test_path, 'r') as file:
    # Process JSONL file line by line
    for line in tqdm(file):
        item = json.loads(line)  # Parse each line as a JSON object
        if len(item['answers'])<1:
            print(item)
            assert 1==2
            continue
        ans = item['answers'][0]
        data_list.append({
            'id': item['id'],
            'Question': item['seed_question'],
            'answer': str(ans),
        })

# Write the updated data to JSON
with open(output_path, mode='w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, indent=4, ensure_ascii=False)