import json

# 读取JSONL文件
dict_list = []
input_file = 'processed/hybrid/new_pretty_numerical_hybird_reformatted_questions.jsonl'
output_file = 'processed/hybrid/hybrid_before_merge.json'

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        dict_item = json.loads(line.strip())  # 读取每一行并解析为字典
        dict_list.append(dict_item)  # 将每个字典添加到列表中

# 将列表写入到新的JSON文件中
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dict_list, f, ensure_ascii=False, indent=4)

print(f"数据已成功写入到 {output_file}")
