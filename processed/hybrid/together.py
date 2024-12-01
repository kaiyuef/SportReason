import json

# 定义输入和输出文件路径
input_file_1 = 'processed/hybrid/dev_reformatted_questions.jsonl'
input_file_2 = 'processed/hybrid/train_reformatted_questions.jsonl'
output_file = 'processed/hybrid/merged_hybrid.jsonl'

# 用于存储合并后的数据的列表
merged_data = []

# 读取第一个jsonl文件并合并数据
with open(input_file_1, 'r', encoding='utf-8') as f1:
    for line in f1:
        data = json.loads(line)
        merged_data.append(data)

# 读取第二个jsonl文件并合并数据
with open(input_file_2, 'r', encoding='utf-8') as f2:
    for line in f2:
        data = json.loads(line)
        merged_data.append(data)

# 将合并后的列表保存到新的jsonl文件中
with open(output_file, 'w', encoding='utf-8') as f_out:
    for item in merged_data:
        json.dump(item, f_out)
        f_out.write('\n')

print(f"合并完成！结果已保存到 {output_file}")
