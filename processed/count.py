import json

# 读取 JSON 文件并统计元素数量，同时将所有的 seed_question 写入 txt 文件
def extract_seed_questions_to_txt(json_file, output_txt_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 检查是否为列表
    if isinstance(data, list):
        with open(output_txt_file, 'w', encoding='utf-8') as txt_file:
            for item in data:
                seed_question = item.get('seed_question', '')
                txt_file.write(seed_question + '\n')  # 写入每个 seed_question，换行
        print(f"已将所有的 seed_question 输出到 {output_txt_file}")
        return len(data)
    else:
        print("JSON 文件内容不是一个列表")
        return None

# 示例使用
json_file = 'processed/tanq/tanq_reformatted.json'  # 替换为你的 JSON 文件路径
output_txt_file = 'processed/tanq/tanq_questions_output.txt'        # 输出的 txt 文件路径

count = extract_seed_questions_to_txt(json_file, output_txt_file)

if count is not None:
    print(f"JSON 列表中有 {count} 个元素")
