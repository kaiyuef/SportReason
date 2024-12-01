import json
import re

# 定义输入和输出文件路径
input_file = 'processed/hybrid/dev_filtered_output.jsonl'
output_file = 'processed/hybrid/dev_numerical_output.json'

# 定义与数值相关的关键词
numerical_keywords = [
    'how many', 'how much', 'amount', 'total', 'number', 'count', 'percentage', 
    'ratio', 'average', 'maximum', 'minimum', 'sum', 'difference', 'increase', 
    'decrease', 'proportion', 'median', 'mean', 'variance', 'percent', 'rate', 
    'frequency', 'size', 'measure', 'volume', 'extent', 'capacity', 'weight', 
    'depth', 'height', 'length', 'width', 'distance', 'range', 'spread', 'estimate',
    'value', 'cost', 'price', 'quantity', 'degree', 'intensity', 'score', 'level',
    'balance', 'remaining', 'left', 'duration', 'time', 'interval', 'multiplied by',
    'divided by', 'half of', 'twice', 'thrice', 'quarter of', 'portion of'
]

# 正则表达式匹配数字
number_pattern = re.compile(r'\d+')

# 用于存储符合条件的元素的列表
filtered_elements = []

# 读取原始JSONL文件并筛选
with open(input_file, 'r') as infile:
    for line in infile:
        # 解析JSON行
        element = json.loads(line)
        
        # 检查"seed_question"是否包含数值相关的关键词
        question_contains_numerical = 'question' in element and any(
            keyword in element['question'].lower() for keyword in numerical_keywords
        )
        
        # 检查"answer"是否包含数字
        answer_contains_number = 'answer-text' in element and number_pattern.search(element['answer-text'])
        
        # 如果seed_question或answer符合条件，将元素加入列表
        if question_contains_numerical or answer_contains_number:
            filtered_elements.append(element)

# 将整个元素列表作为一个大列表写入JSON文件
with open(output_file, 'w') as outfile:
    json.dump(filtered_elements, outfile, ensure_ascii=False, indent=4)

print(f'符合条件的元素已成功写入 {output_file}')
