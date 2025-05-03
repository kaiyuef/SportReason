import json

def update_gold_evidences(json_file_path):
    # 读取 JSON 文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 遍历 list 中的每个 dict
    for item in data:
        if 'gold_evidences' in item and 'gold_evidence_ids' in item:
            # 获取 gold_evidences 的列表和 gold_evidence_ids
            gold_evidences = item['gold_evidences']
            gold_evidence_ids = item['gold_evidence_ids']
            
            # 创建一个新的列表来存储更新后的 gold_evidences
            updated_evidences = []
            
            # 遍历 gold_evidences 并将对应的 id 和 content 结构化
            for idx, evidence in enumerate(gold_evidences):
                if idx < len(gold_evidence_ids):
                    # 创建新的结构，包含 id 和 content
                    updated_evidence = {
                        "id": gold_evidence_ids[idx],
                        "content": gold_evidences  # 将原有的 evidence 放入 content 中
                    }
                    updated_evidences.append(updated_evidence)  # 添加到列表
            
            # 将更新后的列表放回 gold_evidences 字段
            item['gold_evidences'] = updated_evidences

    # 将更新后的内容保存回文件
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 示例调用
json_file_path = 'processed/templeqa/newer_head_sports_reformatted.json'  # 替换为你的json文件路径
update_gold_evidences(json_file_path)
