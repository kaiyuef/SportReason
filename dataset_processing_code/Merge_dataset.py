import json
import re

# 读取三个 JSON 文件
with open('processed/tanq/new_tanq_reformatted.json', 'r') as file1, open('processed/templeqa/newer_dev_sports_reformatted.json', 'r') as file3, open('processed/hybrid/new_dev_hybrid.json', 'r') as file2:
    data1 = json.load(file1)
    data2 = json.load(file2)
    data3 = json.load(file3)

# 从每个数据集中顺序取 200 条数据
sample_size = 200
sampled_data1 = data1[:sample_size]
sampled_data2 = data2[:sample_size]
sampled_data3 = data3[:sample_size]

# 合并三个数据集
merged_data = sampled_data1 + sampled_data2 + sampled_data3

# 用于存储所有已使用的 id
id_set = set()

# 生成唯一的 evidence id，保持原前缀
def generate_unique_evidence_id(base_id, id_set):
    match = re.match(r"([a-zA-Z_]+)(\d+)", base_id)
    if not match:
        return base_id  # 如果没有匹配到数字部分，则保持原 id 不变
    prefix, number = match.groups()  # 提取前缀和数字
    new_id = base_id
    new_number = int(number) + 1  # 增加数字部分
    while new_id in id_set:
        new_id = f"{prefix}{new_number}"
        new_number += 1
    id_set.add(new_id)
    return new_id

# 生成唯一的 sample id 以数字递增
def generate_unique_sample_id(base_id, id_set):
    new_id = base_id
    while new_id in id_set:
        new_id += 1  # 简单地递增数字，直到找到唯一的 id
    id_set.add(new_id)
    return new_id

# 对合并数据进行处理，保持 gold_evidence_ids 和 gold_evidences 中的 id 唯一且同步，同时确保 sample id 唯一
for i, item in enumerate(merged_data):
    # 生成唯一的 sample id
    original_id = item.get('id', i)
    if original_id in id_set:
        item['id'] = generate_unique_sample_id(original_id, id_set)
    else:
        id_set.add(original_id)

    # 构建 gold_evidences 中的 id 到 index 的映射，方便同步修改
    evidence_id_map = {evidence['id']: idx for idx, evidence in enumerate(item.get('gold_evidences', []))}

    # 处理 gold_evidence_ids 和 gold_evidences，保证 id 同步且唯一
    for j, evidence_id in enumerate(item.get('gold_evidence_ids', [])):
        evidence = item['gold_evidences'][evidence_id_map[evidence_id]]

        # 确保 gold_evidence_ids 和 gold_evidences 中的 id 保持同步
        if evidence_id in id_set:
            # 如果该 id 已在全局使用，则生成唯一的 id 并同步到 evidence 中
            new_evidence_id = generate_unique_evidence_id(evidence_id, id_set)
            item['gold_evidence_ids'][j] = new_evidence_id
            evidence['id'] = new_evidence_id
        else:
            # 如果该 id 没有被使用，直接将其添加到 id_set 中，并保持同步
            id_set.add(evidence_id)
            evidence['id'] = evidence_id  # 保证 gold_evidence_ids 和 gold_evidences 中的 id 同步

# 将合并后的数据保存到一个新的 JSON 文件
with open('reformated_merged_dataset.json', 'w') as outfile:
    json.dump(merged_data, outfile, indent=4)

print("顺序采样并合并完成，已保存为 merged_dataset.json")
