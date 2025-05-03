import json

original_file = "merged_dataset_useful_only.jsonl"
generated_file = "new_dataset_async.jsonl"
output_file = "merged_dataset_TanQ_renewed.jsonl"

# Step 1: 读取原始数据为映射 (id -> item)
with open(original_file, "r", encoding="utf-8") as f:
    original_data = [json.loads(line.strip()) for line in f]

original_map = {str(item["id"]): item for item in original_data}

# Step 2: 读取生成数据
with open(generated_file, "r", encoding="utf-8") as f:
    generated_data = [json.loads(line.strip()) for line in f]

# Step 3: 为新生成的问题赋予新的格式
new_samples = []
new_id = 600

for gen in generated_data:
    seed_index = str(gen["seed_id"])
    seed_item = original_data[int(seed_index)]  # 原始问题

    # 获取使用到的 evidence id 列表
    used_ids = [e["id"] for e in gen["used_evidences"]]
    evidences = seed_item.get("gold_evidences", [])

    # 从原始样本中筛选对应的 gold_evidences
    selected_evidences = [ev for ev in evidences if ev["id"] in used_ids]

    # 统计类型
    type_count = {"request_count": 0, "table_count": 0}
    for eid in used_ids:
        if "table" in eid:
            type_count["table_count"] += 1
        elif "text" in eid:
            type_count["request_count"] += 1

    # 构造新样本
    new_sample = {
        "id": new_id,
        "seed_question": gen["new_question"],
        "seed_dataset": "HYBRID",
        "seed_split": "dev",
        "seed_answers": str(gen["answer"]),
        "seed_id": str(seed_item["id"]),
        "extended_question": "",
        "gold_evidence_ids": used_ids,
        "gold_evidence_type": type_count,
        "gold_evidences": selected_evidences
    }

    new_samples.append(new_sample)
    new_id += 1

# Step 4: 合并原始 + 新生成数据并写入
combined = original_data + new_samples

with open(output_file, "w", encoding="utf-8") as f:
    for item in combined:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 成功合并，共输出 {len(combined)} 条数据，保存到: {output_file}")
