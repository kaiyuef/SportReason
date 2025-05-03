import json

# 文件路径
original_jsonl = "merged_dataset_updated.jsonl"
classified_json = "evidence_usefulness_grouped.json"
output_jsonl = "merged_dataset_useful_only.jsonl"

# 1. 加载 evidence usefulness 结果
with open(classified_json, "r", encoding="utf-8") as f:
    classification_data = json.load(f)

# 建立 id -> useful_evidences 映射
useful_map = {
    str(item["id"]): item["useful"] for item in classification_data
}

# 2. 处理原始文件，替换 gold_evidences 和 gold_evidence_ids
with open(original_jsonl, "r", encoding="utf-8") as fin, \
     open(output_jsonl, "w", encoding="utf-8") as fout:
    
    for line in fin:
        item = json.loads(line.strip())
        item_id = str(item.get("id"))

        if item_id in useful_map:
            useful_evidences = useful_map[item_id]
            useful_ids = [e["id"] for e in useful_evidences]

            # ❗ 跳过 evidence 为空的样本
            if not useful_ids:
                continue

            item["gold_evidences"] = useful_evidences
            item["gold_evidence_ids"] = useful_ids

        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ Updated dataset saved to:", output_jsonl)
