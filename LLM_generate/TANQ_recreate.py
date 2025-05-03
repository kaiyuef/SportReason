import json
import time
import os
import asyncio
import random
from openai import AsyncOpenAI
from dotenv import load_dotenv

# ========== 0. 加载环境变量 ==========
load_dotenv()
print(os.getenv("OPENAI_API_KEY"))  # ✅ 检查是否成功加载

# ========== 1. 初始化 OpenAI 客户端 ==========
client = AsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key=os.getenv("OPENAI_API_KEY")  # ✅ 推荐从环境变量读取
)

# ========== 2. 分类 Prompt 模板 ==========
prompt_templates = {
    "sorting": """
You are given tables and related evidence. Ask a **numerical question** that involves **sorting** the data (e.g., "What is the second highest...?", "Which row ranks third in...?").

Make sure:
- The answer is a number.
- Your explanation shows how sorting was used to reach the answer.
""",
    "max_min": """
You are given tables and related evidence. Ask a **numerical question** that requires identifying a **maximum or minimum** (e.g., "What is the highest...?", "What is the smallest number of...?").

Make sure:
- The answer is a number.
- Your explanation describes how you found the max or min value.
""",
    "counting": """
You are given tables and related evidence. Ask a **counting-based numerical question** (e.g., "How many rows satisfy...?", "How many entities meet condition X?").

Make sure:
- The answer is a number.
- Your explanation describes how the count was computed.
""",
    "implicit_temporal": """
You are given tables and related evidence. Ask a **numerical question** that involves **implicit temporal or numerical reasoning**, such as identifying the most recent event, computing a numerical difference, or reasoning over years or values.

Examples:
- "What year was the latest X?"
- "How many times A won championships"

Make sure:
- The answer is a number.
- Your explanation describes the steps in temporal or comparative reasoning.
"""
}

# ========== 3. JSON 清洗函数 ==========
def extract_json(content: str) -> dict:
    try:
        if content.startswith("```json") or content.startswith("```"):
            content = content.strip("` \n")
            content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("❌ JSON decode error:", e)
        raise

# ---------- 4. 单个生成任务函数 ----------
async def generate_variant(item, i, j):
    seed_question   = item.get("seed_question", "N/A")
    seed_answers    = item.get("seed_answers", "N/A")
    gold_evidences  = item.get("gold_evidences", [])

    # ========= 4.1 预处理 evidence_pool =========
    evidence_lookup = {}
    for ev in gold_evidences:
        ev_id = ev.get("id")
        if not ev_id:
            continue
        # 判断是表格还是文本；保存成字符串方便写回
        if "table" in ev_id or "infobox" in ev_id:
            evidence_lookup[ev_id] = [json.dumps(ev["content"], ensure_ascii=False),ev['meta']['url']]
        elif "text" in ev_id:
            evidence_lookup[ev_id] = [json.dumps(ev["content"]['text'], ensure_ascii=False),ev['meta']['url']]
        else:
            evidence_lookup[ev_id] = [json.dumps(ev, ensure_ascii=False), ev['meta']['url']]

    # ========= 4.2 构造 Prompt =========
    category = (list(prompt_templates.keys()))[j]
    chosen_template = prompt_templates[category]

    prompt = f"""{chosen_template}
========================================================
### Task Requirements
1. **Generate a single NEW numerical question** whose answer is a single number.
2. The question **MUST require combining information from at least TWO DIFFERENT tables** in the evidence pool (e.g. `table_101` and `table_205`).
3. You also need to use textual evidences, but the core reasoning must depend on those ≥2 tables. Do not use more than 8 text
4. Provide a concise step‑by‑step reasoning.
5. Check your answer after generating the question-answer pair. The Answer must be correct to the question
6. The answer must be unique and precise (no multiple valid interpretations).

### Evidence Rules
- Select ≥2 table evidences **with distinct IDs**.
- Add any helpful text evidences.
- For every evidence you include, fill in a short "reason" explaining how it supports the answer.

### Output JSON (raw, no markdown)
{{
  "question": "<your generated question>",
  "reasoning": "<step‑by‑step explanation>",
  "answer": <numeric_answer>,
  "gold_evidences": [
    {{
      "id": "<table_id>",
      "evidence_text": "<evidence_text>",
      "reason": "<why it is needed>"
    }}
  ],
  "reasoning_type": "{category}"
}}

Original Question: {seed_question}

Available Evidences: {gold_evidences}

""".strip()

    # ========= 4.3 与 GPT 对话 =========
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs well-formatted JSON for QA datasets."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        content = response.choices[0].message.content.strip()
        structured_output = extract_json(content)

        # ========= 4.4 替换 evidence_text =========
        new_evidence_list = []
        for ev in structured_output.get("gold_evidences", []):
            ev_id = ev.get("id")
            if not ev_id:
                continue
            real_text = evidence_lookup.get(ev_id)[0]
            if real_text is None:
                print(f"⚠️ evidence id={ev_id} 不在 gold_evidences 中 -> 已忽略 (Q{i}-V{j})")
                continue
            new_evidence_list.append({
                "id" : ev_id,
                "content": real_text,
                "reason": ev.get("reason", "Used for answering the question"),
                "type": ev_id.split("_")[0],  # 提取类型
                "url": evidence_lookup.get(ev_id)[1],
            })

        structured_output["gold_evidences"] = new_evidence_list
        structured_output["seed_dataset"]   = "TANQ"
        # ========= 4.5 附加元数据 =========
        structured_output["seed_id"]   = i
        structured_output["variant_id"] = j
        structured_output.setdefault("reasoning_type", category)

        print(f"[Q{i}-V{j}] ✅ Success ({category})")
        return structured_output

    except Exception as e:
        print(f"[Q{i}-V{j}] ❌ Error: {e}")
        return None


# ========== 5. 主函数 ==========
async def main():
    input_file = "dataset/merged_dataset_updated.jsonl"
    output_file = "dataset_inprocess/new_TANQ_filtered.jsonl"

    # 读取数据
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(raw_data[:2]):  # 控制处理样本数
            print(f"\n📦 Processing sample {i}...")

            # 并发生成变体
            tasks = [ generate_variant(item, i, j) for j in range(4) ]
            # 简单节流防止瞬时并发过高
            for t in tasks:
                await asyncio.sleep(1)
            results = await asyncio.gather(*tasks)

            # —— 新增：初步筛选，只保留同时含 ≥2 table 和 ≥2 text 的结果 —— #
            filtered = []
            for res in results:
                if not res:
                    continue
                types = [ev["type"] for ev in res["gold_evidences"]]
                if types.count("table") >= 2 and types.count("text") >= 2:
                    filtered.append(res)

            # 将筛选后的结果写入文件
            for result in filtered:
                out_f.write(json.dumps(result, ensure_ascii=False, indent=4) + "\n")

    print("\n🎉 All done! Filtered output saved to:", output_file)


# ========== 6. 启动 ==========
if __name__ == "__main__":
    asyncio.run(main())
