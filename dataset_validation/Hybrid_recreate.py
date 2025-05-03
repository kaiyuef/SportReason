import json
import os
import re
import asyncio
import random
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ========== 0. 读取环境变量 ==========
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("请在环境变量或 .env 文件中设置 OPENAI_API_KEY")

client = AsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key=OPENAI_KEY
)

# ========== 1. Prompt 模板 ==========
prompt_templates = {
    "sorting": """
You are given a table and related evidence from hyperlinks. Ask a **numerical question** that involves **sorting** the data based on one column to find a value in another column (e.g., "What is the population of the city ranked second by area?", "Which row ranks third in column X, and what is its value in column Y?").

Make sure:
- The final answer is a **number** derived from the sorted data.
- Your explanation clearly shows how sorting the table (and potentially using hyperlink evidence) was used to reach the numerical answer.
- You MUST use the table as evidence. You should try to use relevant hyperlink evidence if it helps answer the question.
""",
    "max_min": """
You are given a table and related evidence from hyperlinks. Ask a **numerical question** that requires identifying a **maximum or minimum value** within the table, potentially refined by hyperlink evidence (e.g., "What is the highest value in column X for entries matching condition Y from the evidence?", "What is the minimum duration mentioned?").

Make sure:
- The final answer is a **number** representing the maximum or minimum.
- Your explanation describes how you found the max/min value using the table and any necessary hyperlink evidence.
- You MUST use the table as evidence.
""",
    "counting": """
You are given a table and related evidence from hyperlinks. Ask a **counting-based numerical question** (e.g., "How many rows satisfy criteria X based on the table and evidence Y?", "How many entities mentioned in the table also appear in hyperlink Z?").

Make sure:
- The final answer is a **number** representing the count.
- Your explanation describes how the count was computed using the table and relevant hyperlink evidence.
- You MUST use the table as evidence.
""",
    "implicit_temporal_numerical": """
You are given a table and related evidence from hyperlinks. Ask a **numerical question** that involves **implicit temporal or numerical reasoning**, such as identifying a value associated with the most recent/oldest event, computing a numerical difference between values/dates found in the table and evidence, or comparing numerical data across different sources.

Examples:
- "What was the value X in the latest year documented in the evidence?"
- "What is the numerical difference between quantity A (from table) and quantity B (from hyperlink)?"
- "How many years passed between event X (table) and event Y (hyperlink)?"

Make sure:
- The final answer is a **number**.
- Your explanation describes the steps in temporal or comparative reasoning using both table and hyperlink evidence.
- You MUST use the table as evidence.
"""
}

# ========== 2. JSON 解析辅助 ==========
def extract_json(content: str) -> dict:
    """Strip Markdown fences & parse JSON."""
    content = re.sub(r"```(?:json)?", "", content).strip()
    return json.loads(content)

# ========== 3. 核心生成函数 ==========
async def generate_variant(full_table, all_hyperlink_evidences, variant_id, file_id, category):
    chosen_template = prompt_templates[category]

    # ---------- Evidence pool ----------
    evidence_pool = (
        [{"id": "table",
          "evidence_text": json.dumps(full_table[1], ensure_ascii=False),
          "reason": "This is the primary data table."}]
        + all_hyperlink_evidences
    )
    evidence_lookup = {e["id"]: e["evidence_text"] for e in evidence_pool}

    prompt = f"""{chosen_template}

---

### Task Requirements
1. **Generate a NEW numerical question** whose answer is a single number.
2. The question **MUST require combining information from the table provided in the evidence pool.
3. You also need to use textual evidences, use more than 2 textual evidences provided in the evidence pool.
4. Provide a concise step‑by‑step reasoning that makes the multi‑table dependency obvious.

### Evidence Rules
- Include the table provided in the evidence pool.
- Select ≥2 textual evidences **with distinct IDs**.
- For every evidence you include, fill in a short "reason" explaining how it supports the answer.
**Output Format:** Return raw JSON only (no Markdown).

{{
  "seed_question": "<Your newly generated question>",
  "reasoning": "<Step-by-step explanation>",
  "seed_answer": <numerical_answer>,
  "gold_evidences": [
    {{
      "id": "<evidence_id>",
      "evidence_text": "<evidence_text>",
      "reason": "<why it is needed>"
    }}
    /* … more evidences … */
  ],
  "reasoning_type": "{category}",
  "source_file": "{file_id}"
}}

evidence pool:{evidence_pool}
""".strip()

    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs strict JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        structured_output = extract_json(response.choices[0].message.content.strip())

        # ---------- 数值校验 ----------
        if not isinstance(structured_output.get("seed_answer"), (int, float)):
            print(f"⚠️ 非数值答案: {file_id}-V{variant_id}")

        # ---------- 证据文字统一替换 ----------
        for ev in structured_output.get("gold_evidences", []):
            ev_id = ev.get("id")
            if "table" in ev_id:
                # 表格证据直接使用原始表格
                ev['id'] = ev_id
                ev["meta"] = full_table[0]
                ev["content"] = json.dumps(full_table[1], ensure_ascii=False)
                ev['type'] = 'table'
            else:
                ev["meta"] = 'http://en.wikipedia.org' + ev_id
                ev["content"] = evidence_lookup.get(ev_id, "")
                ev["type"] = "text"
                if not ev["content"]:
                    print(f"⚠️ 未匹配到 evidence_text (id={ev_id}) in {file_id}-V{variant_id}")

        # ---------- 附加元数据 ----------
        structured_output.setdefault("reasoning_type", category)
        structured_output["variant_id"] = variant_id
        structured_output["source_file"] = file_id
        return structured_output

    except Exception as e:
        print(f"[{file_id}-V{variant_id}] 生成失败: {e}")
        return None


# ========== 4. 主流程 ==========
async def main():
    table_dir = "hybridqa/WikiTables-WithLinks/sports_tables_tok"
    hyperlink_dir = "hybridqa/WikiTables-WithLinks/sports_request_tok"
    output_dir = "regenerated_questions"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "new_hybrid_aligned.jsonl")

    try:
        table_files = [f for f in os.listdir(table_dir) if f.endswith(".json")]
    except FileNotFoundError:
        print("❌ 表格目录不存在")
        return

    processed_tables = 0
    MAX_TABLES = 2       # 需要处理的表格数量上限

    with open(output_file, "w", encoding="utf-8") as fout:
        for file_name in table_files:
            if processed_tables >= MAX_TABLES:
                break

            table_path     = os.path.join(table_dir, file_name)
            hyperlink_path = os.path.join(hyperlink_dir, file_name)
            if not os.path.exists(hyperlink_path):
                print(f"⚠️ 缺少 hyperlink 文件，跳过 {file_name}")
                continue

            # ---------- 加载表格 ----------
            try:
                table_json = json.load(open(table_path, encoding="utf-8"))
                if "header" not in table_json or "data" not in table_json:
                    print(f"⚠️ 非法表格结构 {file_name}")
                    continue
                full_table = [table_json["url"],{
                    "columns": table_json["header"],
                    "rows": [[cell[0] for cell in row] for row in table_json["data"]]
                }]
            except Exception as e:
                print(f"⚠️ 解析表格失败 {file_name}: {e}")
                continue

            # ---------- 加载超链接 ----------
            hyperlink_dict = json.load(open(hyperlink_path, encoding="utf-8"))
            all_hyperlinks = [
                {"id": k, "evidence_text": v,
                 "reason": "External context related to table content."}
                for k, v in hyperlink_dict.items()
            ]

            file_id = file_name[:-5]
            print(f"🔄 处理 {file_id} …")

            prompt_keys = list(prompt_templates.keys())
            tasks = [
                generate_variant(full_table, all_hyperlinks, j, file_id, prompt_keys[j % len(prompt_keys)])
                for j in range(5)
            ]

            results = await asyncio.gather(*tasks)

            for r in results:
                if r:
                    fout.write(json.dumps(r, ensure_ascii=False, indent =4) + "\n")

            processed_tables += 1
            print(f"✅ 完成 {file_id} | 已处理表格数: {processed_tables}")
            await asyncio.sleep(1)   # 简易节流

    print("🎉 脚本结束! 结果已写入", output_file)

# ========== 5. 入口 ==========
if __name__ == "__main__":
    asyncio.run(main())
