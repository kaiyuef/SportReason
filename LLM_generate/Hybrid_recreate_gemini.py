import json
import os
import re
import asyncio
import logging
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APIError

# ========== 0. Configuration & Environment Variables ==========
load_dotenv()

API_KEY = "AIzaSyBIxoRNoYPPgCj8SGC6b_wjArEzn1QTKLA"
# --- API & Model Configuration ---
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME = "gemini-2.5-flash-preview-04-17"
MAX_CONCURRENT_TASKS = 10
API_CALL_DELAY_SECONDS = 0.5
VARIANTS_PER_CATEGORY = 2  

# --- File & Processing Configuration ---
TABLE_DIR = "hybridqa/WikiTables-WithLinks/sports_tables_tok"
HYPERLINK_DIR = "hybridqa/WikiTables-WithLinks/sports_request_tok"
OUTPUT_DIR = "dataset_inprocess"
# NEW: Option to limit the number of tables processed
# Set to an integer (e.g., 10) to process only the first N tables found.
# Set to None to process all tables found in TABLE_DIR.
NUM_SAMPLES_TO_PROCESS = 200  # Or set to None to process all

# ========== Logging Setup ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== 1. Initialize API Client ==========
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)
logging.info(f"API client initialized for base URL: {BASE_URL} using model: {MODEL_NAME}")

# ========== 2. Prompt Templates (Unchanged from previous version) ==========
prompt_templates = {
    "sorting": """
You are given a table and related evidence. Ask a **numerical question** that involves **sorting** the data (e.g., "What is the second highest...?", "Which row ranks third in...?").

Make sure:
- The answer is a number.
- Your explanation shows how sorting was used to reach the answer.
""",
    "max_min": """
You are given a table and related evidence. Ask a **numerical question** that requires identifying a **maximum or minimum** (e.g., "What is the highest...?", "What is the smallest number of...?").

Make sure:
- The answer is a number.
- Your explanation describes how you found the max or min value.
""",
    "counting": """
You are given a table and related evidence. Ask a **counting-based numerical question** (e.g., "How many rows satisfy...?", "How many entities meet condition X?").

Make sure:
- The answer is a number.
- Your explanation describes how the count was computed.
""",
    "implicit_temporal": """
You are given a table and related evidence. Ask a **numerical question** that involves **implicit temporal or numerical reasoning**, such as identifying the most recent event, computing a numerical difference, or reasoning over years or values.

Examples:
- "What year was the latest X?"
- "How many times A won championships"
- "What is the maximum capacity of C"

Make sure:
- The answer is a number.
- Your explanation describes the steps in temporal or comparative reasoning.
"""
}

# ========== 3. JSON Parsing Helper (Unchanged) ==========
def extract_json(content: str, context: str = "") -> dict | None:
    """Strip Markdown fences & parse JSON, returning None on error."""
    try:
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            content = content.strip()
        return json.loads(content)
    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON decode error ({context}): {e}. Content: '{content[:100]}...'")
        return None
    except Exception as e:
        logging.error(f"❌ Unexpected error during JSON extraction ({context}): {e}. Content: '{content[:100]}...'")
        return None

# ========== 4. Core Generation Function (Modified Prompt and Output Structure) ==========
# Added seed_id parameter
async def generate_variant(full_table, all_hyperlink_evidences, variant_id, file_id, seed_id, category, semaphore: asyncio.Semaphore):
    log_prefix = f"[{file_id}-V{variant_id}-{category}]"
    chosen_template = prompt_templates[category]

    # Evidence pool construction (remains the same)
    table_content_str = json.dumps(full_table[1], ensure_ascii=False)
    evidence_pool = [{"id": "table_0", "evidence_text": table_content_str, "reason": "This is the primary data table."}]
    evidence_pool.extend(all_hyperlink_evidences) # Assumes hyperlinks have id, evidence_text, reason

    # Evidence lookup (remains the same)
    evidence_lookup = {e["id"]: e["evidence_text"] for e in evidence_pool}
    evidence_lookup["table_0_data"] = full_table[1] # Store structured data for reconstruction
    evidence_lookup["table_0_url"] = full_table[0]  # Store URL for reconstruction

    # <<< MODIFIED PROMPT >>>
    # Updated Task Requirements and Output JSON Specification
    prompt = f"""{chosen_template}

---

### Task Requirements
1. **Avoid** phrases like “in the table”, “according to the data”, “from the document”, "listed" etc.  
2. Generate a single NEW numerical question whose answer is a single number. 
3. Use **no more than 8** text evidences (in addition to the core evidence facts).  
4. Provide a concise, step-by-step reasoning.  
5. Verify that the answer is correct.  
6. Ensure the answer is unique and precise (no ambiguous interpretations).  
7. **Do NOT** reveal the format or origin of any evidence (e.g., table, document, link).  
8.  The question **MUST** require combining information from the evidence pool.  
9. The question should stand alone as a general knowledge query.
10. The question must obey the rules of open-domain retreival questions

### Evidence Rules
- Add any helpful text evidences (limit 8).
- For every evidence you include, fill in a short "reason" explaining how it supports the answer.

### Examples
- 'Sort the Lewis Hamilton championship seasons after 2015 by the number of races from lowest to highest. What is the number of races in the second season in this sorted list'
- 'How many Super Bowls that the Washington Redskins played in were held in California?'

### BAD EXAMPLES(DO NOT GENERATE THIS KIND OF QUESTIONS)
- 'What is the third highest capacity among the stadiums listed IN THE TABLE?'
- '"How many stadiums listed IN THE TABLE have a capacity greater than 20,000?'

### Output JSON (raw, no markdown)
{{
  "question": "<your generated question>",
  "reasoning": "<step‑by‑step explanation>",
  "answer": <numeric_answer>,
  "gold_evidences": [
    {{
      "id": "<table_or_text_id>",
      "evidence_text": "<evidence_text, **keep this field name** even if it's technically content>",
      "reason": "<why it is needed>"
    }}
    // ... include all selected evidences here
  ],
  "reasoning_type": "{category}"
}}

---
Available Evidence Pool:
{json.dumps(evidence_pool, ensure_ascii=False, indent=2)}

""".strip()

    # Interact with LLM
    structured_output = None
    async with semaphore:
        try:
            await asyncio.sleep(API_CALL_DELAY_SECONDS)
            logging.debug(f"{log_prefix} Requesting completion from model: {MODEL_NAME}")
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs well-formatted JSON for QA datasets."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            content = response.choices[0].message.content.strip()
            logging.debug(f"{log_prefix} Received raw content: '{content[:100]}...'")
            # Use the standard JSON parser
            structured_output_from_llm = extract_json(content, context=log_prefix)

            if not structured_output_from_llm: return None

            # <<< MODIFIED OUTPUT PROCESSING & ASSEMBLY >>>

            # --- Reconstruct Evidences to Match Target Format ---
            processed_evidences = []
            if "gold_evidences" not in structured_output_from_llm or not isinstance(structured_output_from_llm["gold_evidences"], list):
                logging.warning(f"{log_prefix} ⚠️ 'gold_evidences' key missing or not a list in LLM response.")
            else:
                for ev_gen in structured_output_from_llm.get("gold_evidences", []):
                    ev_id = ev_gen.get("id")
                    if not ev_id:
                        logging.warning(f"{log_prefix} ⚠️ Generated evidence missing 'id'. Skipping.")
                        continue

                    # Retrieve reason from LLM output, default if missing
                    reason = ev_gen.get("reason", "Reason not provided by LLM.")
                    processed_ev = None # Placeholder for the processed evidence dict

                    # Handle Table Evidence ('table_0')
                    if ev_id == "table_0":
                        original_table_data = evidence_lookup.get("table_0_data")
                        original_table_url = evidence_lookup.get("table_0_url", "")
                        if original_table_data is None:
                             logging.warning(f"{log_prefix} ⚠️ Table data for id='{ev_id}' not found in lookup. Skipping.")
                             continue
                        processed_ev = {
                            "id": ev_id,
                            # Target format requires content as JSON string for tables
                            "content": json.dumps(original_table_data, ensure_ascii=False),
                            "reason": reason,
                            "type": "table",
                            "url": original_table_url
                        }
                    # Handle Hyperlink Evidence
                    elif not ev_id.startswith("table_"):
                         original_text_content = evidence_lookup.get(ev_id)
                         if original_text_content is None:
                             logging.warning(f"{log_prefix} ⚠️ Hyperlink evidence text for id='{ev_id}' not found in lookup. Skipping.")
                             continue
                         # Reconstruct the full URL for hyperlinks
                         hyperlink_url = 'http://en.wikipedia.org/' + ev_id # Adjusted URL format based on example
                         processed_ev = {
                             "id": ev_id,
                             # Target format requires content as plain string for text
                             "content": original_text_content,
                             "reason": reason,
                             "type": "text",
                             "url": hyperlink_url
                         }
                    # Handle Unknown Evidence IDs
                    else:
                        logging.warning(f"{log_prefix} ⚠️ Unknown evidence id format generated: '{ev_id}'. Skipping.")

                    if processed_ev:
                         processed_evidences.append(processed_ev)

            # --- Assemble Final Output Dictionary in Target Format ---
            final_output = {}
            # Use .get() with fallbacks in case LLM uses old keys despite prompt update
            final_output["question"] = structured_output_from_llm.get("question", structured_output_from_llm.get("seed_question", "Question not generated"))
            final_output["reasoning"] = structured_output_from_llm.get("reasoning", "Reasoning not generated")
            final_output["answer"] = structured_output_from_llm.get("answer", structured_output_from_llm.get("seed_answer", None)) # Default to None if missing
            final_output["gold_evidences"] = processed_evidences # Use the reconstructed list
            final_output["reasoning_type"] = category # From function args
            final_output["seed_dataset"] = "HybridQA" # Hardcoded as per example
            final_output["seed_id"] = seed_id # From function args
            final_output["variant_id"] = variant_id # From function args

            # Final check for numeric answer (if needed)
            if not isinstance(final_output["answer"], (int, float)):
                 # Log warning but keep the result for now, filtering can happen later if needed
                 logging.warning(f"{log_prefix} ⚠️ Final answer is non-numeric: {final_output['answer']}")


            logging.info(f"{log_prefix} ✅ Success.")
            return final_output # Return the dictionary in the target format

        # Error Handling remains the same
        except RateLimitError as e:
            logging.error(f"{log_prefix} ❌ Rate limit error: {e}. Increase delay or reduce concurrency.")
            await asyncio.sleep(5)
            return None
        except APIConnectionError as e:
            logging.error(f"{log_prefix} ❌ API connection error: {e}. Check network/endpoint.")
            return None
        except APIError as e:
             logging.error(f"{log_prefix} ❌ API error: {e} (Status code: {e.status_code})")
             return None
        except json.JSONDecodeError:
             # Should be caught by extract_json, but good to have a fallback
             logging.error(f"{log_prefix} ❌ JSON Decode Error during final processing.")
             return None
        except Exception as e:
            logging.exception(f"{log_prefix} ❌ Unexpected error during generation: {e}")
            return None

# ========== 5. Main Workflow (Pass seed_id, use correct output filename) ==========
async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "new_hybrid_optimized_final.jsonl")

    # 1. 列出所有待处理的表文件
    try:
        all_table_files = [f for f in os.listdir(TABLE_DIR) if f.endswith(".json")]
        logging.info(f"Found {len(all_table_files)} table files in {TABLE_DIR}")
    except Exception as e:
        logging.error(f"❌ Error listing {TABLE_DIR}: {e}")
        return

    # 2. 根据 NUM_SAMPLES_TO_PROCESS 决定要处理的文件列表
    if NUM_SAMPLES_TO_PROCESS:
        files_to_process = all_table_files[:NUM_SAMPLES_TO_PROCESS]
        logging.info(f"Processing first {len(files_to_process)} tables (limit={NUM_SAMPLES_TO_PROCESS})")
    else:
        files_to_process = all_table_files
        logging.info(f"Processing all {len(files_to_process)} tables")

    if not files_to_process:
        logging.warning("No table files to process, exiting.")
        return

    # 3. 为每个 table + 每个 category + 每个变体 生成任务
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    all_tasks = []
    for seed_id, file_name in enumerate(files_to_process):
        file_id = file_name[:-5]
        table_path = os.path.join(TABLE_DIR, file_name)
        hyperlink_path = os.path.join(HYPERLINK_DIR, file_name)

        # 载入 table JSON
        try:
            with open(table_path, 'r', encoding='utf-8') as f:
                tbl = json.load(f)
            full_table = [
                tbl.get("url", ""),
                {"columns": tbl["header"], "rows": [[c[0] if isinstance(c, list) and c else c for c in row] for row in tbl["data"]]}
            ]
        except Exception as e:
            logging.warning(f"Skipping invalid table {file_name}: {e}")
            continue

        # 载入 hyperlink JSON
        try:
            with open(hyperlink_path, 'r', encoding='utf-8') as f:
                link_dict = json.load(f)
            hyperlinks = [
                {"id": k, "evidence_text": v, "reason": "Context from hyperlink."}
                for k, v in link_dict.items() if isinstance(v, str)
            ]
        except Exception as e:
            logging.warning(f"Skipping missing/invalid hyperlinks for {file_name}: {e}")
            continue

        # 创建任务
        for category in prompt_templates.keys():
            for variant_id in range(VARIANTS_PER_CATEGORY):
                all_tasks.append(
                    generate_variant(
                        full_table,
                        hyperlinks,
                        variant_id,
                        file_id,
                        seed_id,
                        category,
                        semaphore
                    )
                )

    logging.info(f"🚀 Running {len(all_tasks)} tasks (concurrency={MAX_CONCURRENT_TASKS})…")
    results = await asyncio.gather(*all_tasks)
    logging.info("✅ Generation complete.")

    # 4. 过滤、打标签并写入
    category_counts = {
        "multi-table + multi text": 0,
        "multi-table": 0,
        "single table+multi text": 0,
        "multi-text": 0
    }
    saved = 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for r in results:
            if not isinstance(r, dict):
                continue

            evidences = r.get("gold_evidences", [])
            table_count = sum(1 for ev in evidences if ev.get("type") == "table")
            text_count  = sum(1 for ev in evidences if ev.get("type") == "text")

            # 分类并丢弃 other
            if table_count >= 2 and text_count >= 2:
                q_cat = "multi-table + multi text"
            elif table_count >= 2:
                q_cat = "multi-table"
            elif table_count == 1 and text_count >= 2:
                q_cat = "single table+multi text"
            elif text_count >= 2:
                q_cat = "multi-text"
            else:
                continue

            r["question_category"] = q_cat
            category_counts[q_cat] += 1

            json.dump(r, fout, ensure_ascii=False, indent=4)
            fout.write("\n")
            saved += 1

    # 5. 输出统计
    logging.info(f"🎉 Done! Saved {saved} items to {output_file}")
    logging.info("Counts per question_category:")
    for cat, cnt in category_counts.items():
        logging.info(f"  {cat}: {cnt}")



# ========== 6. Startup ==========
if __name__ == "__main__":
    asyncio.run(main())





