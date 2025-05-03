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
MAX_CONCURRENT_TASKS = 5
API_CALL_DELAY_SECONDS = 1.1

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
            final_output["seed_dataset"] = "TANQ" # Hardcoded as per example
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
    # Using the output filename from the original script context
    output_file = os.path.join(OUTPUT_DIR, "new_hybrid_optimized2.jsonl")

    try:
        all_table_files = [f for f in os.listdir(TABLE_DIR) if f.endswith(".json")]
        logging.info(f"Found {len(all_table_files)} potential table files in {TABLE_DIR}")
    except FileNotFoundError:
        logging.error(f"❌ Table directory not found: {TABLE_DIR}")
        return
    except Exception as e:
        logging.error(f"❌ Error listing files in {TABLE_DIR}: {e}")
        return

    # Apply NUM_SAMPLES_TO_PROCESS
    files_to_process = all_table_files
    if NUM_SAMPLES_TO_PROCESS is not None and NUM_SAMPLES_TO_PROCESS > 0:
        files_to_process = all_table_files[:NUM_SAMPLES_TO_PROCESS]
        logging.info(f"Processing the first {len(files_to_process)} samples based on NUM_SAMPLES_TO_PROCESS={NUM_SAMPLES_TO_PROCESS}.")
    elif NUM_SAMPLES_TO_PROCESS is not None:
         logging.warning(f"NUM_SAMPLES_TO_PROCESS is set to {NUM_SAMPLES_TO_PROCESS}. No samples will be processed.")
         files_to_process = []
    else:
        logging.info(f"NUM_SAMPLES_TO_PROCESS is None. Processing all {len(files_to_process)} found tables.")

    if not files_to_process:
        logging.warning("No table files selected for processing.")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    processed_tables_count = 0
    all_tasks = []

    # --- Task Creation Phase ---
    logging.info(f"Creating tasks for {len(files_to_process)} selected tables...")
    # <<< Use enumerate to get table_index for seed_id >>>
    for table_index, file_name in enumerate(files_to_process):
        table_path = os.path.join(TABLE_DIR, file_name)
        hyperlink_path = os.path.join(HYPERLINK_DIR, file_name)
        file_id = file_name[:-5]

        if not os.path.exists(hyperlink_path):
            logging.warning(f"⚠️ Missing hyperlink file {hyperlink_path} for table {file_name}. Skipping.")
            continue

        # Load Table Data
        try:
            with open(table_path, 'r', encoding='utf-8') as f_table: table_json = json.load(f_table)
            if not isinstance(table_json.get("header"), list) or not isinstance(table_json.get("data"), list):
                logging.warning(f"⚠️ Invalid table structure in {file_name}. Skipping.")
                continue
            full_table = [
                table_json.get("url", ""),
                {"columns": table_json["header"], "rows": [[cell[0] if isinstance(cell, list) and cell else cell for cell in row] for row in table_json["data"]]}
            ]
        except Exception as e:
            logging.error(f"⚠️ Error processing table file {file_name}: {e}. Skipping.")
            continue

        # Load Hyperlink Data
        try:
            with open(hyperlink_path, 'r', encoding='utf-8') as f_link: hyperlink_dict = json.load(f_link)
            all_hyperlinks = [{"id": k, "evidence_text": v, "reason": "Context from hyperlink."} for k, v in hyperlink_dict.items() if k and isinstance(v, str)]
            if not all_hyperlinks:
                logging.warning(f"⚠️ No valid hyperlinks found in {hyperlink_path} for {file_name}. Skipping.")
                continue
        except Exception as e:
            logging.error(f"⚠️ Error processing hyperlink file {hyperlink_path}: {e}. Skipping.")
            continue

        # Create Generation Tasks
        logging.info(f"📦 Creating generation tasks for table {file_id} (seed_id={table_index})...")
        prompt_keys = list(prompt_templates.keys())
        for j in range(5): # Generate 5 variants per table
            category = prompt_keys[j % len(prompt_keys)]
            # <<< Pass table_index as seed_id >>>
            task = generate_variant(full_table, all_hyperlinks, j, file_id, table_index, category, semaphore)
            all_tasks.append(task)

        processed_tables_count += 1

    # --- Execution Phase ---
    if not all_tasks:
        logging.warning("No tasks were created. Exiting.")
        return

    logging.info(f"🚀 Running {len(all_tasks)} generation tasks concurrently (max {MAX_CONCURRENT_TASKS})...")
    results = await asyncio.gather(*all_tasks)
    logging.info("✅ Task execution finished.")

    # --- Filtering and Writing Results Phase ---
    filtered_results_count = 0
    logging.info(f"Filtering results and writing to {output_file}...")
    logging.info(f"Filter Criteria: Exactly 1 Table AND More than 2 Text Evidences (Text > 2)")

    try:
        with open(output_file, "w", encoding="utf-8") as fout:
            for result in results:
                # Ensure result is not None and is a dictionary before proceeding
                if isinstance(result, dict):
                    # Apply filtering based on the structure of the *processed* evidences
                    evidences = result.get("gold_evidences", [])
                    table_count = sum(1 for ev in evidences if ev.get("type") == "table")
                    text_count = sum(1 for ev in evidences if ev.get("type") == "text")

                    # Filter condition: Exactly 1 table AND more than 2 text evidences
                    if table_count == 1 and text_count > 2:
                        json.dump(result, fout, ensure_ascii=False, indent=4) # Use indent=4 like example
                        fout.write("\n")
                        filtered_results_count += 1
                    else:
                        # Use seed_id and variant_id for logging skipped items
                        s_id = result.get('seed_id', 'N/A')
                        v_id = result.get('variant_id', 'N/A')
                        logging.info(f"Filter SKIPPED for SeedID {s_id}-V{v_id} - Criteria not met (Tables: {table_count}, Texts: {text_count}, Required: T=1, Txt>2)")
                elif result is None:
                    logging.warning("Skipping None result (generation failed earlier)")
                else:
                    logging.error(f"Skipping unexpected result type: {type(result)}")


    except IOError as e:
        logging.error(f"❌ Error writing to output file {output_file}: {e}")
        return
    except Exception as e:
        logging.error(f"❌ Unexpected error during file writing: {e}")
        return

    total_generated = len([r for r in results if r is not None])
    logging.info(f"\n🎉 All done!")
    logging.info(f"   Processed data from {processed_tables_count} tables.")
    logging.info(f"   Successfully generated {total_generated} items before filtering.")
    logging.info(f"   Saved {filtered_results_count} filtered Q&A pairs to: {output_file}")
    logging.info(f"   (Filter Applied: Table Count == 1 AND Text Count > 2)")


# ========== 6. Startup ==========
if __name__ == "__main__":
    asyncio.run(main())





