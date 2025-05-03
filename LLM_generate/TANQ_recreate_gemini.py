import json
import time
import os
import asyncio
import random
import logging
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APIError
from dotenv import load_dotenv

# ========== 0. Configuration & Constants ==========
load_dotenv()

API_KEY = "AIzaSyBIxoRNoYPPgCj8SGC6b_wjArEzn1QTKLA"


BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/" # Keep user's specified base URL
MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Newer model identifier, adjust if needed
INPUT_FILE = "dataset/merged_dataset_updated.jsonl"
OUTPUT_FILE = "dataset_inprocess/new_TANQ_gemini_optimized4.jsonl"
NUM_SAMPLES_TO_PROCESS = 200 # Set to None to process all samples
MAX_CONCURRENT_TASKS = 5 # Limit concurrent API calls
API_CALL_DELAY_SECONDS = 1.1 # Delay between API calls to respect rate limits

# Categories map directly to prompt keys
CATEGORIES = ["sorting", "max_min", "counting", "implicit_temporal"]

# ========== Logging Setup ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== 1. Initialize OpenAI Client ==========
# Use the API key loaded from environment variables
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)
logging.info(f"OpenAI client initialized for base URL: {BASE_URL}")

# ========== 2. Category Prompt Templates ==========
# (Keep the original templates as they define the core logic)
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

# ========== 3. JSON Cleaning Function ==========
def extract_json(content: str, context: str = "") -> dict | None:
    """
    Safely extracts JSON content from a string, handling potential markdown fences.
    Returns None if parsing fails.
    """
    try:
        # Remove potential markdown fences and leading/trailing whitespace/newlines
        if content.startswith("```"):
            content = content.strip("` \n")
            if content.startswith("json"):
                 content = content[4:].strip() # Remove "json" prefix after ```

        return json.loads(content)
    except json.JSONDecodeError as e:
        logging.error(f"❌ JSON decode error ({context}): {e}. Content: '{content[:100]}...'")
        return None
    except Exception as e:
        logging.error(f"❌ Unexpected error during JSON extraction ({context}): {e}. Content: '{content[:100]}...'")
        return None

# ========== 4. Single Generation Task Function ==========
async def generate_variant(item: dict, item_index: int, variant_index: int, category: str, semaphore: asyncio.Semaphore):
    """
    Generates a single question variant for a given item and category.
    Uses a semaphore to limit concurrency.
    """
    seed_question   = item.get("seed_question", "N/A")
    # seed_answers    = item.get("seed_answers", "N/A") # Not used in prompt, commented out
    gold_evidences  = item.get("gold_evidences", [])
    log_prefix = f"[Q{item_index}-V{variant_index}-{category}]"

    # ========= 4.1 Preprocess evidence_pool =========
    evidence_lookup = {}
    for ev in gold_evidences:
        ev_id = ev.get("id")
        if not ev_id:
            logging.warning(f"{log_prefix} ⚠️ Found evidence without ID in item {item_index}. Skipping it.")
            continue

        url = ev.get('meta', {}).get('url', 'URL_Not_Found')
        content_str = ""
        try:
            # Standardize evidence extraction
            if "table" in ev_id or "infobox" in ev_id:
                content_str = json.dumps(ev.get("content", {}), ensure_ascii=False)
            elif "text" in ev_id:
                 # Assuming content is structured like {"text": "...", ...}
                 content_str = json.dumps(ev.get("content", {}).get("text", ""), ensure_ascii=False)
            else: # Fallback for unknown types
                 content_str = json.dumps(ev.get("content", ev), ensure_ascii=False) # Use ev itself if content key missing
            evidence_lookup[ev_id] = {"text": content_str, "url": url}
        except Exception as e:
             logging.error(f"{log_prefix} ❌ Error processing evidence {ev_id}: {e}")
             # Store minimal info even if processing fails partially
             evidence_lookup[ev_id] = {"text": "Error processing content", "url": url}


    # ========= 4.2 Construct Prompt =========
    chosen_template = prompt_templates[category]

    # Ensure gold_evidences passed to the prompt is the original structure
    # The prompt needs the full structure to understand the context
    prompt = f"""{chosen_template}
========================================================
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
- Include at least 2 tables provided
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

Original Question: {seed_question}

Available Evidences: {json.dumps(gold_evidences, ensure_ascii=False, indent=2)}

""".strip() # Pass the original gold_evidences structure as JSON string

    # ========= 4.3 Interact with LLM within Semaphore and with Delay =========
    structured_output = None
    async with semaphore:
        try:
            # Add delay *before* the call inside the semaphore lock
            await asyncio.sleep(API_CALL_DELAY_SECONDS)

            logging.debug(f"{log_prefix} Requesting completion from model: {MODEL_NAME}")
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs well-formatted JSON for QA datasets."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            content = response.choices[0].message.content.strip()
            logging.debug(f"{log_prefix} Received raw content: '{content[:100]}...'")
            structured_output = extract_json(content, context=log_prefix)

            if not structured_output:
                 # Error already logged by extract_json
                 return None # Exit if JSON parsing failed

            # ========= 4.4 Process and Validate Generated Evidences =========
            new_evidence_list = []
            if "gold_evidences" not in structured_output or not isinstance(structured_output["gold_evidences"], list):
                logging.warning(f"{log_prefix} ⚠️ 'gold_evidences' key missing or not a list in LLM response. Skipping evidence processing.")
                structured_output["gold_evidences"] = [] # Ensure the key exists as an empty list
            else:
                for ev_gen in structured_output.get("gold_evidences", []):
                    ev_id = ev_gen.get("id")
                    if not ev_id:
                        logging.warning(f"{log_prefix} ⚠️ Generated evidence missing 'id'. Skipping.")
                        continue

                    original_evidence_data = evidence_lookup.get(ev_id)
                    if original_evidence_data is None:
                        logging.warning(f"{log_prefix} ⚠️ Generated evidence id='{ev_id}' not found in original evidences. Skipping.")
                        continue

                    # Determine type based on ID prefix
                    ev_type = "unknown"
                    if "_" in ev_id:
                        ev_type = ev_id.split("_")[0]

                    new_evidence_list.append({
                        "id": ev_id,
                        # Use the actual content string derived earlier
                        "content": original_evidence_data["text"],
                        "reason": ev_gen.get("reason", "Reason not provided by LLM."),
                        "type": ev_type,
                        "url": original_evidence_data["url"],
                    })

                structured_output["gold_evidences"] = new_evidence_list

            # ========= 4.5 Add Metadata =========
            structured_output["seed_dataset"] = "TANQ"
            structured_output["seed_id"] = item_index
            structured_output["variant_id"] = variant_index
            # Ensure reasoning_type is present, using the category as default
            structured_output.setdefault("reasoning_type", category)

            # Basic validation (can be expanded)
            if not all(k in structured_output for k in ["question", "reasoning", "answer", "gold_evidences", "reasoning_type"]):
                 logging.warning(f"{log_prefix} ⚠️ Output missing required keys.")
                 # Decide if this should return None or proceed with partial data
                 # return None # Option: Treat as failure

            logging.info(f"{log_prefix} ✅ Success.")
            return structured_output

        except RateLimitError as e:
            logging.error(f"{log_prefix} ❌ Rate limit error: {e}. Consider increasing API_CALL_DELAY_SECONDS or reducing MAX_CONCURRENT_TASKS.")
            return None
        except APIConnectionError as e:
            logging.error(f"{log_prefix} ❌ API connection error: {e}. Check network or API endpoint.")
            return None
        except APIError as e:
             logging.error(f"{log_prefix} ❌ OpenAI API error: {e} (Status code: {e.status_code})")
             return None
        except Exception as e:
            logging.exception(f"{log_prefix} ❌ Unexpected error during generation: {e}") # Use logging.exception to include traceback
            return None


# ========== 5. Main Function ==========
async def main():
    """
    Main function to read data, generate variants concurrently, filter, and write results.
    """
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            raw_data = [json.loads(line) for line in f]
        logging.info(f"Read {len(raw_data)} items from {INPUT_FILE}")
    except FileNotFoundError:
        logging.error(f"❌ Input file not found: {INPUT_FILE}")
        return
    except json.JSONDecodeError as e:
         logging.error(f"❌ Error decoding JSON from {INPUT_FILE}: {e}")
         return
    except Exception as e:
        logging.error(f"❌ Error reading input file {INPUT_FILE}: {e}")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    tasks = []

    data_to_process = raw_data
    if NUM_SAMPLES_TO_PROCESS is not None:
        data_to_process = raw_data[:NUM_SAMPLES_TO_PROCESS]
        logging.info(f"Processing the first {len(data_to_process)} samples.")

    for i, item in enumerate(data_to_process):
        logging.info(f"\n📦 Creating tasks for sample {i}...")
        for j, category in enumerate(CATEGORIES):
            # Create a task for each variant generation
            tasks.append(generate_variant(item, i, j, category, semaphore))

    logging.info(f"Created {len(tasks)} generation tasks. Running concurrently (max {MAX_CONCURRENT_TASKS})...")

    # Run tasks concurrently and gather results
    results = await asyncio.gather(*tasks)

    # Filter results and write to output file
    filtered_count = 0
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True) # Ensure output directory exists
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
            for result in results:
                if not result: # Skip None results (errors)
                    continue

                # Filtering logic based on evidence types
                evidence_types = [ev.get("type", "unknown") for ev in result.get("gold_evidences", [])]
                table_count = evidence_types.count("table") + evidence_types.count("infobox") # Count both table and infobox as "table" type for filtering
                text_count = evidence_types.count("text")

                # Apply the filter: >= 2 tables/infoboxes AND >= 2 text evidences
                # Updated requirement check based on user script comments
                if table_count >= 2 and text_count >= 0: # Original script had >= 2 text rule, changed based on prompt which only mentions >=2 tables
                    # **Correction:** The prompt *does* imply text use "You also need to use textual evidences"
                    # **Re-Correction based on user's filter:** The user *implemented* a filter for >=2 text. Let's stick to the user's implemented filter.
                    if text_count >= 2: # Reinstating the user's implemented text filter rule
                        logging.debug(f"Filter PASSED for Q{result['seed_id']}-V{result['variant_id']} (Tables: {table_count}, Texts: {text_count})")
                        json.dump(result, out_f, ensure_ascii=False, indent=4)
                        out_f.write("\n")
                        filtered_count += 1
                    else:
                        logging.info(f"Filter SKIPPED for Q{result['seed_id']}-V{result['variant_id']} - Insufficient text evidence (Tables: {table_count}, Texts: {text_count})")

                else:
                     logging.info(f"Filter SKIPPED for Q{result['seed_id']}-V{result['variant_id']} - Insufficient table evidence (Tables: {table_count}, Texts: {text_count})")

    except IOError as e:
        logging.error(f"❌ Error writing to output file {OUTPUT_FILE}: {e}")
        return
    except Exception as e:
         logging.error(f"❌ Unexpected error during file writing: {e}")
         return


    logging.info(f"\n🎉 All done! {filtered_count} filtered results saved to: {OUTPUT_FILE}")


# ========== 6. Startup ==========
if __name__ == "__main__":
    asyncio.run(main())