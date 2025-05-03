import os
import json
import asyncio
import aiofiles
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = AsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

async def read_jsonl(file_path):
    data = []
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        async for line in f:
            data.append(json.loads(line.strip()))
    return data

async def classify_evidences_with_retry(question, evidences, max_retries=3, delay=2):
    last_exception = None
    for attempt in range(max_retries):
        try:
            return await classify_evidences(question, evidences)
        except Exception as e:
            last_exception = e
            print(f"⚠️ Retry {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay * (attempt + 1))  # 可选：递增等待时间
    raise last_exception

# Classify evidence with reasoning, considering them as a whole
async def classify_evidences(question, evidences):
    evidence_list = []
    evidence_map = {}  # index to {id, text}

    for idx, evidence in enumerate(evidences):
        content = evidence.get("content", "")
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False)
        text = str(content)
        evidence_list.append(f"[{idx}] {text}")
        evidence_map[str(idx)] = {
            "id": evidence.get("id", f"evidence_{idx}"),
            "text": text
        }

    all_evidence_text = "\n".join(evidence_list)

    prompt = f"""
You are given a user question and a list of evidence documents.

Your task is to:
1. Identify which evidences are useful in answering the question (considering they may be useful together).
2. Mark each evidence as either 'useful' or 'useless'.
3. Provide a reason for each classification.

Return your answer in strict JSON format. Here's the exact format to use:

{{
  "useful": [
    {{ "index": 0, "reason": "Reason why this evidence is useful." }},
    ...
  ],
  "useless": [
    {{ "index": 1, "reason": "Reason why this evidence is not useful." }},
    ...
  ]
}}

Only return the JSON. Do not include any explanation or commentary outside the JSON.

Question:
{question}

Evidence List:
{all_evidence_text}
"""

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies and explains the usefulness of evidence."},
            {"role": "user", "content": prompt}
        ]
    )

    raw_response = response.choices[0].message.content.strip()

    # Clean up markdown-style code blocks like ```json ... ```
    if raw_response.startswith("```json"):
        raw_response = raw_response[7:].strip()
    elif raw_response.startswith("```"):
        raw_response = raw_response[3:].strip()
    if raw_response.endswith("```"):
        raw_response = raw_response[:-3].strip()

    if not raw_response:
        raise ValueError("Empty response from OpenAI")

    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format from OpenAI: {e}\nResponse:\n{raw_response}")

    useful = [
        {
            "id": evidence_map[str(item["index"])]["id"],
            "text": evidence_map[str(item["index"])]["text"],
            "reason": item["reason"]
        } for item in result.get("useful", [])
    ]
    useless = [
        {
            "id": evidence_map[str(item["index"])]["id"],
            "text": evidence_map[str(item["index"])]["text"],
            "reason": item["reason"]
        } for item in result.get("useless", [])
    ]

    return useful, useless, raw_response  # return response for debug logging

async def main(jsonl_file):
    data = await read_jsonl(jsonl_file)
    results = []

    for idx, item in enumerate(data[200:400]):
        question = item.get("seed_question", "")
        evidences = item.get("gold_evidences", [])

        if not question or not evidences:
            continue

        print(f"Processing item {idx}...")

        try:
            useful, useless, raw_response = await classify_evidences_with_retry(question, evidences)
        except Exception as e:
            print(f"❌ Error after retries on item {idx}: {e}")
            async with aiofiles.open(f"error_log_{idx}.txt", "w", encoding="utf-8") as f:
                await f.write(f"Question:\n{question}\n\nError:\n{str(e)}\n\nRaw Response:\n{raw_response if 'raw_response' in locals() else 'N/A'}")
            continue

        results.append({
            "id": item.get("id", idx),
            "useful": useful,
            "useless": useless
        })

    # Save results
    async with aiofiles.open("evidence_usefulness_grouped.json", "w", encoding="utf-8") as f:
        await f.write(json.dumps(results, ensure_ascii=False, indent=4))

    print("✅ Completed evidence classification with retry mechanism.")

if __name__ == "__main__":
    jsonl_file = "merged_dataset_updated.jsonl"
    asyncio.run(main(jsonl_file))
