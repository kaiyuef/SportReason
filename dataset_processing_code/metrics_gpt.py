import json
import numpy as np
import asyncio
from openai import AsyncOpenAI
from sklearn.metrics import accuracy_score, f1_score

client = AsyncOpenAI(api_key="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")  # 替换为你的 API key

def normalize_id(doc_id):
    return doc_id.split('_chunk')[0]

def calculate_metrics(retrieved_ids, gold_ids):
    retrieved_ids = [normalize_id(doc_id) for doc_id in retrieved_ids]
    gold_ids = [normalize_id(doc_id) for doc_id in gold_ids]
    y_true = [1] * len(gold_ids)
    y_pred = [1 if gold_id in retrieved_ids else 0 for gold_id in gold_ids]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    return acc, f1

async def evaluate_answer(question, generated_answer, seed_answers, client):
    evaluation_prompt = f"""
    User question: {question}
    GPT-4o response: {generated_answer}
    Reference answers: {json.dumps(seed_answers, ensure_ascii=False)}
    
    Please determine whether GPT-4o's response is correct and provide an explanation.
    
    Your response should be either: "Correct" or "Incorrect" followed by a reason.
    """
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant evaluating responses."},
            {"role": "user", "content": evaluation_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

async def process_files(retrieved_file, gold_file, output_file):
    with open(retrieved_file, 'r') as f:
        retrieved_data = json.load(f)
    with open(gold_file, 'r') as f:
        gold_data = [json.loads(line) for line in f]

    gold_mapping = {}
    qa_mapping = {}
    question_mapping = {}
    for entry in gold_data:
        key = str(entry.get('id'))
        if key:
            seen_ids = set()
            normalized_ids = []
            for evidence in entry.get('gold_evidences', []):
                eid = evidence.get('id')
                if eid:
                    norm_id = normalize_id(eid)
                    if norm_id not in seen_ids:
                        seen_ids.add(norm_id)
                        normalized_ids.append(norm_id)
            gold_mapping[key] = normalized_ids
            answers = entry.get('answers')
            if not answers:
                seed_answer = entry.get('seed_answers', '')
                answers = [seed_answer] if isinstance(seed_answer, str) and seed_answer else []
            qa_mapping[key] = answers
            question_mapping[key] = entry.get('question', '')

    scores = []
    for example_id, entry in retrieved_data.items():
        retrieved_ids = [normalize_id(doc['id']) for doc in entry.get('retrieved_docs', [])]
        gold_ids = gold_mapping.get(example_id, [])
        matched = [doc_id for doc_id in gold_ids if doc_id in retrieved_ids]
        unmatched = [doc_id for doc_id in gold_ids if doc_id not in retrieved_ids]
        if gold_ids:
            acc, f1 = calculate_metrics(retrieved_ids, gold_ids)
        else:
            acc, f1 = None, None

        retrieved_answer = entry.get('answer', '')
        gold_answers = qa_mapping.get(example_id, [])
        question = question_mapping.get(example_id, '')
        
        if gold_answers:
            retrieved_answer_padded = f" {retrieved_answer} "
            qa_match = any(f" {ans} " in retrieved_answer_padded for ans in gold_answers)
            gpt_result = await evaluate_answer(question, retrieved_answer, gold_answers, client)
            gpt_qa_match = gpt_result.lower().startswith("correct")
        else:
            qa_match = None
            gpt_qa_match = None

        num_evidence = len(gold_ids)
        if num_evidence <= 2:
            difficulty = 'easy'
        elif num_evidence <= 5:
            difficulty = 'medium'
        else:
            difficulty = 'hard'

        result = {
            'id': example_id,
            'accuracy': acc,
            'f1_score': f1,
            'matched': matched,
            'unmatched': unmatched,
            'qa_match': qa_match,
            'gpt_qa_match': gpt_qa_match,
            'retrieved_answer': retrieved_answer,
            'gold_answers': gold_answers,
            'gold_evidence_ids': gold_ids,
            'is_correct': qa_match == True,
            'difficulty': difficulty
        }
        scores.append(result)

    with open(output_file, 'w') as f:
        for entry in scores:
            f.write(json.dumps(entry, ensure_ascii=False, indent=4) + '\n')

    return scores

async def main():
    retrieved_file = 'bm25_huggingface-qwen_top50.json'
    gold_file = 'merged_dataset_updated.jsonl'
    output_file = 'bm25_huggingface-qwen_results.jsonl'

    results = await process_files(retrieved_file, gold_file, output_file)

    # 分类统计
    categorized_results = {"easy": [], "medium": [], "hard": []}
    for entry in results:
        difficulty = entry.get('difficulty')
        if difficulty in categorized_results:
            categorized_results[difficulty].append(entry)

    for diff in ['easy', 'medium', 'hard']:
        entries = categorized_results[diff]
        valid_acc = [e['accuracy'] for e in entries if e['accuracy'] is not None]
        valid_f1 = [e['f1_score'] for e in entries if e['f1_score'] is not None]
        qa_matches = [e['qa_match'] for e in entries if e['qa_match'] is not None]

        if valid_acc:
            avg_acc = np.mean(valid_acc)
            var_acc = np.var(valid_acc)
        else:
            avg_acc = None
            var_acc = None

        if valid_f1:
            avg_f1 = np.mean(valid_f1)
            var_f1 = np.var(valid_f1)
        else:
            avg_f1 = None
            var_f1 = None

        if qa_matches:
            qa_accuracy = sum(qa_matches) / len(qa_matches)
        else:
            qa_accuracy = None

        print(f"Category: {diff}")
        if avg_acc is not None:
            print(f"  Evidence - Average Accuracy: {avg_acc:.4f}")
            print(f"  Evidence - Variance of Accuracy: {var_acc:.4f}")
        else:
            print("  Evidence - No valid accuracy data.")
        if avg_f1 is not None:
            print(f"  Evidence - Average F1-score: {avg_f1:.4f}")
            print(f"  Evidence - Variance of F1-score: {var_f1:.4f}")
        else:
            print("  Evidence - No valid F1-score data.")
        if qa_accuracy is not None:
            print(f"  QA Accuracy (string in check): {qa_accuracy:.4f}")
        else:
            print("  No valid QA answer data.")

    print("\nSample Count by Difficulty:")
    for diff in ['easy', 'medium', 'hard']:
        print(f"  {diff.capitalize()}: {len(categorized_results[diff])}")

# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())
