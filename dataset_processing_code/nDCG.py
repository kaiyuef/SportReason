import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, ndcg_score

def normalize_id(doc_id):
    """Removes chunk suffix from document ID."""
    return doc_id.split('_chunk')[0]

def calculate_metrics(retrieved_ids, gold_ids):
    """
    Calculate accuracy, F1-score, and nDCG based on gold evidence presence in retrieved results.

    Args:
        retrieved_ids (list): List of retrieved evidence IDs.
        gold_ids (list): List of gold evidence IDs.

    Returns:
        tuple: Accuracy, F1-score, and nDCG score.
    """
    # Normalize IDs for comparison
    retrieved_ids = [normalize_id(doc_id) for doc_id in retrieved_ids]
    gold_ids = [normalize_id(doc_id) for doc_id in gold_ids]
    
    # Success if any gold ID appears in retrieved results
    y_true = [1] * len(gold_ids)
    y_pred = [1 if gold_id in retrieved_ids else 0 for gold_id in gold_ids]
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # Compute nDCG with consideration of order in both gold and retrieved lists
    relevance = np.zeros(len(retrieved_ids))
    gold_rank = {doc_id: len(gold_ids) - i for i, doc_id in enumerate(gold_ids)}  # Higher weight for earlier gold docs
    
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in gold_rank:
            relevance[i] = gold_rank[doc_id] / (i + 1)  # Adjusting by retrieval order
    
    relevance = relevance.reshape(1, -1)  # Reshape for sklearn
    ideal_relevance = sorted(relevance[0], reverse=True)
    ndcg = ndcg_score([ideal_relevance], relevance) if len(relevance[0]) > 0 else 0.0
    
    return acc, f1, ndcg

def process_files(retrieved_file, gold_file, output_file):
    """
    Process the retrieved evidence and gold evidence files, compute ACC, F1, and nDCG for each entry.
    Saves detailed results to a JSONL file.

    Args:
        retrieved_file (str): Path to the JSON file containing retrieved evidence IDs.
        gold_file (str): Path to the JSONL file containing gold evidence data with 'gold_evidences'.
        output_file (str): Path to save detailed matching results.
    """
    # Load the datasets
    with open(retrieved_file, 'r') as f:
        retrieved_data = json.load(f)

    with open(gold_file, 'r') as f:
        gold_data = [json.loads(line) for line in f]

    # Create a mapping from ID to gold evidence contents
    gold_mapping = {
        entry['id']: [evidence['id'] for evidence in entry.get('gold_evidences', [])]
        for entry in gold_data
    }

    scores = []
    
    for entry in retrieved_data:  # Process only the first 200 entries
        question_id = entry['id']
        retrieved_ids = entry['pipeline_output']['retrieved_evidence_ids']
        gold_ids = gold_mapping.get(question_id, [])

        matched = [doc_id for doc_id in retrieved_ids if normalize_id(doc_id) in gold_ids]
        unmatched = [doc_id for doc_id in retrieved_ids if normalize_id(doc_id) not in gold_ids]
        
        if gold_ids:
            acc, f1, ndcg = calculate_metrics(retrieved_ids, gold_ids)
            result = {
                'question_id': question_id,
                'accuracy': acc,
                'f1_score': f1,
                'ndcg': ndcg,
                'matched': matched,
                'unmatched': unmatched
            }
        else:
            result = {
                'question_id': question_id,
                'accuracy': None,
                'f1_score': None,
                'ndcg': None,
                'matched': [],
                'unmatched': retrieved_ids
            }
        
        scores.append(result)
    
    with open(output_file, 'w') as f:
        for entry in scores:
            f.write(json.dumps(entry) + '\n')

    return scores

# Example usage
retrieved_file = 'bge_huggingface-qwen.json'  # Path to your retrieved evidence JSON file
gold_file = 'merged_dataset_updated.jsonl'      # Path to your gold evidence JSONL file
output_file = 'detailed_results.jsonl'          # Path to save matching details

results = process_files(retrieved_file, gold_file, output_file)

# Calculate average and variance of ACC, F1, and nDCG scores
valid_acc = [entry['accuracy'] for entry in results if entry['accuracy'] is not None]
valid_f1 = [entry['f1_score'] for entry in results if entry['f1_score'] is not None]
valid_ndcg = [entry['ndcg'] for entry in results if entry['ndcg'] is not None]

average_acc = np.mean(valid_acc)
variance_acc = np.var(valid_acc)

average_f1 = np.mean(valid_f1)
variance_f1 = np.var(valid_f1)

average_ndcg = np.mean(valid_ndcg)
variance_ndcg = np.var(valid_ndcg)

print("Evaluation completed. Detailed results saved to 'detailed_results.jsonl'.")
print(f"Average Accuracy: {average_acc:.4f}")
print(f"Variance of Accuracy: {variance_acc:.4f}")
print(f"Average F1-score: {average_f1:.4f}")
print(f"Variance of F1-score: {variance_f1:.4f}")
print(f"Average nDCG: {average_ndcg:.4f}")
print(f"Variance of nDCG: {variance_ndcg:.4f}")
