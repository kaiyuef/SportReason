import json
import re
import numpy as np
import math
from collections import Counter
from sentence_transformers import SentenceTransformer, util

# ─── Setup ────────────────────────────────────────────────────────────────────
try:
    qa_embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model: {e}")


def extract_boxed_answers(text):
    if not isinstance(text, str):
        return []
    return re.findall(r'\\(?:boxed|boxedanswer){(.*?)}', text)


def semantic_match(pred, golds, threshold=0.85):
    if not pred or not golds:
        return False, 0.0
    emb_pred = qa_embedding_model.encode(str(pred), normalize_embeddings=True, show_progress_bar=False)
    best = 0.0
    for g in golds:
        sim = util.cos_sim(
            emb_pred,
            qa_embedding_model.encode(str(g), normalize_embeddings=True, show_progress_bar=False)
        ).item()
        best = max(best, sim)
    return best >= threshold, best


def compute_retrieval_metrics_at_k(retrieved, gold, k=14):
    topk = retrieved[:k]
    rels = [1 if d in gold else 0 for d in topk]
    # nDCG@k
    dcg  = sum(r / math.log2(i+2) for i, r in enumerate(rels))
    idcg = sum(1.0 / math.log2(i+2) for i in range(min(len(gold), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    # precision/recall/f1@k
    hits = sum(rels)
    prec = hits / k
    rec  = hits / len(gold) if gold else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    # EM@k: all gold in top-k?
    em   = 1.0 if set(gold).issubset(set(topk)) else 0.0
    return ndcg, prec, rec, f1, em


# ─── Core processing ──────────────────────────────────────────────────────────
def process_files(retrieved_file, gold_file, slice_start=None, slice_end=None):
    with open(retrieved_file, 'r', encoding='utf-8') as f:
        retrieved_data = json.load(f)
    items = list(retrieved_data.items())
    if slice_start is not None and slice_end is not None:
        items = items[slice_start:slice_end]

    gold_data = [json.loads(l) for l in open(gold_file, 'r', encoding='utf-8') if l.strip()]

    # build maps
    gold_evid_map, qa_map, ev_count_map = {}, {}, {}
    for e in gold_data:
        k = str(e['id'])
        seen, lst = set(), []
        for ev in e.get('gold_evidences', []):
            eid = ev.get('id')
            if eid and eid not in seen:
                seen.add(eid); lst.append(eid)
        gold_evid_map[k] = lst
        ans = e.get('answers') or e.get('seed_answers', [])
        qa_map[k] = [str(a) for a in (ans if isinstance(ans, list) else [ans]) if a]
        types = e.get('gold_evidence_type', {})
        ev_count_map[k] = (types.get('table',0), types.get('text',0))

    scores = []
    for qid, entry in items:
        qid = str(qid)
        all_ret = [d['id'] for d in entry.get('retrieved_docs', []) if d.get('id')]
        topk   = all_ret[:14]
        gold_ids = gold_evid_map.get(qid, [])

        ndcg, prec, rec, f1, em = compute_retrieval_metrics_at_k(topk, gold_ids, k=14)

        raw   = entry.get('answer','')
        boxed = extract_boxed_answers(raw)
        pred  = boxed[0] if boxed else ''
        gold_ans = qa_map.get(qid, [])
        qa_em, qa_sim = semantic_match(pred, gold_ans)

        diff = 'easy' if len(gold_ids)<=1 else 'medium' if len(gold_ids)<=3 else 'hard'
        t_cnt, x_cnt = ev_count_map.get(qid, (0,0))
        if   t_cnt==0 and x_cnt>0:     cat='pure_text'
        elif x_cnt==0 and t_cnt>0:     cat='pure_table'
        elif t_cnt==1 and x_cnt>0:     cat='single_table+text'
        elif t_cnt>1 and x_cnt>0:      cat='multi_table+text'
        else:                          cat='other'

        scores.append({
            'id':                 qid,
            'difficulty':         diff,
            'evidence_category':  cat,
            'nDCG@14':            ndcg,
            'Precision@14':       prec,
            'Recall@14':          rec,
            'F1@14':              f1,
            'EM@14':              em,
            'QA_EM':              qa_em,
            'QA_sim':             qa_sim,
            'predicted_answer':   pred,
            'gold_answers':       gold_ans,
            'retrieved_ids':      topk,
            'gold_ids':           gold_ids,
        })

    return scores


# ─── Summaries ────────────────────────────────────────────────────────────────
def summarize_by_difficulty(scores):
    print("\n=== By Difficulty ===")
    for diff in ['easy','medium','hard']:
        sub = [r for r in scores if r['difficulty']==diff]
        if not sub: continue
        print(f"{diff.capitalize()} (n={len(sub)}) → "
              f"nDCG@14: {np.mean([r['nDCG@14'] for r in sub]):.4f}, "
              f"P@14: {np.mean([r['Precision@14'] for r in sub]):.4f}, "
              f"R@14: {np.mean([r['Recall@14'] for r in sub]):.4f}, "
              f"F1@14: {np.mean([r['F1@14'] for r in sub]):.4f}, "
              f"EM@14: {np.mean([r['EM@14'] for r in sub]):.4f}")


def summarize_by_category(scores):
    print("\n=== By Evidence Category ===")
    cats = ['pure_text','pure_table','single_table+text','multi_table+text']
    for cat in cats:
        sub = [r for r in scores if r['evidence_category']==cat]
        if not sub:
            print(f"{cat:20s}: n=0")
            continue
        print(f"{cat:20s} (n={len(sub)}) → "
              f"nDCG@14: {np.mean([r['nDCG@14'] for r in sub]):.4f}, "
              f"P@14: {np.mean([r['Precision@14'] for r in sub]):.4f}, "
              f"R@14: {np.mean([r['Recall@14'] for r in sub]):.4f}, "
              f"F1@14: {np.mean([r['F1@14'] for r in sub]):.4f}, "
              f"EM@14: {np.mean([r['EM@14'] for r in sub]):.4f}")


def summarize_by_retrieval_recall_bins(scores):
    print("\n=== QA EM by Retrieval Recall@14 Bins ===")
    bins = [(0,25),(25,50),(50,75),(75,100)]
    for low, high in bins:
        sub = [r for r in scores if low <= r['Recall@14']*100 < high]
        n = len(sub)
        if n == 0:
            print(f"{low:2d}-{high:3d}%: n=0")
            continue
        correct = sum(1 for r in sub if r['QA_EM'])
        rate = correct / n
        print(f"{low:2d}-{high:3d}%: n={n}, QA_EM rate={rate:.4f}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    retrieved_file = 'answer_results/chroma_qwen.json'
    gold_file      = 'dataset/merged_dataset_final.jsonl'
    start_idx      = 0
    end_idx        = 200

    scores = process_files(retrieved_file, gold_file, start_idx, end_idx)

    # write detailed JSONL
    with open('evaluation_results/eval_with_all_metrics.jsonl','w', encoding='utf-8') as out:
        for r in scores:
            out.write(json.dumps(r, ensure_ascii=False, indent=4) + '\n')

    print("=== Overall Summary ===")
    summarize_by_difficulty(scores)
    summarize_by_category(scores)
    summarize_by_retrieval_recall_bins(scores)
