#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align gold_evidences to corpus by title‑filtered FAISS + BGE embedding.

• 只在 gold.title 对应的 corpus 子集里检索
• 不做跨页面的模糊匹配
• 标题组内相似度 ≥ SIM_THRESHOLD 即替换 id
• 否则创建新文档（唯一 id）并同步索引
• 同步更新 sample["gold_evidence_ids"]
"""

import json, re, itertools, logging, numpy as np, faiss
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------- 配置 ----------
MODEL_NAME     = "BAAI/bge-m3"
SIM_THRESHOLD  = 0.60
LOG_FILE       = "gold_evidence_bge_title_filtered.log"

DS_IN  = "numsports/regenerate_combine/reformatted_merged_dataset.jsonl"
TXT_IN = "numsports/regenerate_combine/regenerated_corpus/corpus_text_chunked.jsonl"
TBL_IN = "numsports/regenerate_combine/regenerated_corpus/corpus_tables.jsonl"
INF_IN = "numsports/regenerate_combine/regenerated_corpus/corpus_infobox.jsonl"

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ---------- IO ----------
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip(): yield json.loads(ln)

def save_jsonl(lst, path):
    with open(path, "w", encoding="utf-8") as f:
        for obj in lst:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------- 文本预处理 ----------
def flatten(o):
    if isinstance(o, dict):
        return " ".join(itertools.chain.from_iterable((k, flatten(v)) for k, v in o.items()))
    if isinstance(o, list):
        return " ".join(flatten(x) for x in o)
    return str(o)

def clean(t: str):
    return re.sub(r"\s+", " ", t).strip()

# ---------- 嵌入 + 索引 ----------
model = SentenceTransformer(MODEL_NAME)
def embed(texts):
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def build_index(mat):
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat)
    return idx

def ids_by_title(corpus):
    m = {}
    for i, doc in enumerate(corpus):
        title = doc.get("title", "")
        m.setdefault(title, []).append(i)
    return m

# ---------- 核心对齐 ----------
def kind_of(gold):
    if "type" in gold:
        return gold["type"].lower()
    gid = gold.get("id", "")
    if gid.startswith("text"):   return "text"
    if gid.startswith("table"):  return "table"
    return "infobox"

def new_id(kind, corpus):
    return f"{kind}_{len(corpus):06d}"

def sync_ids(sample, old, new):
    if "gold_evidence_ids" in sample:
        sample["gold_evidence_ids"] = [
            new if x == old else x
            for x in sample["gold_evidence_ids"]
        ]

def align(dataset, text, table, info):
    # ---------- 1) 预计算三类语料向量 ----------
    vec_text  = embed([clean(flatten(d["content"])) for d in text]).astype("float32")
    vec_table = embed([clean(flatten(d["content"])) for d in table]).astype("float32")
    vec_info  = embed([clean(flatten(d["content"])) for d in info]).astype("float32")

    # ---------- 2) 建立 “title → 局部 FAISS 索引” ----------
    idx_pack = {}
    for kind, corpus, mat in [
        ("text",   text,  vec_text),
        ("table",  table, vec_table),
        ("infobox", info, vec_info),
    ]:
        by_title   = ids_by_title(corpus)
        title_idx  = {}
        for title, idxs in by_title.items():
            submat = mat[np.array(idxs)]
            if submat.size == 0:
                continue
            dim      = submat.shape[1]
            idx_obj  = faiss.IndexFlatIP(dim)      # <<< BUG FIX：add() 之后不再用 “or” 语句
            idx_obj.add(submat)                    # add() 本身返回 None
            title_idx[title] = (idx_obj, np.array(idxs))
        idx_pack[kind] = (corpus, title_idx)

    # ---------- 3) 对齐每条 sample ----------
    for sample in tqdm(dataset, desc="Aligning by title"):
        for gold in sample.get("gold_evidences", []):
            kind            = kind_of(gold)
            corpus, t_index = idx_pack[kind]
            old_id          = gold.get("id", "")
            title           = gold.get("title", "")

            # ----- 3A) 同标题检索 -----
            if title in t_index:
                idx_obj, idxs = t_index[title]

                gold_vec      = embed([clean(flatten(gold["content"]))]).astype("float32")
                D, I          = idx_obj.search(gold_vec, 1)

                if D[0, 0] >= SIM_THRESHOLD:
                    # 命中 —— 复用旧文档 id
                    new_id_val             = corpus[int(idxs[I[0, 0]])]["id"]
                    gold["id"]             = new_id_val             # ✅ 替换 gold_evidences 中的 id
                    sync_ids(sample, old_id, new_id_val)            # ✅ 替换 gold_evidence_ids 列表
                    logging.info(f"[{kind}] title='{title}' matched id={new_id_val} sim={D[0,0]:.3f}")
                    continue

            # ----- 3B) 未命中 —— 新建文档 -----
            new_id_val = new_id(kind, corpus)
            corpus.append({
                "id":      new_id_val,
                "title":   title or f"auto-{new_id_val}",
                "url":     gold.get("url") or gold.get("meta", {}).get("url", ""),
                "content": gold["content"],
                "added_by":"bge-faiss-title"
            })
            gold["id"] = new_id_val
            sync_ids(sample, old_id, new_id_val)
            logging.info(f"[{kind}] NEW doc title='{title}' => id={new_id_val}")

            # 增量写入索引
            vec_new  = embed([clean(flatten(gold["content"]))]).astype("float32")
            if title not in t_index:
                dim       = vec_new.shape[1]
                idx_obj   = faiss.IndexFlatIP(dim)
                t_index[title] = (idx_obj, np.array([], dtype=int))
            idx_obj, idxs = t_index[title]
            idx_obj.add(vec_new)
            t_index[title] = (idx_obj, np.append(idxs, len(corpus) - 1))

    return dataset, text, table, info


# ---------- 执行 ----------
dataset      = list(read_jsonl(DS_IN))
text_corpus  = list(read_jsonl(TXT_IN))
table_corpus = list(read_jsonl(TBL_IN))
info_corpus  = list(read_jsonl(INF_IN))

dataset, text_corpus, table_corpus, info_corpus = align(
    dataset, text_corpus, table_corpus, info_corpus
)

# ---------- 保存 ----------
save_jsonl(text_corpus,  TXT_IN.replace(".jsonl", "_bge.jsonl"))
save_jsonl(table_corpus, TBL_IN.replace(".jsonl", "_bge.jsonl"))
save_jsonl(info_corpus,  INF_IN.replace(".jsonl", "_bge.jsonl"))
save_jsonl(dataset,      DS_IN.replace(".jsonl", "_bge.jsonl"))

print("✅ 完成 title‑filtered FAISS+BGE 对齐，所有 evidence_id 已同步。")
