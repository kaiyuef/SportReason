#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunk long text docs in corpus_text.jsonl into ≤MAX_TOKENS chunks
保持字段格式:  {"id","original_id","content"}

依赖:
    pip install tqdm transformers spacy
    python -m spacy download en_core_web_sm
"""

from pathlib import Path
import json, re, logging, itertools
from tqdm import tqdm
from transformers import AutoTokenizer
import spacy

# ---------------- 参数 ----------------
MAX_TOKENS_PER_CHUNK = 100   # 单 chunk 最大 token
OVERLAP_TOKENS       = 0     # 相邻 chunk token 重叠
TOKENIZER_MODEL      = "BAAI/bge-m3"
INPUT_FILE           = "regenerate_combine/regenerated_corpus/expanded_corpus_text.jsonl"
OUTPUT_FILE          = "regenerate_combine/regenerated_corpus/expanded_corpus_text_chunked.jsonl"

# ---------------- 初始化 ----------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
nlp       = spacy.load("en_core_web_sm", disable=["ner","tagger","parser"])  # 只需 sentence splitter
nlp.add_pipe("sentencizer")

# ---------------- 工具 ----------------
def jsonl_iter(path: str):
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                try:
                    yield json.loads(ln)
                except Exception as e:
                    logging.warning(f"跳过无法解析行: {e}")

def save_jsonl(lst, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for obj in lst:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def sentences(text: str):
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]

def token_len(txt: str):
    return len(tokenizer(txt, add_special_tokens=False)["input_ids"])

# ---------------- 分块 ----------------
def chunk_text(entry, max_tok, overlap_tok):
    sents = sentences(entry["content"])
    chunks, cur, cur_tok = [], [], 0

    for sent in sents:
        l = token_len(sent)
        if l == 0: continue
        # 若加当前句超限 -> 先保存现有 chunk
        if cur and cur_tok + l > max_tok:
            chunks.append((" ".join(cur), cur_tok))
            # overlap: 把末尾若干 token 句子重新放入新 chunk
            if overlap_tok > 0:
                overlap_text = tokenizer.decode(
                    tokenizer(" ".join(cur), add_special_tokens=False)["input_ids"][-overlap_tok:]
                )
                cur, cur_tok = [overlap_text], token_len(overlap_text)
            else:
                cur, cur_tok = [], 0
        # 追加句子
        cur.append(sent)
        cur_tok += l

    if cur:
        chunks.append((" ".join(cur), cur_tok))
    return [c[0] for c in chunks]

# ---------------- 主流程 ----------------
def main():
    dataset = [d for d in jsonl_iter(INPUT_FILE)
               if isinstance(d, dict) and "id" in d and "content" in d and d["content"].strip()]

    output = []
    for entry in tqdm(dataset, desc="Chunking"):
        chunks = chunk_text(entry, MAX_TOKENS_PER_CHUNK, OVERLAP_TOKENS)
        for idx, ch_text in enumerate(chunks, 1):
            output.append({
                "id"         : f"{entry['id']}_chunk_{idx:02d}",
                "title"      : entry["title"],
                "original_id": entry["id"],
                "content"    : ch_text,
                "url"        : entry["url"],
            })

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(output, OUTPUT_FILE)
    print(f"✅ 完成！共写入 {len(output)} 个 chunk → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
