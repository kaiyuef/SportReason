#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-build Chroma indexes under
    numsports/indexes/{model_safe}/{corpus_name}_index
using *exactly* the embedding model passed via --model_name.

🆕 2025-05-11 修订
──────────────────
* 修复 EmbeddingFunction 返回值类型：去除 `convert_to_numpy=False`，使用默认 numpy 输出并 `.tolist()`。
* 限制 `max_seq_len` 不超过模型最大 position embeddings，避免超长输入。
* 兼容较旧 sentence-transformers，无 quantization/chunking。
"""

from __future__ import annotations

import argparse, json, re, itertools, os, gc
from pathlib import Path
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction
import chromadb
from tqdm import tqdm

os.environ.setdefault("HF_HOME", "/scratch/kf2365/.cache")

MAX_TEXT_LENGTH = 1000  # safety truncation (chars)

# ───────────────────── Embedding Function ──────────────────────
class MyEmbeddingFunction(EmbeddingFunction):
    """Thin wrapper so ChromaDB can call model.encode lazily."""

    def __init__(self, model: SentenceTransformer, batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, inputs: List[str]):
        # 默认返回 numpy.ndarray，可直接 tolist()
        with torch.inference_mode():
            embeddings = self.model.encode(
                inputs,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        return embeddings.tolist()

# ───────────────────── Utility helpers ─────────────────────────

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line := line.strip():
                yield json.loads(line)

def flatten(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " ".join(itertools.chain.from_iterable((k, flatten(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return " ".join(flatten(x) for x in obj)
    return str(obj)

clean = lambda text: re.sub(r"\s+", " ", flatten(text)).strip()

# ─────────────────────── Main routine ──────────────────────────

def main(args: argparse.Namespace):
    corpus_root   = Path(args.base_corpus_dir)
    persist_root  = Path(args.base_persist_dir)

    model_name = args.model_name
    model_safe = model_name.replace("/", "_")

    # ─── Load model ───
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    if device == "cuda":
        st_model.half()

    # Clamp max_seq_length to model's tokenizer max
    try:
        model_max_tok = st_model.tokenizer.model_max_length
    except AttributeError:
        model_max_tok = getattr(st_model, 'max_seq_length', None) or 512
    if args.max_seq_len > model_max_tok:
        print(f"[WARN] --max_seq_len {args.max_seq_len} exceeds model tokenizer max {model_max_tok}, using {model_max_tok}")
    st_model.max_seq_length = min(args.max_seq_len, model_max_tok)

    emb_dim  = st_model.get_sentence_embedding_dimension()
    embed_fn = MyEmbeddingFunction(st_model, batch_size=args.embed_batch_size)

    # ─── Walk through corpus dirs ───
    for corpus_dir in sorted(corpus_root.iterdir()):
        if not corpus_dir.is_dir():
            continue
        corpus_name = corpus_dir.name
        print(f"\n=== 构建索引: '{corpus_name}' ===")

        persist_dir = persist_root / model_safe / f"{corpus_name}_index"
        persist_dir.mkdir(parents=True, exist_ok=True)

        client   = chromadb.PersistentClient(path=str(persist_dir))
        col_name = f"{corpus_name}_corpus"

        try:
            collection = client.get_collection(name=col_name)
            if collection.metadata.get("dimension") != emb_dim:
                print(f"[WARN] 维度不一致，重建 collection '{col_name}'")
                client.delete_collection(name=col_name)
                raise chromadb.errors.NotFoundError
        except (ValueError, chromadb.errors.NotFoundError):
            collection = client.get_or_create_collection(
                name=col_name,
                embedding_function=embed_fn,
                metadata={"hnsw:space": "cosine", "dimension": emb_dim},
            )

        buf_txt, buf_id, buf_meta = [], [], []
        total = 0

        jsonl_files = list(corpus_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"⚠️  跳过空目录: {corpus_name}")
            continue

        for jf in jsonl_files:
            for row in read_jsonl(jf):
                # char-level safety truncate; token truncation via max_seq_length
                txt = clean(row.get("contents", ""))[:MAX_TEXT_LENGTH]
                buf_txt.append(txt)
                buf_id.append(row.get("id"))
                buf_meta.append({"title": row.get("title", ""), "url": row.get("url", "")})

                if len(buf_id) >= args.buffer_size:
                    collection.add(ids=buf_id, documents=buf_txt, metadatas=buf_meta)
                    total += len(buf_id)
                    print(f"🟢 已插入 {total} 条...")
                    buf_txt, buf_id, buf_meta = [], [], []

        if buf_id:  # flush remainder
            collection.add(ids=buf_id, documents=buf_txt, metadatas=buf_meta)
            total += len(buf_id)

        print(f"✅ 完成 '{corpus_name}' ➜ 总共插入 {total} 条记录")
        torch.cuda.empty_cache(); gc.collect()

# ────────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("Build Chroma index from JSONL corpora")
    p.add_argument("--base_corpus_dir",  default="numsports/corpus")
    p.add_argument("--base_persist_dir", default="numsports/indexes")
    p.add_argument("--model_name",       default="infly/inf-retriever-v1")
    p.add_argument("--embed_batch_size", type=int, default=64)
    p.add_argument("--buffer_size",      type=int, default=5000)
    p.add_argument("--max_seq_len",      type=int, default=512, help="Override model max_seq_length if needed.")
    main(p.parse_args())
