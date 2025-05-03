#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-build Chroma indexes under
numsports/indexes/{model_safe}/{type}_index,
using the *exact* embedding model passed via --model_name.
"""
import argparse, json, re, itertools, os, shutil
from pathlib import Path

import torch, chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

BATCH_SIZE = 5_000
os.environ.setdefault("TRANSFORMERS_CACHE", "/scratch/kf2365/.cache")

# ---------------------------- Utils --------------------------
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

def flatten(o):
    if isinstance(o, str):
        return o
    if isinstance(o, dict):
        return " ".join(itertools.chain.from_iterable((k, flatten(v)) for k, v in o.items()))
    if isinstance(o, list):
        return " ".join(flatten(x) for x in o)
    return str(o)

clean = lambda t: re.sub(r"\s+", " ", flatten(t)).strip()

# ---------------------------- 主流程 --------------------------
def main(args):
    base_corpus_dir  = Path(args.base_corpus_dir)
    base_persist_dir = Path(args.base_persist_dir)
    model_name       = args.model_name
    model_safe       = model_name.replace("/", "_")

    # ── 加载 ST 模型一次，用来计算所有向量 ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    emb_dim  = st_model.get_sentence_embedding_dimension()

    # 本地 embedding_function：用同一模型，防止 Chroma 默认挂 MiniLM
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device=device,
        trust_remote_code=True,
        normalize_embeddings=True,
    )

    for sub in sorted(base_corpus_dir.iterdir()):
        if not sub.is_dir():
            continue
        corpus_name = sub.name
        docs, texts, ids, metas = [], [], [], []

        for file in sub.glob("*.jsonl"):
            docs.extend(list(read_jsonl(file)))

        if not docs:
            print(f"跳过空目录: {corpus_name}")
            continue

        print(f"\n=== 构建索引: '{corpus_name}' (共 {len(docs)} 条) ===")

        for d in docs:
            texts.append(clean(d.get("content", "")))
            ids.append(d.get("id"))
            metas.append({"title": d.get("title", ""), "url": d.get("url", "")})

        # ---- Persist 路径 ----
        persist_dir = base_persist_dir / model_safe / f"{corpus_name}_index"
        persist_dir.mkdir(parents=True, exist_ok=True)

        # ---- 创建 / 获取 collection ----
        client = chromadb.PersistentClient(path=str(persist_dir))
        col_name = f"{corpus_name}_corpus"

        try:
            collection = client.get_collection(name=col_name)
            # 若维度不匹配 → 先删后建
            if collection.metadata.get("dimension") != emb_dim:
                print(f"[WARN] 维度不一致，重建 collection '{col_name}'")
                client.delete_collection(name=col_name)
                raise chromadb.errors.NotFoundError  # 触发下方重建
        except (ValueError, chromadb.errors.NotFoundError):
            collection = client.get_or_create_collection(
                name=col_name,
                embedding_function=embed_fn,      # 明确声明本地 encoder
                metadata={                       # 明确写入维度 & 距离度量
                    "hnsw:space": "cosine",
                    "dimension":  emb_dim,
                },
            )

        # ---- 生成向量 ----
        embeds = st_model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=True).tolist()

        # ---- 分批写入 ----
        for start in tqdm(range(0, len(ids), BATCH_SIZE),
                          desc=f"Inserting '{corpus_name}'"):
            end = min(start + BATCH_SIZE, len(ids))
            collection.add(
                ids        = ids[start:end],
                documents  = texts[start:end],
                metadatas  = metas[start:end],
                embeddings = embeds[start:end],
            )

        print(f"✅ 完成 '{corpus_name}' ➜ 持久化到 '{persist_dir}'")

# ---------------------------- CLI -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_corpus_dir",  default="numsports/corpus")
    parser.add_argument("--base_persist_dir", default="numsports/indexes")
    parser.add_argument("--model_name",       default="BAAI/bge-m3")
    main(parser.parse_args())
