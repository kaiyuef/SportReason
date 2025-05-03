#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UnifiedRetriever v3.3  (auto-index, chunked rerank, per-source top-N)
====================================================================
• 稀疏检索 : BM25 (Pyserini)
• 稠密检索 : Chroma (text / table / infobox)
• 交叉重排 : FlagEmbedding (小批量 + AMP)
"""
from __future__ import annotations
import os, json, torch, logging
from typing import Sequence

# ────────── 常量 & 工具 ──────────
ROOT_INDEX_DIR = "numsports/indexes"        # 写死根目录
os.environ.setdefault("TRANSFORMERS_CACHE", "/scratch/kf2365/.cache")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

def _safe(model: str) -> str:
    """'BAAI/bge-m3' → 'BAAI_bge-m3'"""
    return model.replace("/", "_").replace("\\", "_")

def _idx_path(model: str, idx_type: str) -> str:
    """numsports/indexes/{safe_model}/{idx_type}_index"""
    return os.path.join(ROOT_INDEX_DIR, _safe(model), f"{idx_type}_index")

# ────────── 1. BM25 ──────────
from pyserini.search.lucene import LuceneSearcher
class BM25Retriever:
    def __init__(self, index_dir: str = "indexes/lucene-index", k1=1.2, b=0.9):
        if not os.path.exists(index_dir):
            raise FileNotFoundError(index_dir)
        self.s = LuceneSearcher(index_dir)
        self.s.set_bm25(k1, b)

    def _pack(self, hit):
        raw = self.s.doc(hit.docid).raw()
        if raw.strip().startswith("{"):
            try:
                raw = json.loads(raw).get("contents", raw)
            except Exception:
                pass
        return {"id": hit.docid, "content": raw, "score": hit.score, "source": "bm25"}

    def retrieve(self, q, k):        return [self._pack(h) for h in self.s.search(q, k)]
    def retrieve_batch(self, qs, k): return [self.retrieve(q, k) for q in qs]

# ────────── 2. Chroma Helper ──────────
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class _Chroma:
    """始终使用本地 SentenceTransformerEmbeddingFunction 编码查询。"""
    def __init__(self, path, collection, model, device, metric, label):
        self.metric, self.label = metric, label
        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=model,
            device=device,
            normalize_embeddings=True,
            trust_remote_code=True,
        )

        client = chromadb.PersistentClient(path=path)
        try:
            self.col = client.get_collection(name=collection)
        except (ValueError, chromadb.errors.NotFoundError):
            # 若不存在则自动创建
            self.col = client.get_or_create_collection(
                name=collection, embedding_function=self._embed_fn
            )

    def _sim(self, d):            # 距离→相似度
        return 1 - d if self.metric == "cosine" else -d

    def retrieve_batch(self, qs: list[str], k: int):
        SQLITE_MAX_VARS = 999
        chunk_size = max(1, SQLITE_MAX_VARS // max(k, 1))
        results = [[] for _ in qs]

        for st in range(0, len(qs), chunk_size):
            rng    = range(st, min(st + chunk_size, len(qs)))
            sub_qs = [qs[i] for i in rng]

            # ★ 始终本地编码，避免服务器端维度不一致
            embeds = self._embed_fn(sub_qs)
            out    = self.col.query(query_embeddings=embeds, n_results=k)

            for off, (ids, docs, dists) in enumerate(
                zip(out["ids"], out["documents"], out["distances"])
            ):
                packed = [
                    {
                        "id": i,
                        "content": doc,
                        "score": self._sim(dist),
                        "source": self.label,
                    }
                    for i, doc, dist in zip(ids, docs, dists)
                ]
                results[rng.start + off] = packed

        return results


# ────────── 3. FlagEmbedding Reranker ──────────
from FlagEmbedding import FlagLLMReranker

# ────────── 4. Unified Retriever ──────────
class UnifiedRetriever:
    def __init__(
        self,
        *,
        embedding_model: str = "BAAI/bge-m3",

        # ★ 默认改成 build 阶段生成的名字 ★
        text_collection: str = "text_corpus",
        table_collection: str = "table_corpus",
        infobox_collection: str = "infobox_corpus",

        # 检索规模
        text_k: int = 50, table_k: int = 10, infobox_k: int = 10,
        final_text_k: int = 10, final_table_k: int = 2, final_infobox_k: int = 2,

        distance_metric: str = "cosine",

        use_bm25: bool = False,
        bm25_k: int = 10,
        bm25_index_dir: str = "indexes/lucene-index",

        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-v2-gemma",
        reranker_batch: int = 8,
        max_rerank_docs: int | None = None,

        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # —— 自动推断索引路径 ——
        self.text_index   = _idx_path(embedding_model, "text")
        self.table_index  = _idx_path(embedding_model, "table")
        self.infobox_index= _idx_path(embedding_model, "infobox")

        # —— 检索候选规模 ——
        self.text_k, self.table_k, self.info_k = text_k, table_k, infobox_k
        # —— 最终输出规模 ——
        self.final_limits = {"text": final_text_k, "table": final_table_k, "infobox": final_infobox_k}

        self.reranker_batch = reranker_batch
        self.max_rerank_docs = max_rerank_docs if (max_rerank_docs and max_rerank_docs > 0) else None

        # —— Chroma —— 
        self.dense_text = _Chroma(
            self.text_index, text_collection,
            embedding_model, self.device, distance_metric, "text"
        )
        self.dense_table = _Chroma(
            self.table_index, table_collection,
            embedding_model, self.device, distance_metric, "table"
        )
        self.dense_info = _Chroma(
            self.infobox_index, infobox_collection,
            embedding_model, self.device, distance_metric, "infobox"
        )

        # —— BM25 ——
        self.bm25 = BM25Retriever(bm25_index_dir) if use_bm25 else None
        self.bm25_k = bm25_k

        # —— Reranker ——
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = FlagLLMReranker(
                reranker_model, device=self.device, use_fp16=True, trust_remote_code=True
            )

    # ====== 私有: 分批重排 ======
    def _rerank_chunked(self, queries, docs_lists):
        flat, meta = [], []
        for qi, (q, docs) in enumerate(zip(queries, docs_lists)):
            if self.max_rerank_docs:
                docs = docs[: self.max_rerank_docs]
                docs_lists[qi] = docs
            for d in docs:
                meta.append((qi, d))
                flat.append([q, d["content"]])

        scores = []
        for s in range(0, len(flat), self.reranker_batch):
            chunk = flat[s: s + self.reranker_batch]
            with torch.inference_mode(), torch.cuda.amp.autocast():
                scores.extend(self.reranker.compute_score(chunk))
            torch.cuda.empty_cache()

        per_q = [[] for _ in queries]
        for (qi, d), sc in zip(meta, scores):
            d["score"] = float(sc)
            per_q[qi].append((sc, d))
        return [[d for _, d in sorted(lst, key=lambda x: x[0], reverse=True)]
                for lst in per_q]

    # ====== 私有: 按源截断 ======
    def _select_top(self, docs):
        kept, cnt = [], {"text": 0, "table": 0, "infobox": 0}
        for d in docs:
            src = d["source"]
            if src in cnt and cnt[src] < self.final_limits[src]:
                kept.append(d); cnt[src] += 1
            if all(cnt[s] >= self.final_limits[s] for s in cnt):
                break
        return kept

    # ====== batch 主接口 ======
    def retrieve_batch(self, qs: Sequence[str]):
        txt = self.dense_text.retrieve_batch(qs, self.text_k)
        tbl = self.dense_table.retrieve_batch(qs, self.table_k)
        inf = self.dense_info.retrieve_batch(qs, self.info_k)
        bm  = self.bm25.retrieve_batch(qs, self.bm25_k) if self.bm25 else [[] for _ in qs]

        combined = [t + ta + i + b for t, ta, i, b in zip(txt, tbl, inf, bm)]
        ranked   = self._rerank_chunked(qs, combined) if self.use_reranker else combined
        return [self._select_top(docs) for docs in ranked]

    # ====== 单条查询 ======
    def retrieve(self, q: str):
        return self.retrieve_batch([q])[0]


# ────────── quick test ──────────
if __name__ == "__main__":
    ur = UnifiedRetriever(
        embedding_model="BAAI/bge-m3",
        text_k=50, table_k=10, infobox_k=10,
        final_text_k=10, final_table_k=2, final_infobox_k=2,
        use_bm25=False, reranker_batch=16,
    )
    for d in ur.retrieve("量子霍尔效应 提出者"):
        print(f"[{d['source']:^7}] {d['score']:.3f}  {d['id']}")
