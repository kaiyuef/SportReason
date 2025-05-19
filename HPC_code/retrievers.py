#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UnifiedRetriever v3.6  (auto-index, auto-reranker-detect, per-source top-N)
===========================================================================
• Sparse  : BM25 (Pyserini)                    – optional
• Dense   : Chroma (text / table / infobox)    – optional
• Rerank  : Auto-detect
            ─ CrossEncoder   (Sentence-Transformers)
            ─ HF Seq-CLS     (AutoModelForSequenceClassification)
            ─ LLM-based      (AutoModelForCausalLM, Yes/No log-prob)
"""

from __future__ import annotations
import os, json, logging, pathlib, torch
from typing import Sequence
from tqdm import tqdm


# ────────── 环境 & 常量 ──────────
ROOT_INDEX_DIR = "numsports/indexes"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_PROGRESS_BAR"] = "1"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------------------------------------------------
#  辅助函数
# -------------------------------------------------------------------------
def _safe(model: str) -> str:
    return model.replace("/", "_").replace("\\", "_")

def _idx_path(model: str, idx_type: str) -> str:
    return os.path.join(ROOT_INDEX_DIR, _safe(model), f"{idx_type}_index")

# -------------------------------------------------------------------------
#  1. BM25  (Pyserini)
# -------------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, root_dir: str, *, k1=1.2, b=0.9):
        from pyserini.search.lucene import LuceneSearcher
        self.searchers = {}
        for src in ("text", "table", "infobox"):
            idx = os.path.join(root_dir, src)
            if not pathlib.Path(idx).is_dir():
                logging.warning(f"BM25 子索引不存在，跳过: {idx}")
                continue
            s = LuceneSearcher(idx)
            s.set_bm25(k1, b)
            self.searchers[src] = s
        if not self.searchers:
            raise FileNotFoundError(f"BM25 索引缺失于 {root_dir}")

    @staticmethod
    def _pack(hit, searcher, src):
        raw = searcher.doc(hit.docid).raw()
        if raw.lstrip().startswith("{"):
            try:
                j = json.loads(raw)
                doc_id = j.get("id", hit.docid)
                txt = j.get("contents", raw)
            except Exception:
                doc_id, txt = hit.docid, raw
        else:
            doc_id, txt = hit.docid, raw
        return {"id": str(doc_id), "content": txt, "score": float(hit.score), "source": src}

    def retrieve_batch(self, qs: Sequence[str], k_cfg: dict[str,int] | int):
        if isinstance(k_cfg, int):
            k_cfg = {s: k_cfg for s in ("text","table","infobox")}
        results = [[] for _ in qs]
        for src, searcher in self.searchers.items():
            k = k_cfg.get(src, 0)
            if k <= 0:
                continue
            for qi, q in enumerate(qs):
                for hit in searcher.search(q, k):
                    results[qi].append(self._pack(hit, searcher, src))
        for lst in results:
            lst.sort(key=lambda d: d["score"], reverse=True)
        return results

# -------------------------------------------------------------------------
#  2. Chroma Dense Retriever
# -------------------------------------------------------------------------
class _Chroma:
    def __init__(self, path, collection, model, device, metric, label):
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        self.metric, self.label = metric, label
        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=model, device=device, normalize_embeddings=True, trust_remote_code=True)
        client = chromadb.PersistentClient(path=path)
        try:
            self.col = client.get_collection(collection)
        except Exception:
            self.col = client.get_or_create_collection(
                collection, embedding_function=self._embed_fn)

    def _sim(self, d):
        return 1 - d if self.metric == "cosine" else -d

    def retrieve_batch(self, qs: list[str], k: int):
        SQLITE_MAX_VARS = 999
        chunk = max(1, SQLITE_MAX_VARS // max(k, 1))
        res = [[] for _ in qs]
        embeds = self._embed_fn(qs)
        for st in range(0, len(qs), chunk):
            out = self.col.query(
                query_embeddings=embeds[st:st+chunk],
                n_results=k
            )
            for off, (ids, docs, dists) in enumerate(
                    zip(out["ids"], out["documents"], out["distances"])):
                res[st+off] = [
                    {"id": i, "content": d, "score": self._sim(dist), "source": self.label}
                    for i, d, dist in zip(ids, docs, dists)
                ]
        return res

# -------------------------------------------------------------------------
#  3. Reranker 家族
# -------------------------------------------------------------------------
from sentence_transformers import CrossEncoder
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForCausalLM)
import contextlib, math, torch

class CrossEncoderReranker:
    def __init__(self, model_name:str, device:str):
        self.ce = CrossEncoder(
            model_name, device=device, max_length=512,
            default_activation_function=None,
            trust_remote_code=True
        )

    @torch.inference_mode()
    def compute_score(self, pairs, bs=32):
        scores = []
        for st in range(0, len(pairs), bs):
            sub = pairs[st:st+bs]
            preds = self.ce.predict(
                sub, convert_to_numpy=False,
                batch_size=len(sub), show_progress_bar=False
            )
            scores.extend(preds)
        return scores

class HFSeqClsReranker:
    def __init__(self, model_name:str, device:str):
        self.tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        
        if self.tok.pad_token_id is None:
            # eos_token 一定存在，否则模型根本没法生成结束符
            self.tok.pad_token   = self.tok.eos_token
            self.tok.pad_token_id= self.tok.eos_token_id
            
        self.m = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        ).to(device).eval()

    @torch.inference_mode()
    def compute_score(self, pairs, bs=32):
        scores = []
        for st in range(0, len(pairs), bs):
            sub = pairs[st:st+bs]
            toks = self.tok(
                sub, padding=True, truncation=True,
                return_tensors="pt", max_length=512
            ).to(self.m.device)
            logits = self.m(**toks).logits.view(-1).float().cpu().tolist()
            scores.extend(logits)
        return scores

# ---------------------------------------------------------------------
#  新版 LLMReranker  —— 采用官方推荐输入构造 & Yes/No 对比打分
# ---------------------------------------------------------------------
import contextlib, math, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMReranker:
    """
    使用 LLM (“Yes/No”) 方式进行重排评分。
    改进要点
    -------------
    1. 采用官方 prepare_for_model 方式构造输入，避免手动拼接错误。
    2. 按模型实际 max_position_embeddings 动态截断，优先保留 passage。
    3. 同时取 “Yes” 与 “No” 两个 token 的 logits，返回差值 (Yes - No) 作为分数，
       使不同模型/批次间分数更可比。
    4. 仅在 CUDA 设备时启用 AMP，CPU 走 float32。
    """

    PROMPT = ("Given a query A and a passage B, determine whether the passage "
              "contains an answer to the query by providing a prediction of "
              "either 'Yes' or 'No'.")

    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name,
                                                 trust_remote_code=True)
        self.m = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            trust_remote_code=True,
        ).to(device).eval()

        # 多 token “Yes”/“No”
        self.yes_ids = self.tok("Yes", add_special_tokens=False)["input_ids"]
        self.no_ids  = self.tok("No",  add_special_tokens=False)["input_ids"]

        # 预先缓存 prompt & 分隔符 token
        self._prompt_ids = self.tok(self.PROMPT,
                                    add_special_tokens=False)["input_ids"]
        self._sep_ids    = self.tok("\n", add_special_tokens=False)["input_ids"]

        # 最大长度由模型 config 决定
        self._ctx_max = getattr(self.m.config, "max_position_embeddings", 2048)

    # ---- 内部：批量构造输入 -------------------------------------------------
    def _build_inputs(self,
                      pairs: list[tuple[str, str]],
                      max_len: int | None = None):
        """返回 pad 后的 input_ids / attention_mask（torch.Tensor, batch first）"""
        max_len = max_len or self._ctx_max
        tok = self.tok
        prompt_ids, sep_ids = self._prompt_ids, self._sep_ids
        inputs = []

        q_max = max_len * 3 // 4         # query 至多占 3/4
        p_max = max_len                  # passage 先给足，后面 prepare_for_model 会截 second

        for q, p in pairs:
            q_ids = tok(f"A: {q}", add_special_tokens=False,
                        truncation=True, max_length=q_max)["input_ids"]
            p_ids = tok(f"B: {p}", add_special_tokens=False,
                        truncation=True, max_length=p_max)["input_ids"]

            # prepare_for_model 负责插入 `[BOS]` 和截断第二段 (passage)
            item = tok.prepare_for_model(
                [tok.bos_token_id] + q_ids,
                sep_ids + p_ids,
                truncation='only_second',
                max_length=max_len,
                padding=False,
                return_attention_mask=False,
                add_special_tokens=False,
            )
            # 手动拼接 sep + prompt
            item["input_ids"] = item["input_ids"] + sep_ids + prompt_ids
            item["attention_mask"] = [1] * len(item["input_ids"])
            inputs.append(item)

        return tok.pad(
            inputs,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ).to(self.device)

    # ---- 公开 API ----------------------------------------------------------
    @torch.inference_mode()
    def compute_score(self, pairs: list[tuple[str, str]], bs: int = 4):
        """
        返回与 pairs 对齐的分数列表  score = logit(Yes) - logit(No)
        """
        scores: list[float] = []
        amp_ctx = (torch.cuda.amp.autocast() if "cuda" in self.device
                   else contextlib.nullcontext())

        for st in range(0, len(pairs), bs):
            batch_pairs = pairs[st: st + bs]
            batch_inputs = self._build_inputs(batch_pairs)

            with amp_ctx:
                out = self.m(**batch_inputs).logits  # (B, T, V)

            # 末 token logits
            last_logits = out[:, -1, :]              # (B, V)

            # 求 Yes/No 的平均 logit（多 token 时取均值）
            yes_logit = last_logits[:, self.yes_ids].mean(dim=-1)
            no_logit  = last_logits[:, self.no_ids].mean(dim=-1)

            batch_score = (yes_logit - no_logit).float().cpu().tolist()
            scores.extend(batch_score)

        return scores


class JinaSeqClsReranker(HFSeqClsReranker):
    """专门处理 JinaAI reranker（自定义 XLM‑RobertaFlashConfig）。"""
    def __init__(self, model_name: str, device: str):
        super().__init__(model_name, device)  # 父类已加 trust_remote_code=True

class NVEmbedEmbeddingFunction:
    """
    用于 nvidia/NV-Embed-v2 的自定义 embedding function。
    返回 shape=(N, D) 的 numpy.ndarray。
    """
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model     = AutoModel.from_pretrained(model_name, trust_remote_code=True)\
                                  .to(device).eval()

    def __call__(self, texts: list[str]):
        # 批量 tokenize
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        with torch.inference_mode():
            out = self.model(**toks)
            # 简单做 mean pooling
            last = out.last_hidden_state  # (batch, seq, dim)
            embs = last.mean(dim=1)       # (batch, dim)
        return embs.cpu().numpy().tolist()
# -------- Reranker 选择 --------
def _choose_reranker(model_name: str, device: str):
    name = model_name.lower()

    # --- 0) JinaAI 专用分支 ------------------------------------------
    if name.startswith("jinaai/"):
        logging.info("[Reranker] Detected JinaAI model – using HF Seq‑CLS loader")
        return JinaSeqClsReranker(model_name, device)

    # --- 1) CrossEncoder 优先 ----------------------------------------
    try:
        if "reranker" in name and not any(
            t in name for t in ("gemma", "llama", "gpt", "mistral", "qwen", "phi", "inf","gte")
        ):
            return CrossEncoderReranker(model_name, device)

        # --- 2) HF Seq‑CLS 尝试 --------------------------------------
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(cfg, "num_labels") and cfg.num_labels > 1:
            # 支持分类头，直接初始化一次
            return HFSeqClsReranker(model_name, device)


    except Exception as e:
        logging.warning(f"[Reranker] Seq‑CLS load failed → fallback to LLM ({e})")

    # --- 3) LLM fallback --------------------------------------------
    return LLMReranker(model_name, device)


# -------------------------------------------------------------------------
#  4. UnifiedRetriever
# -------------------------------------------------------------------------
class UnifiedRetriever:
    def __init__(self, *,
        embedding_model="BAAI/bge-m3",
        text_k=50, table_k=10, infobox_k=10,
        final_text_k=10, final_table_k=2, final_infobox_k=2,
        distance_metric="cosine",
        use_bm25=False, bm25_k=50, bm25_index_dir="numsports/indexes/lucene_index",
        dense_enabled=True,
        use_reranker=True, reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_batch=64, max_rerank_docs=None,
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # dense
        self.dense_text  = _Chroma(
            _idx_path(embedding_model,"text"), "text_corpus",
            embedding_model, self.device, distance_metric,"text"
        ) if dense_enabled and text_k>0 else None
        self.dense_table = _Chroma(
            _idx_path(embedding_model,"table"), "table_corpus",
            embedding_model, self.device, distance_metric,"table"
        ) if dense_enabled and table_k>0 else None
        self.dense_info  = _Chroma(
            _idx_path(embedding_model,"infobox"), "infobox_corpus",
            embedding_model, self.device, distance_metric,"infobox"
        ) if dense_enabled and infobox_k>0 else None
        self.text_k, self.table_k, self.info_k = text_k, table_k, infobox_k
        self.final_limits = {"text":final_text_k,"table":final_table_k,"infobox":final_infobox_k}

        # bm25
        self.bm25 = BM25Retriever(bm25_index_dir) if use_bm25 else None
        self.bm25_k_cfg = bm25_k if isinstance(bm25_k,dict) else {
            s: bm25_k for s in ("text","table","infobox")
        }

        # reranker
        self.use_reranker = use_reranker
        if use_reranker:
            logging.info(f"[Reranker] Auto-detecting type for '{reranker_model}' …")
            self.reranker = _choose_reranker(reranker_model, self.device)
        self.reranker_batch = reranker_batch
        self.max_rerank_docs = max_rerank_docs or None

        # prompt prefix
        self._prompt_q = "query: "
        self._prompt_d = {
            "text":"text poll: ",
            "table":"table poll: ",
            "infobox":"infobox poll: "
        }

    # —— utils —— ------------------------------------------------------
    @staticmethod
    def _linearize(doc, max_tok=256):
        src, txt = doc["source"], doc["content"]
        if src == "text":
            return txt[:2048]
        try:
            j = json.loads(txt)
        except Exception:
            return txt[:2048]
        if src == "table":
            head = list(j.get("header", []))[:4]
            rows = j.get("rows", [])[:3]
            rows = [" | ".join(map(str, r[:4])) for r in rows]
            return (", ".join(head) + "; " + "; ".join(rows))[:max_tok*8]
        if src == "infobox":
            items = list(j.items())[:6]
            return ("; ".join(f"{k}: {v}" for k,v in items))[:max_tok*8]
        return txt[:2048]

    def _rerank(self, qs, docs_lists):
        if not self.use_reranker:
            return docs_lists

        # 构造所有要打分的 (query, doc) 对
        pairs, meta = [], []
        for src in ("text", "table", "infobox"):
            for qi, docs in enumerate(docs_lists):
                cand = [d for d in docs if d["source"] == src]
                if self.max_rerank_docs:
                    cand = cand[:self.max_rerank_docs]
                for d in cand:
                    meta.append((qi, d))
                    pairs.append((
                        self._prompt_q + qs[qi],
                        self._prompt_d[src] + self._linearize(d)
                    ))

        if not pairs:
            return docs_lists

        total_batches = (len(pairs) + self.reranker_batch - 1) // self.reranker_batch
        scores = []
        # 单一进度条
        for st in tqdm(
            range(0, len(pairs), self.reranker_batch),
            desc="[Rerank] Total",
            dynamic_ncols=True,
            total=total_batches
        ):
            batch = pairs[st:st+self.reranker_batch]
            batch_scores = self.reranker.compute_score(batch, bs=len(batch))
            scores.extend(batch_scores)

        # 将分数写回原文档结构并重排序
        for (qi, d), sc in zip(meta, scores):
            d["score"] = float(sc)

        return [
            sorted(lst, key=lambda x: x["score"], reverse=True)
            for lst in docs_lists
        ]

    def _select_top(self, docs):
        kept, cnt = [], {"text":0,"table":0,"infobox":0}
        for d in docs:
            s = d["source"]
            if cnt[s] < self.final_limits[s]:
                kept.append(d)
                cnt[s] += 1
            if all(cnt[k] >= self.final_limits[k] for k in cnt):
                break
        return kept

    # —— public API —— --------------------------------------------------
    def retrieve_batch(self, qs:Sequence[str]):
        res = [[] for _ in qs]
        if self.dense_text:
            for qi, h in enumerate(self.dense_text.retrieve_batch(qs, self.text_k)):
                res[qi].extend(h)
        if self.dense_table:
            for qi, h in enumerate(self.dense_table.retrieve_batch(qs, self.table_k)):
                res[qi].extend(h)
        if self.dense_info:
            for qi, h in enumerate(self.dense_info.retrieve_batch(qs, self.info_k)):
                res[qi].extend(h)
        if self.bm25:
            for qi, h in enumerate(self.bm25.retrieve_batch(qs, self.bm25_k_cfg)):
                res[qi].extend(h)

        # 初步排序
        for lst in res:
            lst.sort(key=lambda d: d["score"], reverse=True)

        # 重新打分并筛顶
        reranked = self._rerank(qs, res)
        return [self._select_top(lst) for lst in reranked]

    def retrieve(self, q:str):
        return self.retrieve_batch([q])[0]

# -------------------------------------------------------------------------
#  quick test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    ur = UnifiedRetriever(
        embedding_model   ="BAAI/bge-m3",
        text_k=50, table_k=10, infobox_k=10,
        final_text_k=10, final_table_k=2, final_infobox_k=2,
        use_bm25=False,
        reranker_model="BAAI/bge-reranker-v2-gemma",  # 自动识别为 LLM reranker
        reranker_batch=4,
        max_rerank_docs=64,
    )
    for d in ur.retrieve("量子霍尔效应 提出者"):
        print(f"[{d['source']:^7}] {d['score']:.3f}  {d['id']}")
