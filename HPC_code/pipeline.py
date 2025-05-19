#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two‑Stage QA Pipeline  –  multi‑corpus aware  (reader‑optimised, sliceable)
==========================================================================
• Retrieval  ➜ UnifiedRetriever‑v3   (BM25 / Chroma / Hybrid)
• Rerank     ➜ Auto-detect ⟶ (CrossEncoder / Seq-CLS / LLM-YesNo)
• Reader     ➜ HuggingFaceReader     (vLLM backend, inner batch)

新增能力
--------
1. `--embed_model`     控制嵌入模型，自动定位 numsports/indexes/{model}/... 索引
2. `--start_idx / --end_idx`  切片评测区间（0‑based，end 不含）
3. `--stage {all,retrieval,reader}` 单一参数控制三种执行模式
      • all        : 检索 + Reader（默认）
      • retrieval  : 只跑检索，生成 retrieved_docs.json 后退出
      • reader     : 只跑 Reader，复用已有 retrieved_docs.json
4. 当 `--retriever bm25` 时：
   • 仅跑稀疏检索，稠密 top‑k 自动置 0
   • 结果目录强制为 numsports/answer_results/bm25
   • 结果文件固定前缀 "bm25_"
"""
from __future__ import annotations
import argparse, json, os, sys, torch, textwrap
from pathlib import Path

# ---------- 环境变量 ----------


# ---------- 本地模块 ----------
ROOT = Path(__file__).parent
sys.path.append(ROOT.as_posix())
from retrievers import UnifiedRetriever           # noqa: E402
from readers    import HuggingFaceReader          # noqa: E402

# ═════════════════ Retrieval Wrapper ═════════════════
class RetrievalPipeline:
    """统一检索包装：负责配置 UnifiedRetriever 并批量检索。"""
    def __init__(
        self,
        *,
        embedding_model: str,
        mode: str,
        distance: str,
        device: str | None,
        use_reranker: bool,
        reranker_model: str,
        text_k: int,
        table_k: int,
        infobox_k: int,
        final_text_k: int,
        final_table_k: int,
        final_infobox_k: int,
        reranker_batch: int,
        max_rerank_docs: int | None,
    ):
        if mode not in {"bm25", "chroma", "hybrid"}:
            raise ValueError("retriever mode 必须是 bm25 / chroma / hybrid")

        dense_on = mode in {"chroma", "hybrid"}
        bm25_on  = mode in {"bm25",  "hybrid"}

        # ----- 纯 BM25 时关闭稠密检索，但保留各源的 bm25_k -----
        bm25_k_dict = {"text": text_k, "table": table_k, "infobox": infobox_k}
        if not dense_on:
            text_k = table_k = infobox_k = 0  # 禁用 Chroma

        print(
            textwrap.dedent(f"""\
            [INFO] Retriever={mode}, embed_model={embedding_model}, distance={distance},
                   reranker={'OFF' if not use_reranker else reranker_model},
                   batch={reranker_batch}, max_docs={max_rerank_docs},
                   final_k(text/table/inf)={final_text_k}/{final_table_k}/{final_infobox_k}""")
        )

        self.retriever = UnifiedRetriever(
            embedding_model  = embedding_model,
            # 稀疏 / 稠密 开关
            use_bm25        = bm25_on,
            dense_enabled   = dense_on,

            # 稠密 Top‑K
            text_k          = text_k,
            table_k         = table_k,
            infobox_k       = infobox_k,

            # 稀疏 Top‑K
            bm25_k          = bm25_k_dict if bm25_on else 0,
            bm25_index_dir  = "numsports/indexes/lucene_index",

            # 输出/重排控制
            final_text_k    = final_text_k,
            final_table_k   = final_table_k,
            final_infobox_k = final_infobox_k,
            distance_metric = distance,
            use_reranker    = use_reranker,
            reranker_model  = reranker_model,
            reranker_batch  = reranker_batch,
            max_rerank_docs = (
                max_rerank_docs if max_rerank_docs and max_rerank_docs > 0 else None
            ),
            device          = device,
        )

    def retrieve_batch(self, qs):
        return self.retriever.retrieve_batch(qs)


# ═════════════════ Reading Wrapper ══════════════════
class ReadingPipeline:
    """批量高效 Reader 封装：一次性生成所有答案。"""

    def __init__(self, key: str):
        model_map = {
            "qwen":  "Qwen/Qwen2.5-7B-Instruct",
            "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
        }
        if key not in model_map:
            raise ValueError("reader 只能选 qwen / llama")
        print(f"[INFO] Loading reader ➜ {model_map[key]}")
        self.reader = HuggingFaceReader(model_name=model_map[key])

    def answer(self, queries, docs_lists):
        assert len(queries) == len(docs_lists), "queries / docs_lists 长度不一致"
        outputs = self.reader.generate_answer(queries, docs_lists)
        torch.cuda.empty_cache()
        return outputs

# ═════════════════  Pipeline Driver ══════════════════
def run_pipeline(
    input_path: str,
    output_dir: str | None,
    *,
    stage: str,                   # ★ new parameter
    embedding_model: str,
    retriever: str,
    distance: str,
    reranker_model: str,
    reader: str,
    device: str,
    text_k: int,
    table_k: int,
    infobox_k: int,
    final_text_k: int,
    final_table_k: int,
    final_infobox_k: int,
    reranker_batch: int,
    max_rerank_docs: int | None,
    start_idx: int | None,
    end_idx:   int | None,
):
    # -------- 输出目录策略 --------
    # 规范化 reranker 子文件夹名
    folder_reranker = (                              # ★ UPDATED
        "no_rerank" if reranker_model.lower() == "none"
        else reranker_model.replace("/", "_")
    )

    if output_dir is None:                           # ★ UPDATED
        if retriever == "bm25":
            output_dir = f"numsports/answer_results/bm25/{folder_reranker}"
        else:
            output_dir = f"numsports/answer_results/{retriever}/{folder_reranker}"
    # 若用户手动给了路径但仍想跑 bm25，则强制覆盖到规范路径
    elif retriever == "bm25" and not output_dir.startswith("numsports/answer_results/bm25/"):
        print(
            "[WARN] bm25 模式已重定向到规范路径 "
            f"numsports/answer_results/bm25/{folder_reranker}"
        )
        output_dir = f"numsports/answer_results/bm25/{folder_reranker}"

    inp      = Path(input_path)
    out_dir  = Path(output_dir);  out_dir.mkdir(parents=True, exist_ok=True)

    retrieved_fp = out_dir / "retrieved_docs.json"

    # 结果文件名前缀
    if retriever == "bm25":
        fn_prefix = "bm25"
    else:
        fn_prefix = embedding_model.replace("/", "_")
    reranker_tag = reranker_model.replace("/", "_") if reranker_model.lower() != "none" else "no_rerank"
    final_fp = out_dir / f"{fn_prefix}_{reranker_tag}.json"

    # -------- 读入数据 --------
    data = [json.loads(l) for l in inp.read_text("utf-8").splitlines() if l.strip()]

    if start_idx is not None or end_idx is not None:
        s = start_idx or 0
        e = end_idx   or len(data)
        print(f"[INFO] Slice samples: {s} → {e} (orig {len(data)})")
        data = data[s:e]

    queries = [d["seed_question"] for d in data]
    id_map  = {d["seed_question"]: d["id"] for d in data}

    # -------- 检索阶段 --------
    if stage in {"all", "retrieval"}:
        ret_pipe = RetrievalPipeline(
            embedding_model = embedding_model,
            mode            = retriever,
            distance        = distance,
            device          = device,
            use_reranker    = (reranker_model.lower() != "none"),
            reranker_model  = reranker_model,
            text_k          = text_k,
            table_k         = table_k,
            infobox_k       = infobox_k,
            final_text_k    = final_text_k,
            final_table_k   = final_table_k,
            final_infobox_k = final_infobox_k,
            reranker_batch  = reranker_batch,
            max_rerank_docs = max_rerank_docs,
        )
        doc_lists = ret_pipe.retrieve_batch(queries)
        retrieved = {q: docs for q, docs in zip(queries, doc_lists)}
        retrieved_fp = out_dir / "retrieved_docs.jsonl"
        try:
            with retrieved_fp.open("w", encoding="utf-8") as f:
                for q, docs in zip(queries, doc_lists):
                    f.write(json.dumps({"query": q, "retrieved_docs": docs}, ensure_ascii=False) + "\n")
            print(f"[INFO] Retrieval‑only 完成，结果保存在 {retrieved_fp}")
        except Exception as e:
            print(f"[ERROR] 写入检索结果失败: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

        del ret_pipe
        torch.cuda.empty_cache()
        #torch.cuda.ipc_collect()

        if stage == "retrieval":
            return

    else:  # stage == "reader"
        print("[INFO] Reader‑only 模式 – 复用 cached retrieved_docs.json")
        if not retrieved_fp.exists():
            raise FileNotFoundError(
                f"未找到缓存检索文件 {retrieved_fp}，请先运行 stage=retrieval 或 stage=all"
            )
        # 新的 retrieved_docs.jsonl 读取方式
        retrieved_data = [json.loads(l) for l in retrieved_fp.read_text("utf-8").splitlines()]
        retrieved = {item["query"]: item["retrieved_docs"] for item in retrieved_data}

    # -------- 阅读阶段 --------
    reader_pipe = ReadingPipeline(reader)
    answers     = reader_pipe.answer(queries, [retrieved[q] for q in queries])

    # -------- 合并并保存 --------
    final = {}
    for q, ans in zip(queries, answers):
        qid = id_map.get(q)
        if not qid:
            continue
        final[qid] = {
            "query":           q,
            "answer":          ans.get("answer", ""),
            "prompt":          ans.get("prompt", ""),
            "documents_used":  ans.get("documents_used", []),
            "retrieved_docs":  retrieved[q],
        }

    final_fp.write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[INFO] Saved ➜ {final_fp}")
    sys.exit(0)



# ═════════════════════ CLI ════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser("Numsports QA Pipeline")
    p.add_argument("--input",  default="numsports/dataset/merged_dataset_final.jsonl")
    p.add_argument("--output", default=None,
                   help="结果输出目录；若省略则自动根据 --retriever 选择")

    p.add_argument("--embed_model", default="BAAI/bge-m3",
                   help="SentenceTransformer 嵌入模型 (控制索引路径)")

    p.add_argument("--retriever", choices=["bm25", "chroma", "hybrid"], default="chroma")
    p.add_argument("--distance",  choices=["cosine", "l2"],             default="cosine")
    p.add_argument("--reranker_model", default="BAAI/bge-reranker-v2-gemma",
                   help="'none' 关闭交叉重排")

    p.add_argument("--reader", choices=["qwen", "llama"], default="qwen")
    p.add_argument("--device", default="cuda")

    # 候选 Top‑K
    p.add_argument("--text_k",    type=int, default=50)
    p.add_argument("--table_k",   type=int, default=10)
    p.add_argument("--infobox_k", type=int, default=10)
    # 重排后保留 Top‑K
    p.add_argument("--final_text_k",    type=int, default=10)
    p.add_argument("--final_table_k",   type=int, default=2)
    p.add_argument("--final_infobox_k", type=int, default=2)

    p.add_argument("--reranker_batch",  type=int, default=16)
    p.add_argument("--max_rerank_docs", type=int, default=70,
                   help=">0 时限制每问参与重排的文档数")

    # Slice
    p.add_argument("--start_idx", type=int, default=None)
    p.add_argument("--end_idx",   type=int, default=None)

    # ★ unified stage flag
    p.add_argument("--stage", choices=["all", "retrieval", "reader"],
                   default="all", help="选择执行阶段")

    a = p.parse_args()

    run_pipeline(
        a.input, a.output,
        stage           = a.stage,
        embedding_model = a.embed_model,
        retriever       = a.retriever,
        distance        = a.distance,
        reranker_model  = a.reranker_model,
        reader          = a.reader,
        device          = a.device,
        text_k          = a.text_k,
        table_k         = a.table_k,
        infobox_k       = a.infobox_k,
        final_text_k    = a.final_text_k,
        final_table_k   = a.final_table_k,
        final_infobox_k = a.final_infobox_k,
        reranker_batch  = a.reranker_batch,
        max_rerank_docs = (a.max_rerank_docs if a.max_rerank_docs and a.max_rerank_docs > 0 else None),
        start_idx       = a.start_idx,
        end_idx         = a.end_idx,
    )
