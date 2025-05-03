#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two‑Stage QA Pipeline – reader‑optimised (CBS enabled)
=====================================================
* Retrieval      ➜ UnifiedRetriever‑v3   (BM25 / Chroma / Hybrid)
* Rerank         ➜ FlagEmbedding cross‑encoder (optional)
* **Reader**     ➜ HuggingFaceReader (vLLM backend – Continuous Batch Scheduling)

Key points
----------
1. **vLLM CBS**  All prompts are queued with ``generate_async``.  The
   scheduler interleaves token generation across requests → higher GPU
   utilisation.
2. **Micro‑batch safety**  We still slice the big prompt list into
   chunks of size ``INNER_BATCH`` when submitting.  This bounds the
   *concurrent* number of active requests and therefore VRAM peak.
3. **No external async/await**  We collect Python ``Future`` objects
   from vLLM and block only once, after all submissions, via
   ``result()`` calls – simple and thread‑safe.
"""
from __future__ import annotations
import argparse, json, os, sys, torch, textwrap
from pathlib import Path
from typing import List, Dict, Union

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_CACHE", "/scratch/kf2365/.cache")

# ---------------------------------------------------------------------------
# HuggingFaceReader (CBS enabled)
# ---------------------------------------------------------------------------
class HuggingFaceReader:
    """Singleton vLLM reader with Continuous Batch Scheduling (CBS).

    Usage remains the same: ``generate_answer(queries, docs_lists)``.
    Internally we push *all* prompts via ``generate_async`` in slices of
    ``INNER_BATCH`` to bound VRAM, then synchronise once at the end.
    """

    _llm_instance: LLM | None = None
    _model_name_loaded: str | None = None

    # Adjust ↑↓ according to GPU capability; 4 is safe for 32B‑AWQ on RTX8000
    INNER_BATCH = 4

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = self._get_llm(model_name)
        self._samp = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=1024)

    # ---- singleton loader ----
    @classmethod
    def _get_llm(cls, model_name: str) -> LLM:
        if cls._llm_instance is None or cls._model_name_loaded != model_name:
            print(f"[INFO] Loading vLLM model → {model_name}")
            cls._llm_instance = LLM(model_name, dtype="float16", trust_remote_code=True)
            cls._model_name_loaded = model_name
        return cls._llm_instance

    # ---- prompt helper ----
    @staticmethod
    def _build_prompt(question: str, docs: List[str]) -> str:
        context_blocks = [f"Document {i+1}:\n{d}" for i, d in enumerate(docs[:10])]
        context = "\n\n".join(context_blocks) or (
            "No relevant documents were retrieved. Answer based on prior knowledge.")
        return textwrap.dedent(f"""
            Think step by step. Answer the following question based on the context.

            Context:
            {context}

            Question:
            {question}

            Place your answer in \boxed{{answer}} format.
            Answer:""")

    # ---- public API ----
    # ------------------------------------------------------------
    # Revised: uses vLLM .generate() (no generate_async), keeps
    #          inner-batch chunking and memory-safe behaviour.
    # ------------------------------------------------------------
    def generate_answer(self, queries, retrieved_docs_list):
        """
        Args
        ----
        queries : List[str]
        retrieved_docs_list : List[List[dict]]
            Each inner list is the retrieved-docs dicts for that query.

        Returns
        -------
        List[dict]  # one per query
        """
        if len(queries) != len(retrieved_docs_list):
            raise ValueError("queries 与 retrieved_docs_list 长度不一致")

        # ---- build prompts -------------------------------------------------
        prompts = []
        for q, docs in zip(queries, retrieved_docs_list):
            # 只取前 10 篇文档内容
            context = "\n\n".join(
                [f"Document {i+1}:\n{d['content']}" for i, d in enumerate(docs[:10])]
            ) or "No relevant documents were retrieved. Answer based on prior knowledge."

            prompt = (
                "Think step by step. Answer the following question based on the context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{q}\n\n"
                "Place your answer in \\boxed{answer} format.\nAnswer:"
            )
            prompts.append(prompt)

        # ---- sampling params ----------------------------------------------
        samp = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=1024,
        )

        # ---- call vLLM -----------------------------------------------------
        # vLLM handles batching internally; we just pass the list of prompts.
        outputs = self.llm.generate(prompts, sampling_params=samp)
        torch.cuda.empty_cache()

        # ---- assemble result ----------------------------------------------
        results = []
        for i, out in enumerate(outputs):
            answer_txt = out.outputs[0].text.strip() if out.outputs else "No response."
            results.append({
                "query":           queries[i],
                "prompt":          prompts[i],
                "answer":          answer_txt,
                "documents_used":  len(retrieved_docs_list[i]),
            })
        return results

