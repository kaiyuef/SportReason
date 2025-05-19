#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid QA Reader using Qwen‑32B‑AWQ with vLLM (CBS Enabled)
===========================================================
* Supports: Mixed Table/Text Evidence
* Prompt: Step-by-step reasoning with boxed answer
* Inference: Batched with Streaming Token Scheduling (CBS)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import torch
import textwrap
from pathlib import Path
from typing import List, Dict, Union

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------


class HuggingFaceReader:
    """vLLM Reader with Continuous Batch Scheduling (CBS), optimized for hybrid QA."""

    _llm_instance: LLM | None = None
    _model_name_loaded: str | None = None

    def __init__(self, model_name: str):
        self.model_name = model_name

        # 获取 GPU 数
        num_gpus = torch.cuda.device_count()
        print(f"[INFO] Detected {num_gpus} CUDA device(s)")

        # 动态设置 batch size
        if num_gpus >= 2:
            self.INNER_BATCH = 4  # safe for 2×48G
        else:
            self.INNER_BATCH = 2  # safe for 1×48G

        self.MAX_EVIDENCE_TOKENS = 4096
        self.llm = self._get_llm(model_name)
        self._samp = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=1024)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    @classmethod
    def _get_llm(cls, model_name: str) -> LLM:
        if cls._llm_instance is None or cls._model_name_loaded != model_name:
            print(f"[INFO] Loading vLLM model → {model_name}")
            cls._llm_instance = LLM(
                model_name,
                dtype="float16",
                trust_remote_code=True,
                tensor_parallel_size=torch.cuda.device_count(),  # 多卡自动启用 TP
            )
            cls._model_name_loaded = model_name
        return cls._llm_instance


    def generate_answer(
        self,
        queries: List[str],
        retrieved_docs_list: List[List[dict]]
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Generate answers with context-aware, structured prompting.
        """
        if len(queries) != len(retrieved_docs_list):
            raise ValueError("queries 与 retrieved_docs_list 长度不一致")

        prompts: List[str] = []
        for q, docs in zip(queries, retrieved_docs_list):
            # 1) 拼接前 10 篇文档内容，加标注
            context_blocks = []
            for i, d in enumerate(docs[:10]):
                source = d.get("source", "Unknown")  # optional metadata
                context_blocks.append(f"Document {i+1} [{source}]:\n{d['content'].strip()}")

            raw_context = "\n\n".join(context_blocks) or "No relevant documents were retrieved."

            # 2) token 截断
            tokens = self.tokenizer.encode(raw_context, add_special_tokens=False)
            if len(tokens) > self.MAX_EVIDENCE_TOKENS:
                tokens = tokens[: self.MAX_EVIDENCE_TOKENS]
            context = self.tokenizer.decode(tokens, skip_special_tokens=True)

            # 3) Prompt 构造（支持多轮 reasoning + boxed 格式输出）
            prompt = textwrap.dedent(f"""
                You are a professional question answering system that uses evidence to compute final answers.
                Analyze the evidence step by step, and conclude your answer in the format: \\boxed{{final_answer}}.

                - Use only the given evidence.
                - If the answer is not found, say: \\boxed{{Not answerable}}.
                - If multiple numbers or facts are relevant, perform reasoning before answering.

                ### Evidence
                {context}

                ### Question
                {q}

                ### Answer (step-by-step, then final boxed answer)
            """).strip()

            prompts.append(prompt)

        # 4) vLLM batch 推理
        outputs = self.llm.generate(prompts, sampling_params=self._samp)
        torch.cuda.empty_cache()

        # 5) 整理返回结果
        results: List[Dict[str, Union[str, int]]] = []
        for i, out in enumerate(outputs):
            answer_txt = out.outputs[0].text.strip() if out.outputs else "No response."
            results.append({
                "query": queries[i],
                "prompt": prompts[i],
                "answer": answer_txt,
                "documents_used": len(retrieved_docs_list[i]),
            })
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="The vLLM model identifier to load"
    )
    parser.add_argument(
        "queries", nargs='+', type=str,
        help="One or more queries to answer"
    )
    args = parser.parse_args()

    reader = HuggingFaceReader(args.model_name)
    results = reader.generate_answer(args.queries, [[] for _ in args.queries])
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
