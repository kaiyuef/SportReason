#!/usr/bin/env python3
"""
Improved Search‑O1 runner (v1.1, 2025‑05‑11)
────────────────────────────────────────────
Key fixes
─────────
1. *Stop‑token bug* — 允许模型自行输出 `<|end_search_query|>`，解析才能成功。
2. *Fuzzy parser*   — `extract_between_fuzzy` 能容忍缺失 end‑token。
3. *Sequence liveness* — 只有达到 max_turn 才终止，避免过早 `finished=True`。
4. 细节：重构 token 计数、日志，删除未用 import。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import httpx                    # async HTTP client
from openai import OpenAI, OpenAIError
from tqdm import tqdm

# Third‑party helpers
from bing_search import (
    bing_web_search,
    extract_relevant_info,
    extract_snippet_with_context,
)
from evaluate import run_evaluation
from prompts import (
    get_singleqa_search_o1_instruction,
    get_webpage_to_reasonchain_instruction,
    get_task_instruction_openqa,
)

# Optional token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None


# ────────────────────────────── 常量 ──────────────────────────────
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY   = "<|end_search_query|>"

MODEL_CTX_LIMIT = {
    "gpt-4o":            128000,
    "gpt-4o-mini":        8000,
    "gpt-4o-mini-128k": 128000,
    "gpt-4-turbo":      128000,
}
RETRY_DELAYS = [1, 2, 4, 8]


# ──────────────────────────── CLI 解析 ────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Run Search‑O1 via OpenAI API (fixed)")
    p.add_argument("--data_file",           required=True)
    p.add_argument("--openai_model",        required=True)
    p.add_argument("--openai_api_key",      help="Overrides env OPENAI_API_KEY")
    p.add_argument("--openai_base_url")
    p.add_argument("--bing_subscription_key", required=True)
    p.add_argument("--bing_endpoint",
                   default="https://api.bing.microsoft.com/v7.0/search")
    p.add_argument("--subset_num",   type=int, default=-1)
    p.add_argument("--temperature",  type=float, default=0.7)
    p.add_argument("--top_p",        type=float, default=0.8)
    p.add_argument("--max_turn",     type=int, default=15)
    p.add_argument("--max_search_limit", type=int, default=10)
    p.add_argument("--top_k",        type=int, default=10)
    p.add_argument("--max_doc_len",  type=int, default=3000)
    p.add_argument("--concurrency",  type=int, default=20)
    p.add_argument("--use_jina",     action="store_true")
    p.add_argument("--jina_api_key")
    return p.parse_args()


# ──────────────────────────── 工具函数 ────────────────────────────
def count_tokens(text: str, model: str) -> int:
    """粗略或精确计算 token 数。"""
    if not tiktoken:
        return len(text) // 4
    enc = (tiktoken.encoding_for_model(model)
           if model in tiktoken.list_encoding_names()
           else tiktoken.get_encoding("cl100k_base"))
    return len(enc.encode(text))


def trim_messages(messages: List[Dict[str, str]], model: str,
                  budget: int) -> List[Dict[str, str]]:
    """削减最早的 user turns 以保持在 ctx 预算内（保留 system+最新）。"""
    if not tiktoken:
        return messages
    while count_tokens(" ".join(m["content"] for m in messages), model) > budget \
            and len(messages) > 2:
        messages.pop(1)
    return messages


def safe_completion(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: Optional[List[str]] = None,
) -> str:
    """Wrapper that retries and never returns None (defaults to empty string)."""
    ctx_limit = MODEL_CTX_LIMIT.get(model, 128000)
    messages = trim_messages(messages, model, ctx_limit - max_tokens)
    for delay in [0] + RETRY_DELAYS:
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            if stop:
                kwargs["stop"] = stop

            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            return content or ""
        except OpenAIError as e:
            if delay == RETRY_DELAYS[-1] or getattr(e, "status_code", None) not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(delay)
    return ""



# === PATCH 1: 更宽容的 parser ====================================
def extract_between_fuzzy(text: str, start: str,
                          end: str | None = None) -> Optional[str]:
    """
    Return substring between `start` and `end`.
    If `end` missing, take up to first newline / EOF.
    """
    try:
        s = text.rindex(start) + len(start)
    except ValueError:
        return None
    if end and (idx := text.find(end, s)) != -1:
        return text[s:idx].strip()
    remainder = text[s:]
    return remainder.splitlines()[0].strip() if remainder else None
# ===============================================================


def merge_steps(old: str, new: str) -> str:
    """Merge 'Step N:' blocks, deduping by step index."""
    pat = re.compile(r"^Step\s+(\d+):", re.MULTILINE)

    def _parse(txt: str):
        d = {}
        for m in pat.finditer(txt):
            k = int(m.group(1))
            start = m.end()
            nxt = pat.search(txt, start)
            d[k] = txt[start: (nxt.start() if nxt else None)].strip()
        return d or None

    A, B = _parse(old), _parse(new)
    if not A or not B:
        return old + "\n\n" + new
    A.update({k: v for k, v in B.items() if "DELETE THIS STEP" not in v})
    return "\n\n".join(f"Step {k}: {v}" for k, v in sorted(A.items()))


async def fetch_one(url: str, timeout: int = 20) -> str:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as cli:
        r = await cli.get(url)
        r.raise_for_status()
        return r.text


async def fetch_page_contents(urls: Set[str], concurrency: int = 20) -> Dict[str, str]:
    sem = asyncio.Semaphore(concurrency)

    async def _task(u):
        async with sem:
            try:
                return u, await fetch_one(u)
            except Exception:
                return u, ""

    return {u: txt for u, txt in await asyncio.gather(*(_task(u) for u in urls))}


# ─────────────────────────────── 主流程 ───────────────────────────────
def main():
    args = parse_args()

    # ── OpenAI client ──
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("❌  OPENAI_API_KEY missing (flag or env)")
    client = OpenAI(api_key=api_key, base_url=args.openai_base_url)

    model_ctx  = MODEL_CTX_LIMIT.get(args.openai_model, 128000)
    gen_tokens = min(4096, model_ctx // 8)

    # ── Load dataset ──
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    if args.subset_num > 0:
        data = data[: args.subset_num]

    # ── Caches & output dir ──
    cache_dir    = Path("cache"); cache_dir.mkdir(exist_ok=True)
    search_cache = json.loads((cache_dir / "search_cache.json").read_text("utf-8")
                              ) if (cache_dir / "search_cache.json").exists() else {}
    url_cache    = json.loads((cache_dir / "url_cache.json").read_text("utf-8")
                              ) if (cache_dir / "url_cache.json").exists() else {}

    out_dir = Path("outputs") / f"{Path(args.data_file).stem}.{args.openai_model}.search_o1.fixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Sequence objects ──
    sequences = []
    for item in data:
        prompt = (
            get_singleqa_search_o1_instruction(args.max_search_limit) +
            get_task_instruction_openqa(item["seed_question"])
        )
        sequences.append({
            "item": item,
            "prompt": prompt,
            "output": "",
            "history": [],
            "search_count": 0,
            "executed_queries": set(),
            "finished": False,
        })

    start_time = time.time()
    records    = []

    # ────────────────── multi‑turn loop ──────────────────
    for turn in range(1, args.max_turn + 1):
        pending = [s for s in sequences if not s["finished"]]
        if not pending:
            break
        print(f"Turn {turn} – {len(pending)} pending")

        # ① 让模型继续思考 / 生成查询
        for seq in pending:
            reply = safe_completion(
                client,
                [
                    {"role": "system", "content": "You are a helpful reasoner."},
                    {"role": "user",   "content": seq["prompt"]},
                ],
                model=args.openai_model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=gen_tokens,
                # === PATCH 2: 删除 stop 让模型输出 end‑token =========
                # stop=[END_SEARCH_QUERY],
            )
            if not reply:
                print(f"⚠️  Empty reply at turn {turn} for ID {seq['item']['id']} → finish")
                seq["finished"] = True
                continue
            seq["history"].append(reply)
            seq["prompt"]  += reply
            seq["output"]  += reply

        # ② 提取 (seq, query) 配对
        query_pairs: List[Tuple[dict, str]] = []
        for seq in pending:
            q = extract_between_fuzzy(seq["output"], BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
            if q and seq["search_count"] < args.max_search_limit and \
               q not in seq["executed_queries"]:
                query_pairs.append((seq, q))
            # === PATCH 3: 不再立刻终结，直到达到 max_turn =========
            # else:
            #     seq["finished"] = True

        # ③ 进行 Bing 搜索
        for seq, q in query_pairs:
            res = search_cache.get(q)
            if res is None:
                try:
                    res = bing_web_search(q, args.bing_subscription_key, args.bing_endpoint)
                except Exception as e:
                    print(f"🔴 Bing error on '{q[:60]}…': {e}")
                    res = {}
                search_cache[q] = res
            info = extract_relevant_info(res)[: args.top_k]
            seq["search_count"]      += 1
            seq["executed_queries"].add(q)
            seq["relevant_info"]      = info

        # ④ 抓网页正文
        new_urls = {it["url"] for seq, _ in query_pairs
                    for it in seq.get("relevant_info", [])
                    if it["url"] not in url_cache}
        if new_urls:
            url_cache.update(asyncio.run(
                fetch_page_contents(new_urls, concurrency=args.concurrency)))

        # ⑤ 结合网页 → 生成 reasoning
        for seq, _ in query_pairs:
            docs = ""
            for idx, info in enumerate(seq.get("relevant_info", [])):
                url     = info.get("url", "")
                snippet = info.get("snippet", "")
                raw     = url_cache.get(url, "")
                _, ctx  = extract_snippet_with_context(raw, snippet, args.max_doc_len)
                docs   += (f"**Web Page {idx+1}:**\n"
                           f"{json.dumps(info, ensure_ascii=False, indent=2)}\n"
                           f"{ctx}\n")

            reasoning = safe_completion(
                client,
                [
                    {"role": "system", "content": "You are a helpful reasoner."},
                    {"role": "user",
                     "content": get_webpage_to_reasonchain_instruction(
                         seq["output"], seq["item"]["seed_question"], docs)},
                ],
                model=args.openai_model,
                temperature=0.7,
                top_p=0.8,
                max_tokens=gen_tokens,
            )
            if reasoning.startswith("Step"):
                seq["output"] += "\n\n" + reasoning
            else:
                seq["output"]  = merge_steps(seq["output"], reasoning)

        # ⑥ 如果该序列本轮没产生新查询且达到 max_turn，则结束
        for seq in pending:
            if turn == args.max_turn or seq["search_count"] >= args.max_search_limit:
                seq["finished"] = True

    # ─────────────────── 保存 & 评估 ───────────────────
    for seq in sequences:
        records.append({
            "id":            seq["item"]["id"],
            "question":      seq["item"]["seed_question"],
            "answer_chain":  seq["output"],
            "search_count":  seq["search_count"],
        })

    ts       = time.strftime("%m.%d-%H%M")
    out_path = out_dir / f"reasoning_records.{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print("✅  Output saved to", out_path)

    # ——— 调用现成 evaluation ———
    outputs     = [r["answer_chain"] for r in records]
    input_list  = [seq["item"]["seed_question"] for seq in sequences]
    run_evaluation(data, input_list, outputs,
                   Path(args.data_file).stem, str(out_dir),
                   time.time() - start_time, "all")

    # ——— 刷新缓存 ———
    (cache_dir / "search_cache.json").write_text(
        json.dumps(search_cache, ensure_ascii=False, indent=2))
    (cache_dir / "url_cache.json").write_text(
        json.dumps(url_cache, ensure_ascii=False, indent=2))

    print("🎉  Done.")


# ──────────────────────────── 入口 ────────────────────────────
if __name__ == "__main__":
    main()
