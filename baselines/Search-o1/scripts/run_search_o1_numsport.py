#!/usr/bin/env python3
"""
Async Search‑O1 runner (v2.0‑async‑jina, 2025‑05‑17)
─────────────────────────────────────────────────────
- 仍使用 AsyncOpenAI 进行并发调用（无 vLLM 依赖）
- 支持多数据集、多 prompt 逻辑（GPQA / NQ / HotpotQA / Math500 / LiveCode…）
- 集成 Jina API / 原生 HTTP 双模式网页抓取
- Bing/搜索、URL 缓存 与 token‑budget 修剪保持异步实现
"""

from __future__ import annotations
import argparse, asyncio, json, os, re, time
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

import httpx
from openai import AsyncOpenAI
from tqdm import tqdm

# 👉 prompt 构造函数，与 run_search_o1.py 相同
from prompts import (
    get_gpqa_search_o1_instruction, get_math_search_o1_instruction,
    get_code_search_o1_instruction, get_singleqa_search_o1_instruction,
    get_multiqa_search_o1_instruction, get_webpage_to_reasonchain_instruction,
    get_task_instruction_openqa, get_task_instruction_math,
    get_task_instruction_multi_choice, get_task_instruction_code,
)

from bing_search import (
    bing_web_search, extract_relevant_info,
    extract_snippet_with_context,      # 复用原实现
)

from evaluate import run_evaluation    # ↓ 如需评估仍可使用
try:
    import tiktoken
except ImportError:
    tiktoken = None


# ------------------------------ 常量 ------------------------------
BEGIN_SEARCH_QUERY  = "<|begin_search_query|>"
END_SEARCH_QUERY    = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT   = "<|end_search_result|>"

MODEL_CTX_LIMIT = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 8_000,
    "gpt-4o-mini-128k": 128_000,
    "gpt-4-turbo": 128_000,
}
RETRY_DELAYS = [2, 4, 8, 16]   # 指数回退

# ------------------------------ CLI ------------------------------
def parse_args():
    p = argparse.ArgumentParser("Async Search‑O1 (OpenAI + Jina)")

    # 数据集 / 文件输入
    p.add_argument("--dataset_name", choices=[
        'gpqa','math500','aime','amc','livecode','nq','triviaqa','hotpotqa',
        '2wiki','musique','bamboogle','num_sports_500'
    ])
    p.add_argument("--split",           choices=['test','diamond','main','extended'], default='test')
    p.add_argument("--data_file",       help="若直接给定 jsonl 路径，则忽略 dataset_name/split")
    p.add_argument("--subset_num", type=int, default=-1)

    # OpenAI 模型 & 搜索参数
    p.add_argument("--openai_model", required=True)
    p.add_argument("--openai_api_key")
    p.add_argument("--openai_base_url")
    p.add_argument("--bing_subscription_key", required=True)
    p.add_argument("--bing_endpoint", default="https://api.bing.microsoft.com/v7.0/search")

    # 生成 & 搜索限制
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p",       type=float, default=0.8)
    p.add_argument("--max_turn",    type=int,   default=15)
    p.add_argument("--max_search_limit", type=int, default=10)
    p.add_argument("--top_k",       type=int, default=10)
    p.add_argument("--max_doc_len", type=int, default=3000)

    # 并发/批处理
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--batch_size",  type=int, default=5)

    # Jina 网页抓取
    p.add_argument("--use_jina",    action="store_true", help="启用 Jina AI Reader")
    p.add_argument("--jina_api_key", type=str, default=None)

    return p.parse_args()


# ------------------------------ token 工具 ------------------------------
def count_tokens(text: str, model: str) -> int:
    if not tiktoken:
        return len(text)//4
    enc = (tiktoken.encoding_for_model(model)
           if model in tiktoken.list_encoding_names()
           else tiktoken.get_encoding("cl100k_base"))
    return len(enc.encode(text))

def trim_messages(messages: List[Dict[str,str]], model: str, budget: int) -> List[Dict[str,str]]:
    """在 message 数组中过度超长时裁剪第二条、第三条…（保留开头&最新）"""
    if not tiktoken: return messages
    while count_tokens(" ".join(m["content"] for m in messages), model) > budget and len(messages) > 2:
        messages.pop(1)
    return messages


# ------------------------------ 辅助函数 ------------------------------
def extract_between_fuzzy(text: str, start: str, end: str|None=None) -> Optional[str]:
    try: s = text.rindex(start) + len(start)
    except ValueError: return None
    if end and (idx:=text.find(end, s))!=-1:
        return text[s:idx].strip()
    remainder = text[s:]
    return remainder.splitlines()[0].strip() if remainder else None

def merge_steps(old: str, new: str) -> str:
    """参照 run_search_o1.py 的 replace_recent_steps (删/替换) 的轻量并集版本"""
    pat = re.compile(r"^Step\s+(\d+):", re.MULTILINE)
    def _parse(txt):
        d={}
        for m in pat.finditer(txt):
            k=int(m.group(1)); start=m.end(); nxt=pat.search(txt,start)
            d[k]=txt[start:(nxt.start() if nxt else None)].strip()
        return d
    A,B=_parse(old),_parse(new)
    if not B: return old+"\n\n"+new
    for k,v in B.items():
        if "DELETE THIS STEP" in v:
            A.pop(k, None)
        else:
            A[k]=v
    return "\n\n".join(f"Step {k}: {v}" for k,v in sorted(A.items()))


# ------------------------------ Jina / HTTP 抓取 ------------------------------
async def http_fetch(session:httpx.AsyncClient, url:str, timeout:int=20)->str:
    r=await session.get(url, timeout=timeout, follow_redirects=True)
    r.raise_for_status(); return r.text

async def jina_fetch(session:httpx.AsyncClient, url:str, jina_key:str)->str:
    hdr={"Authorization":f"Bearer {jina_key}"} if jina_key else {}
    # Jina Reader: https://r.jina.ai/http://<url>
    api=f"https://r.jina.ai/http://{url}"
    r=await session.get(api, headers=hdr, timeout=30)
    r.raise_for_status(); return r.text

async def fetch_page_contents(urls:Set[str], concurrency:int,
                              use_jina:bool, jina_key:Optional[str])->Dict[str,str]:
    sem=asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as session:
        async def _task(u):
            async with sem:
                try:
                    return u, (await (jina_fetch(session,u,jina_key) if use_jina else http_fetch(session,u)))
                except Exception: return u,""
        tasks=[_task(u) for u in urls]
        results=await asyncio.gather(*tasks)
    return {u:txt for u,txt in results}


# ------------------------------ OpenAI completion ------------------------------
async def async_safe_completion(client, messages, model, temperature, top_p,
                                max_tokens, semaphore, stop=None):
    async with semaphore:
        ctx_limit = MODEL_CTX_LIMIT.get(model, 128000)
        messages = trim_messages(messages, model, ctx_limit-max_tokens)
        for delay in [0]+RETRY_DELAYS:
            try:
                kwargs=dict(model=model, messages=messages, temperature=temperature,
                            top_p=top_p, max_tokens=max_tokens)
                if stop: kwargs["stop"]=stop
                resp=await client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                # 跳过 bad prompt
                if hasattr(e,"status_code") and e.status_code==400 and "invalid_prompt" in str(e):
                    print(f"⚠️ Skip invalid prompt: {str(e)[:80]}"); return ""
                if delay==RETRY_DELAYS[-1] or getattr(e,"status_code",None) not in {429,500,502,503,504}:
                    print("❌ Fatal:",e); return ""
                print("⏳ Retry:",e); await asyncio.sleep(delay)
        print("❌ Fatal: all retries failed")
        return ""


# ------------------------------ 主程序 ------------------------------
async def main_async():
    args=parse_args()

    # -------------- 数据加载 --------------
    if args.data_file:
        with open(args.data_file,'r',encoding='utf-8') as f:
            raw_data=[json.loads(l) for l in f if l.strip()]
    else:
        # 根据 dataset_name / split 读取
        d=args.dataset_name; sp=args.split
        if d=='livecode':
            path=f'./data/LiveCodeBench/{sp}.json'
        elif d in ['math500','gpqa','aime','amc']:
            path=f'./data/{d.upper()}/{sp}.json'
        elif dataset_name == 'num_sports':
            data_path = f'baselines/Search-o1/data/num_sports_500.jsonl'
        else:
            path=f'./data/QA_Datasets/{d}.json'
        with open(path,'r',encoding='utf-8') as f:
            raw_data=json.load(f)
    if args.subset_num>0: raw_data=raw_data[:args.subset_num]

    # -------------- OpenAI client --------------
    client=AsyncOpenAI(api_key=args.openai_api_key or os.getenv("OPENAI_API_KEY"),
                       base_url=args.openai_base_url)
    gen_tokens=min(4096, MODEL_CTX_LIMIT.get(args.openai_model,128_000)//8)

    # -------------- 缓存 --------------
    cache_dir=Path("cache"); cache_dir.mkdir(exist_ok=True)
    search_cache= json.loads((cache_dir/"search_cache.json").read_text("utf-8")) \
                    if (cache_dir/"search_cache.json").exists() else {}
    url_cache   = json.loads((cache_dir/"url_cache.json").read_text("utf-8")) \
                    if (cache_dir/"url_cache.json").exists() else {}

    # -------------- 输出目录 --------------
    out_dir=Path("outputs")/f"{args.dataset_name or Path(args.data_file).stem}.{args.openai_model}.search_o1.async"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------- 构造 prompt / 序列 --------------
    def build_instruction_and_user(item):
        # 支持 dataset_name；若用户仅给 data_file 则默认为 single‑QA
        d=args.dataset_name
        q=item['Question'] if 'Question' in item else item.get('seed_question','')
        if not d:   # 回退
            instr=get_singleqa_search_o1_instruction(args.max_search_limit)
            user = get_task_instruction_openqa(q)
            return instr+user

        if d in ['nq','triviaqa']:
            instr=get_singleqa_search_o1_instruction(args.max_search_limit)
            user = get_task_instruction_openqa(q)
        elif d in ['hotpotqa','musique','bamboogle','2wiki','num_sports_500']:
            instr=get_multiqa_search_o1_instruction(args.max_search_limit)
            user = get_task_instruction_openqa(q)
        elif d in ['math500','aime','amc']:
            instr=get_math_search_o1_instruction(args.max_search_limit)
            user = get_task_instruction_math(q)
        elif d=='gpqa':
            instr=get_gpqa_search_o1_instruction(args.max_search_limit)
            user = get_task_instruction_multi_choice(q)
        elif d=='livecode':
            instr=get_code_search_o1_instruction(args.max_search_limit)
            user = get_task_instruction_code(q, question_title=item.get('question_title',""))
        else:
            instr=get_singleqa_search_o1_instruction(args.max_search_limit)
            user = get_task_instruction_openqa(q)
        return instr+user

    sequences=[{
        "item": itm,
        "prompt": build_instruction_and_user(itm),
        "output": "",
        "history": [],
        "search_count": 0,
        "executed_queries": set(),
        "finished": False,
    } for itm in raw_data]

    # -------------- 异步并发 --------------
    semaphore=asyncio.Semaphore(args.concurrency)
    start_time=time.time()

    for turn in range(1, args.max_turn+1):
        pending=[s for s in sequences if not s["finished"]]
        if not pending: break
        print(f"🌀 Turn {turn} – {len(pending)} active")

        # -------- Generate LLM replies --------
        async def gen(seq):
            reply=await async_safe_completion(
                client, [{"role":"system","content":"You are a helpful reasoner."},
                         {"role":"user","content":seq["prompt"]}],
                args.openai_model, args.temperature, args.top_p, gen_tokens, semaphore,
            )
            if not reply: seq["finished"]=True
            else:
                seq["history"].append(reply)
                seq["prompt"] += reply
                seq["output"] += reply
        await asyncio.gather(*[gen(s) for s in pending])

        # -------- Collect search queries --------
        query_pairs=[]
        for s in pending:
            q=extract_between_fuzzy(s["output"], BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
            if q and s["search_count"]<args.max_search_limit and q not in s["executed_queries"]:
                query_pairs.append((s,q))

        # -------- Execute Bing search (cached) --------
        for seq,q in query_pairs:
            res=search_cache.get(q)
            if res is None:
                try:
                    res=bing_web_search(q,args.bing_subscription_key,args.bing_endpoint)
                except Exception as e:
                    print(f"🔴 Bing error '{q[:50]}': {e}"); res={}
                search_cache[q]=res
            info=extract_relevant_info(res)[:args.top_k]
            seq["relevant_info"]=info
            seq["search_count"]+=1
            seq["executed_queries"].add(q)

        # -------- Batch fetch URLs with Jina / HTTP --------
        new_urls={it["url"] for seq,_ in query_pairs for it in seq.get("relevant_info",[]) if it["url"] not in url_cache}
        if new_urls:
            fetched=await fetch_page_contents(new_urls, args.concurrency,
                                              args.use_jina, args.jina_api_key)
            url_cache.update(fetched)

        # -------- Reasoning over webpages --------
        async def gen_reason(seq):
            docs=""
            for idx,info in enumerate(seq.get("relevant_info",[])):
                url=info.get("url",""); snippet=info.get("snippet","")
                raw=url_cache.get(url,"")
                _,ctx=extract_snippet_with_context(raw,snippet,args.max_doc_len)
                docs+=f"**Web Page {idx+1}:**\n{json.dumps(info,ensure_ascii=False,indent=2)}\n{ctx}\n"
            reasoning = await async_safe_completion(
                client,
                [{"role":"system","content":"You are a helpful reasoner."},
                 {"role":"user","content":
                    get_webpage_to_reasonchain_instruction(seq["output"],
                                                           seq['item'].get('Question',seq['item'].get('seed_question','')),
                                                           docs)}],
                args.openai_model, args.temperature, args.top_p, gen_tokens, semaphore,
            )
            if reasoning.startswith("Step"):  # 标准增补
                seq["output"] += "\n\n"+reasoning
            else:                              # 合并
                seq["output"]=merge_steps(seq["output"], reasoning)
        await asyncio.gather(*[gen_reason(s) for s,_ in query_pairs])

        # -------- 终止条件 --------
        for s in pending:
            if turn==args.max_turn or s["search_count"]>=args.max_search_limit:
                s["finished"]=True

    # -------------- 保存输出 & 评估 --------------
    records=[{
        "id":s["item"].get("id",None),
        "question":s["item"].get("Question",s["item"].get("seed_question","")),
        "answer_chain":s["output"],
        "search_count":s["search_count"]
    } for s in sequences]

    ts=time.strftime("%m.%d-%H%M")
    out_path=out_dir/f"reasoning_records.{ts}.json"
    json.dump(records, open(out_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print("✅ Saved reasoning to", out_path)

    # 可选评估
    outputs=[r["answer_chain"] for r in records]
    input_list=[s['item'].get('Question',s['item'].get('seed_question','')) for s in sequences]
    run_evaluation(raw_data, input_list, outputs,
                   args.dataset_name or Path(args.data_file).stem,
                   str(out_dir), time.time()-start_time, args.split if args.dataset_name else "all")

    # 缓存更新
    (cache_dir/"search_cache.json").write_text(json.dumps(search_cache,ensure_ascii=False,indent=2))
    (cache_dir/"url_cache.json").write_text(json.dumps(url_cache,ensure_ascii=False,indent=2))
    print("🎉 Done.")


if __name__=="__main__":
    asyncio.run(main_async())
