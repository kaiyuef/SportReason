#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust merger for multiple QA-style datasets (json / jsonl).
边读边写：流式标准化并写出 JSONL，同时统计四种 question_category 数量。
修订日期: 2025-05-04
"""

import json
import hashlib
import re
import random
from pathlib import Path
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SAMPLE_SIZE     = None          # 每个数据集抽样数量；None=全部
RANDOM_SAMPLING = False         # True ➜ random.sample
INPUT_FILES     = [
    "dataset_inprocess/merged_temptableqa.jsonl",
    "dataset_inprocess/new_hybrid_optimized_final.jsonl",
    "dataset_inprocess/new_TANQ_gemini_optimized4.jsonl",
    "dataset_inprocess/new_TANQ_gemini_optimized5.jsonl",
    "dataset_inprocess/new_TANQ_gemini_optimized2.jsonl",
    "dataset_inprocess/new_TANQ_gemini_optimized3.jsonl",
    
]
OUTPUT_FILE     = "dataset/merged_dataset_5_4_1.jsonl"
BAD_LINES_LOG   = "dataset/bad_lines.log"
MAX_PER_CATEGORY  = 600   

_DROP_FIELDS = {
    "seed_question", "seed_answers", "seed_answer", "seed_dataset",
    "answers", "used_evidences", "gold_evidences", "gold_evidence_ids",
    "gold_evidence_type", "id", "reasoning_type",
    "_source_file_index", "_src_path"
}

# ---------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------
sample_id_pool: set[str] = set()
evidence_registry: Dict[str, Tuple[str,int]] = {}
prefix_counters: defaultdict[str,int] = defaultdict(int)

# ---------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------
def next_sample_id() -> str:
    sid = f"sample_{len(sample_id_pool)+1}"
    sample_id_pool.add(sid)
    return sid

def sha1(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()

def serialise(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)

_prefix_pat = re.compile(r"^([A-Za-z]+)")
def infer_prefix(eid: str, ev_dict: dict|None=None) -> str:
    if "_" in eid:
        return eid.split("_",1)[0]
    m = _prefix_pat.match(eid)
    if m:
        return m.group(1)
    if ev_dict and ev_dict.get("type"):
        return ev_dict["type"]
    return "misc"

def new_evidence_id(old_id: str, ev_dict: dict|None=None) -> str:
    prefix = infer_prefix(old_id, ev_dict)
    prefix_counters[prefix] += 1
    return f"{prefix}_{prefix_counters[prefix]:06d}"

def load_dataset(path: str, idx: int) -> List[dict]:
    data, bad = [], []
    def _record_bad(ln, snippet):
        bad.append((ln, snippet))
    if path.endswith(".jsonl"):
        buf = ""
        with open(path, encoding="utf-8") as fp:
            for ln_no, line in enumerate(fp,1):
                if not line.strip(): continue
                buf += line
                level, in_str, esc = 0, False, False
                for ch in buf:
                    if esc:
                        esc = False; continue
                    if ch == '\\':
                        esc = True; continue
                    if ch == '"':
                        in_str = not in_str; continue
                    if in_str:
                        continue
                    if ch=='{': level+=1
                    elif ch=='}': level-=1
                if level == 0:
                    try:
                        obj = json.loads(buf)
                    except json.JSONDecodeError:
                        _record_bad(ln_no, buf[:120])
                        buf = ""
                        continue
                    obj["_source_file_index"] = idx
                    data.append(obj)
                    buf = ""
        if buf:
            _record_bad("EOF", buf[:120])
    else:
        with open(path, encoding="utf-8") as fp:
            raw = json.load(fp)
        for obj in raw:
            obj["_source_file_index"] = idx
            data.append(obj)

    if bad:
        with open(BAD_LINES_LOG, "a", encoding="utf-8") as flog:
            for ln, snippet in bad:
                flog.write(f"{path}  line:{ln}\n{snippet}\n\n")
    return data

def detect_evidence_type(eid: str, ev: dict|None=None) -> str:
    if ev and "type" in ev:
        return ev["type"]
    p = infer_prefix(eid, ev)
    return p if p in {"table","infobox","text"} else "text"

def process_evidence(ev: dict, src_idx: int) -> Tuple[str,dict]:
    ev = deepcopy(ev)
    raw = ev.get("content", ev.get("evidence_text",""))
    content_str = serialise(raw)
    h = sha1(content_str)

    eid = ev.get("id") or new_evidence_id(ev.get("type","misc"), ev)
    if eid in evidence_registry:
        prev_hash, prev_src = evidence_registry[eid]
        if prev_src!=src_idx or prev_hash!=h:
            eid = new_evidence_id(eid, ev)

    evidence_registry[eid] = (h, src_idx)
    ev["id"] = eid
    if isinstance(ev.get("content"), (dict,list)):
        ev["content"] = content_str
    return eid, ev

def standardise_sample(raw: dict) -> dict:
    sid = next_sample_id()
    std = {
        "id": sid,
        "seed_question": raw.get("seed_question","") or raw.get("question",""),
        "seed_dataset": raw.get("seed_dataset","") or Path(raw.get("_src_path","")).stem,
        "seed_answers": raw.get("seed_answers") or ([raw["seed_answer"]] if "seed_answer" in raw else []),
        "answers":     raw.get("answers") or ([raw["seed_answer"]] if "seed_answer" in raw else []),
        "reasoning_type": raw.get("reasoning_type","")
    }

    ev_list = raw.get("gold_evidences") or raw.get("used_evidences") or []
    gold_ids, gold_objs, type_counter = [], [], Counter()
    for ev in ev_list:
        eid, ev_std = process_evidence(ev, raw["_source_file_index"])
        gold_ids.append(eid)
        gold_objs.append(ev_std)
        type_counter[ detect_evidence_type(eid, ev_std) ] += 1

    std["gold_evidence_ids"]  = gold_ids
    std["gold_evidence_type"] = dict(type_counter)
    std["gold_evidences"]     = gold_objs
    std["meta"] = {k:v for k,v in raw.items() if k not in _DROP_FIELDS}
    return std


# ---------------------------------------------------------------------
# 主流程：流式读写 & 统计 question_category（含 temptableqa→single-table）并限量写入
# ---------------------------------------------------------------------

category_counts = defaultdict(int)
valid_categories = {
    "multi-table + multi text",
    "multi-table",
    "single table+multi text",
    "multi-text",
    "single-table"
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for idx, path in enumerate(INPUT_FILES):
        ds = load_dataset(path, idx)
        if SAMPLE_SIZE is not None:
            ds = (random.sample(ds, SAMPLE_SIZE) if RANDOM_SAMPLING else ds[:SAMPLE_SIZE])
        for item in ds:
            item["_src_path"] = path
            std = standardise_sample(item)

            # 1. 确定 question_category
            cat = std["meta"].get("question_category")
            if not cat and std.get("seed_dataset") == "temptableqa":
                cat = "single-table"

            # 2. 只有有效类别才考虑写入，并且限量
            if cat in valid_categories:
                if category_counts[cat] < MAX_PER_CATEGORY:
                    # 把 question_category 写入顶层
                    std["question_category"] = cat
                    # 写出
                    fout.write(json.dumps(std, ensure_ascii=False) + "\n")
                    # 累加该类别计数
                    category_counts[cat] += 1
                else:
                    # 已达到该类别最大数，跳过写入
                    continue
            # 如果不在 valid_categories，直接跳过

# 最终打印统计
print(f"✅ 合并并写出完成：流式写入到 {OUTPUT_FILE}")
print(f"（每类上限 {MAX_PER_CATEGORY} 条）")
print("五种 question_category 写入计数：")
for cat in [
    "multi-table + multi text",
    "multi-table",
    "single table+multi text",
    "multi-text",
    "single-table"
]:
    print(f"  {cat}: {category_counts.get(cat, 0)}")
