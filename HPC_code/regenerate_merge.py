#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust merger for multiple QA‑style datasets (json / jsonl, possibly indented).
Ensures evidence‑ID uniqueness with content‑aware logic, keeps prefix format,
and outputs JSONL.

Author: ChatGPT (2025‑04‑20)
"""

import ast
import json
import random
import hashlib
import re
from pathlib import Path
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SAMPLE_SIZE      = 600            # 每个数据集抽样数量；None=全部
RANDOM_SAMPLING  = False          # True ➜ random.sample
INPUT_FILES      = [
    "regenerate_combine/new_hybrid_aligned.jsonl",
    "regenerate_combine/new_TANQ.jsonl",
    "regenerate_combine/merged_temptableqa.jsonl",
]
OUTPUT_FILE      = "regenerate_combine/reformatted_merged_dataset.jsonl"
BAD_LINES_LOG    = "regenerate_combine/bad_lines.log"      # 解析失败行

# ---------------------------------------------------------------------
# 全局注册表
# ---------------------------------------------------------------------
sample_id_pool: set[str] = set()

# evidence 相关结构
EvidenceInfo = Tuple[str, int]    # (content_hash, source_file_index)
evidence_registry: Dict[str, EvidenceInfo] = {}   # 记录已用 evidence_id -> (hash, file_idx)
prefix_counters: defaultdict[str, int] = defaultdict(int)  # 生成新 id 用

def next_sample_id() -> str:
    idx = len(sample_id_pool) + 1
    new_id = f"sample_{idx}"
    sample_id_pool.add(new_id)
    return new_id

def compute_hash(content: str) -> str:
    """sha1 摘要，避免长字符串全量比较。"""
    return hashlib.sha1(content.encode("utf-8")).hexdigest()

_prefix_pat = re.compile(r"^([A-Za-z_]+)")

def new_evidence_id(old_id: str) -> str:
    """
    生成与 old_id 同前缀的新 id，如 table000001
    """
    m = _prefix_pat.match(old_id)
    prefix = m.group(1) if m else "ev"
    prefix_counters[prefix] += 1
    return f"{prefix}{prefix_counters[prefix]:06d}"

# ---------------------------------------------------------------------
# 帮助：读取 json / jsonl（支持缩进）
# ---------------------------------------------------------------------
def balanced(buf: str) -> bool:
    """判断 buf 中 { } 是否配平（忽略字符串字面量）"""
    level, in_str, esc = 0, False, False
    for ch in buf:
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            level += 1
        elif ch == '}':
            level -= 1
    return level == 0

def load_dataset(path: str, idx: int) -> List[dict]:
    """
    读取 json / (单行|多行)jsonl
    idx: 当前文件在 INPUT_FILES 的顺序，用来区分跨文件冲突
    """
    data, bad = [], []
    if path.endswith(".jsonl"):
        buf = ""
        with open(path, "r", encoding="utf-8") as fp:
            for ln_no, line in enumerate(fp, 1):
                if not line.strip():
                    continue
                buf += line
                if balanced(buf):
                    chunk = buf.strip()
                    buf = ""
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        try:
                            obj = ast.literal_eval(chunk)
                        except Exception:
                            bad.append((ln_no, chunk[:120]))
                            continue
                    obj["_source_file_index"] = idx   # 标记来源
                    data.append(obj)
        if buf:  # EOF 仍有残留
            bad.append(("EOF", buf[:120]))
    else:
        with open(path, "r", encoding="utf-8") as fp:
            raw_list = json.load(fp)
        for obj in raw_list:
            obj["_source_file_index"] = idx
            data.append(obj)

    if bad:
        with open(BAD_LINES_LOG, "a", encoding="utf-8") as flog:
            for ln, snippet in bad:
                flog.write(f"{path}  line:{ln}\n{snippet}\n\n")

    return data

# ---------------------------------------------------------------------
# Evidence 工具
# ---------------------------------------------------------------------
def detect_evidence_type(ev: dict) -> str:
    if "type" in ev:
        return ev["type"]
    eid = ev.get("id", "")
    if eid.startswith("table") or eid == "table":
        return "table"
    if eid.startswith("infobox"):
        return "infobox"
    return "text"

def serialise_content(value) -> str:
    """把 dict/list 转成有序 JSON 字符串；字符串保持原样。"""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)

def process_evidence(ev: dict, src_file_idx: int) -> Tuple[str, dict]:
    """
    返回 (最终 evidence_id, evidence_obj)
    - 将 content 转字符串（若是结构）
    - 根据全局规则调整 id
    """
    ev = deepcopy(ev)

    # 1) 归一化 content 用于 hash
    raw_content = (
        ev.get("content")
        if "content" in ev else
        ev.get("evidence_text", "")
    )
    content_str = serialise_content(raw_content)
    content_hash = compute_hash(content_str)

    # 2) 若无 id，给个默认
    if "id" not in ev or not ev["id"]:
        ev["id"] = new_evidence_id("ev")   # 'ev000001' 形式
    eid = ev["id"]

    # 3) 决定是否需要换 id
    if eid in evidence_registry:
        prev_hash, prev_src_idx = evidence_registry[eid]
        if prev_src_idx == src_file_idx:
            # 同一文件 → hash 不同才换 id
            if prev_hash != content_hash:
                eid = new_evidence_id(eid)
        else:
            # 不同文件 → 必换
            eid = new_evidence_id(eid)

    # 4) 更新 registry
    evidence_registry[eid] = (content_hash, src_file_idx)
    ev["id"] = eid

    # 5) 保证 content 是 str（避免嵌套再序列化）——仅当存在 content
    if "content" in ev and isinstance(ev["content"], (dict, list)):
        ev["content"] = content_str

    return eid, ev

# ---------------------------------------------------------------------
# 核心标准化
# ---------------------------------------------------------------------
def standardise_sample(raw: dict) -> dict:
    sid = next_sample_id()
    std = {
        "id": sid,
        "seed_question": raw.get("seed_question") or raw.get("question") or "",
        "seed_dataset": raw.get("seed_dataset") or raw.get("source_file")
                        or Path(raw.get("_src_path", "dataset")).stem,
        "seed_answers": (
            raw.get("seed_answers")
            or ([raw["seed_answer"]] if "seed_answer" in raw else [])
        ),
        "answers": (
            raw.get("answers")
            or ([raw["seed_answer"]] if "seed_answer" in raw else [])
        ),
        "reasoning_type": raw.get("reasoning_type", "")
    }

    # evidence 列表抽取
    if "gold_evidences" in raw:
        ev_list = raw["gold_evidences"]
    else:
        ev_list = raw.get("used_evidences", [])

    gold_ids, gold_objs, type_counter = [], [], Counter()
    src_idx = raw["_source_file_index"]

    for ev in ev_list:
        eid, processed_ev = process_evidence(ev, src_idx)
        gold_ids.append(eid)
        gold_objs.append(processed_ev)
        type_counter[detect_evidence_type(processed_ev)] += 1

    std["gold_evidence_ids"]  = gold_ids
    std["gold_evidence_type"] = dict(type_counter)
    std["gold_evidences"]     = gold_objs

    # meta: 剩余字段
    drop_keys = {
        "seed_question", "seed_answers", "seed_answer", "seed_dataset",
        "answers", "used_evidences", "gold_evidences", "gold_evidence_ids",
        "gold_evidence_type", "id", "reasoning_type",
        "_source_file_index", "_src_path"
    }
    std["meta"] = {k: v for k, v in raw.items() if k not in drop_keys}

    return std

# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
all_samples: List[dict] = []
for idx, path in enumerate(INPUT_FILES):
    dataset = load_dataset(path, idx)

    if SAMPLE_SIZE is not None:
        dataset = (
            random.sample(dataset, min(SAMPLE_SIZE, len(dataset)))
            if RANDOM_SAMPLING else dataset[:SAMPLE_SIZE]
        )

    for item in dataset:
        item["_src_path"] = path
        all_samples.append(standardise_sample(item))

# ---------------------------------------------------------------------
# 写出 JSONL
# ---------------------------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
    for obj in all_samples:
        fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ 合并完成：{len(all_samples)} 条样本 ➜ {OUTPUT_FILE}")
print("⚠️ 若 bad_lines.log 存在内容，请查看无法解析的原始行。")
