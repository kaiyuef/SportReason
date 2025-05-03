#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust merger for multiple QA-style datasets (json / jsonl, possibly indented).
Ensures evidence-ID uniqueness with content-aware logic (type_000001 格式)
and outputs JSONL.

修订日期: 2025-04-28
"""

import ast, json, random, hashlib, re
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
OUTPUT_FILE      = "regenerate_combine/reformatted_merged_dataset1.jsonl"
BAD_LINES_LOG    = "regenerate_combine/bad_lines.log"      # 解析失败行

# ---------------------------------------------------------------------
# 全局注册表
# ---------------------------------------------------------------------
sample_id_pool: set[str] = set()

EvidenceInfo = Tuple[str, int]           # (content_hash, source_file_index)
evidence_registry: Dict[str, EvidenceInfo] = {}            # evidence_id → (hash, file_idx)
prefix_counters: defaultdict[str, int] = defaultdict(int)  # 生成新 id 用

# ---------------------- 基础工具 ----------------------
def next_sample_id() -> str:
    idx = len(sample_id_pool) + 1
    sid = f"sample_{idx}"
    sample_id_pool.add(sid)
    return sid

def sha1(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()

def serialise(value) -> str:
    """把 dict/list 转成有序 JSON 字符串；原始 str 保持不变"""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)

# ---------------------- Evidence ID 逻辑 ----------------------
# 旧格式: table000001 / text023, 新格式: table_000001 / text_000023
_prefix_pat = re.compile(r"^([A-Za-z]+)")

def infer_prefix(eid: str, ev_dict: dict | None = None) -> str:
    """优先从 eid 抽取；否则从 ev['type']；再否则 'misc'"""
    if "_" in eid:
        return eid.split("_", 1)[0]
    m = _prefix_pat.match(eid)
    if m:
        return m.group(1)
    if ev_dict and ev_dict.get("type"):
        return ev_dict["type"]
    return "misc"

def new_evidence_id(old_id: str, ev_dict: dict | None = None) -> str:
    """生成同前缀、带下划线的新 id，例: table_000123"""
    prefix = infer_prefix(old_id, ev_dict)
    prefix_counters[prefix] += 1
    return f"{prefix}_{prefix_counters[prefix]:06d}"

# ---------------------- JSON / JSONL 读取 ----------------------
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
    """支持 .json / .jsonl（多行 JSON）"""
    data, bad = [], []
    if path.endswith(".jsonl"):
        buf = ""
        with open(path, encoding="utf-8") as fp:
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
                    obj["_source_file_index"] = idx
                    data.append(obj)
        if buf:  # EOF 残留
            bad.append(("EOF", buf[:120]))
    else:  # .json（列表）
        with open(path, encoding="utf-8") as fp:
            raw_list = json.load(fp)
        for obj in raw_list:
            obj["_source_file_index"] = idx
            data.append(obj)

    if bad:
        with open(BAD_LINES_LOG, "a", encoding="utf-8") as flog:
            for ln, snippet in bad:
                flog.write(f"{path}  line:{ln}\n{snippet}\n\n")
    return data

# ---------------------- Evidence 处理 ----------------------
def detect_evidence_type(eid: str, ev: dict | None = None) -> str:
    """根据 id 或显式字段推断 type"""
    if ev and "type" in ev:
        return ev["type"]
    prefix = infer_prefix(eid, ev)
    if prefix in {"table", "infobox", "text"}:
        return prefix
    return "text"   # 默认

def process_evidence(ev: dict, src_idx: int) -> Tuple[str, dict]:
    """
    归一化 evidence:
      1) content 序列化 ➜ hash
      2) 解决重复 id ➜ 生成新 id（type_000001）
      3) content 保证为 str
    """
    ev = deepcopy(ev)

    # ---------- content / hash ----------
    raw_content = ev.get("content", ev.get("evidence_text", ""))
    content_str = serialise(raw_content)
    h = sha1(content_str)

    # ---------- 基准 id ----------
    if "id" not in ev or not ev["id"]:
        ev["id"] = new_evidence_id(ev.get("type", "misc"), ev)  # 推断前缀
    eid = ev["id"]

    # ---------- 冲突检测 ----------
    if eid in evidence_registry:
        prev_hash, prev_src = evidence_registry[eid]
        need_new = (prev_src != src_idx) or (prev_hash != h)
        if need_new:
            eid = new_evidence_id(eid, ev)

    # ---------- 注册 & 内容归一化 ----------
    evidence_registry[eid] = (h, src_idx)
    ev["id"] = eid
    if isinstance(ev.get("content"), (dict, list)):
        ev["content"] = content_str
    return eid, ev

# ---------------------- 样本标准化 ----------------------
def standardise_sample(raw: dict) -> dict:
    sid = next_sample_id()
    std = {
        "id": sid,
        "seed_question": raw.get("seed_question") or raw.get("question") or "",
        "seed_dataset": raw.get("seed_dataset") or raw.get("source_file")
                        or Path(raw.get("_src_path", "dataset")).stem,
        "seed_answers": (
            raw.get("seed_answers") or
            ([raw["seed_answer"]] if "seed_answer" in raw else [])
        ),
        "answers": (
            raw.get("answers") or
            ([raw["seed_answer"]] if "seed_answer" in raw else [])
        ),
        "reasoning_type": raw.get("reasoning_type", "")
    }

    # ---------- evidence ----------
    ev_list = (
        raw.get("gold_evidences") or
        raw.get("used_evidences") or
        []
    )
    gold_ids, gold_objs, type_counter = [], [], Counter()
    src_idx = raw["_source_file_index"]

    for ev in ev_list:
        eid, ev_std = process_evidence(ev, src_idx)
        gold_ids.append(eid)
        gold_objs.append(ev_std)
        type_counter[detect_evidence_type(eid, ev_std)] += 1

    std["gold_evidence_ids"]  = gold_ids
    std["gold_evidence_type"] = dict(type_counter)
    std["gold_evidences"]     = gold_objs

    # ---------- meta ----------
    drop = {
        "seed_question", "seed_answers", "seed_answer", "seed_dataset",
        "answers", "used_evidences", "gold_evidences", "gold_evidence_ids",
        "gold_evidence_type", "id", "reasoning_type",
        "_source_file_index", "_src_path"
    }
    std["meta"] = {k: v for k, v in raw.items() if k not in drop}

    return std

# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
all_samples: List[dict] = []
for idx, path in enumerate(INPUT_FILES):
    ds = load_dataset(path, idx)
    if SAMPLE_SIZE is not None:
        ds = random.sample(ds, min(SAMPLE_SIZE, len(ds))) if RANDOM_SAMPLING else ds[:SAMPLE_SIZE]
    for item in ds:
        item["_src_path"] = path
        all_samples.append(standardise_sample(item))

# ---------------------- 写出 ----------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
    for obj in all_samples:
        fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ 合并完成：{len(all_samples)} 条样本 ➜ {OUTPUT_FILE}")
print("⚠️ 若 bad_lines.log 存在内容，请检查无法解析的原始行。")
