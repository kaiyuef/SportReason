#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sqlite3
import json
from pathlib import Path
from pprint import pprint

# —— STEP 1: 配置你的路径 —— 
db_path = Path("numsports/indexes/jinaai_jina-embeddings-v3/infobox_index/chroma.sqlite3")

# —— STEP 2: 打开数据库 —— 
conn   = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# —— STEP 3: 列出所有表 —— 
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
print("Found tables:\n", tables, "\n")

def print_table_info(table_name):
    cursor.execute(f"PRAGMA table_info('{table_name}');")
    cols = cursor.fetchall()
    names = [c[1] for c in cols]
    print(f"Columns in '{table_name}': {names}\n")
    return names

# —— STEP 4: 打印关键表的结构 —— 
for tbl in ("embeddings", "embedding_metadata", "embedding_fulltext_search_content"):
    if tbl in tables:
        print_table_info(tbl)
    else:
        print(f"[WARN] no table named '{tbl}'\n")

# —— STEP 5: 拉取并打印前 5 条示例 —— 
print("=== First 5 rows in 'embeddings' ===")
try:
    cursor.execute("SELECT * FROM embeddings LIMIT 5;")
    for row in cursor.fetchall():
        print(row)
except sqlite3.OperationalError as e:
    print("  ", e)
print()

print("=== First 5 rows in 'embedding_metadata' (pivoted) ===")
# grouping metadata by id
cursor.execute("SELECT id, key, string_value, int_value, float_value, bool_value FROM embedding_metadata LIMIT 25;")
meta_rows = cursor.fetchall()
# 聚合成 { id: { key: value, ... }, ... }
grouped = {}
for rid, key, s, i, f, b in meta_rows:
    val = s if s is not None else (i if i is not None else (f if f is not None else b))
    grouped.setdefault(rid, {})[key] = val
for rid, md in list(grouped.items())[:5]:
    print(f"ID={rid} →", md)
print()

print("=== First 5 rows in 'embedding_fulltext_search_content' ===")
try:
    cursor.execute("SELECT rowid, content FROM embedding_fulltext_search_content LIMIT 5;")
    for row in cursor.fetchall():
        rowid, content = row
        snippet = content[:200] + ("…" if len(content) > 200 else "")
        print(f"{rowid}: {snippet}")
except sqlite3.OperationalError as e:
    print("  ", e)
print()

conn.close()
