#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse extra HTML files and emit ‘distractor’ corpus in SAME format
as corpus_text_final.jsonl / corpus_tables_final.jsonl /
corpus_infobox_final.jsonl, 并在读取时对 TEXT 做 ≤100-token 切块。
"""

import os, json, re, itertools
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
import spacy
from transformers import AutoTokenizer

# ---------- chunking 配置 ----------
MAX_TOK_CHUNK = 100
OVERLAP_TOK   = 0

# 加载 sentence-splitter
nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")
# 用于计算 token 长度
_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

def sent_split(t: str) -> list[str]:
    return [s.text.strip() for s in nlp(t).sents if s.text.strip()]

def token_len(t: str) -> int:
    return len(_tokenizer(t, add_special_tokens=False)["input_ids"])

def chunk_text(text: str) -> list[str]:
    sents = sent_split(text)
    if not sents:
        return []
    max_tok = MAX_TOK_CHUNK
    overlap = min(OVERLAP_TOK, max_tok - 1)
    chunks, cur, tok = [], [], 0
    for s in sents:
        l = token_len(s)
        if cur and tok + l > max_tok:
            chunks.append(" ".join(cur))
            if overlap:
                tail_ids = _tokenizer(" ".join(cur), add_special_tokens=False)["input_ids"][-overlap:]
                ov = _tokenizer.decode(tail_ids)
                cur, tok = [ov], token_len(ov)
            else:
                cur, tok = [], 0
        cur.append(s)
        tok += l
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# ---------- util ----------
def extract_infobox(soup: BeautifulSoup):
    for table in soup.find_all('table'):
        if 'infobox' in ' '.join(table.get('class', [])).lower():
            rows, info, cur = table.find_all('tr'), {}, None
            for tr in rows:
                th, td = tr.find('th'), tr.find('td')
                if th and td:
                    h = th.get_text(" ", strip=True)
                    d = ' | '.join(td.stripped_strings)
                    info.setdefault(cur or h, {})[h] = d
                elif th:  # section header
                    cur = th.get_text(" ", strip=True)
                    info[cur] = {}
            return info
    return None

def extract_text(soup: BeautifulSoup):
    main = soup.find('div', class_='mw-parser-output')
    if not main:
        return ""
    stop = {"references", "external links", "see also", "further reading", "notes"}
    out = []
    for el in main.find_all(['h2','h3','h4','p','ul','ol','li'], recursive=True):
        if el.name in {'h2','h3','h4'}:
            sec = el.get_text(" ", strip=True)
            if sec.lower() in stop:
                break
            out.append(f"## {sec}")
        else:
            t = el.get_text(" ", strip=True)
            if t:
                out.append(t)
    return "\n\n".join(out)

def extract_tables(soup: BeautifulSoup):
    main = soup.find('div', class_='mw-parser-output')
    if not main:
        return []
    tbls = []
    for tb in main.find_all('table'):
        cls = ' '.join(tb.get('class', [])).lower()
        if 'wikitable' in cls or 'sortable' in cls:
            hdr = [th.get_text(" ", strip=True) for th in tb.find_all('th')]
            rows = [
                [c.get_text(" ", strip=True) for c in tr.find_all(['td','th'])]
                for tr in tb.find_all("tr")[1:]
            ]
            tbls.append({"columns": hdr, "rows": rows})
    return tbls

# ---------- core ----------
def process_html_folder(html_dir, t_out, i_out, tbl_out):
    id_counters = dict(text=0, infobox=0, table=0)

    def new_id(kind: str) -> str:
        id_counters[kind] += 1
        return f"{kind}_exp_{id_counters[kind]:06d}"

    with open(t_out,  'w', encoding='utf-8') as f_text, \
         open(i_out,  'w', encoding='utf-8') as f_info, \
         open(tbl_out,'w', encoding='utf-8') as f_tbl:

        for fname in tqdm(os.listdir(html_dir), desc="Expanded HTML"):
            if not fname.endswith('.html'):
                continue

            title = fname[:-5].replace('_', ' ')
            url   = f"https://en.wikipedia.org/wiki/{fname[:-5]}"

            html_path = Path(html_dir) / fname
            soup = BeautifulSoup(html_path.read_text(encoding='utf-8'), 'html.parser')

            # ---------- TEXT with chunking ----------
            full_text = extract_text(soup)
            for chunk in chunk_text(full_text):
                f_text.write(json.dumps({
                    "id":      new_id("text"),
                    "title":   title,
                    "url":     url,
                    "content": chunk
                }, ensure_ascii=False) + "\n")

            # ---------- INFOBOX ----------
            ib = extract_infobox(soup)
            if ib:
                f_info.write(json.dumps({
                    "id":      new_id("infobox"),
                    "title":   title,
                    "url":     url,
                    "content": json.dumps(ib, ensure_ascii=False)
                }, ensure_ascii=False) + "\n")

            # ---------- TABLES ----------
            for tbl in extract_tables(soup):
                f_tbl.write(json.dumps({
                    "id":      new_id("table"),
                    "title":   title,
                    "url":     url,
                    "content": json.dumps(tbl, ensure_ascii=False)
                }, ensure_ascii=False) + "\n")

# ---------- CLI ----------
if __name__ == "__main__":
    html_dir = "wikipedia_expanded_html"
    process_html_folder(
        html_dir,
        t_out   = "extended_corpus/text/expanded_corpus_text.jsonl",
        i_out   = "extended_corpus/infobox/expanded_corpus_infobox.jsonl",
        tbl_out = "extended_corpus/table/expanded_corpus_tables.jsonl"
    )
