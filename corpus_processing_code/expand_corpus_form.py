#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse extra HTML files and emit ‘distractor’ corpus in SAME format
as corpus_text_final.jsonl / corpus_tables_final.jsonl /
corpus_infobox_final.jsonl

输出示例
{
  "id"     : "text_exp_000123",
  "title"  : "Some Title",
  "url"    : "https://en.wikipedia.org/wiki/Some_Title",
  "content": "clean text or JSON-string"
}
"""

import os, json, re
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------- util ----------
def extract_infobox(soup: BeautifulSoup):
    for table in soup.find_all('table'):
        if 'infobox' in ' '.join(table.get('class', [])).lower():
            rows, info, cur = table.find_all('tr'), {}, None
            for tr in rows:
                th, td = tr.find('th'), tr.find('td')
                if th and td:
                    h, d = th.get_text(" ", strip=True), ' | '.join(td.stripped_strings)
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
    stop = {"references", "external links", "see also",
            "further reading", "notes"}
    out = []
    for el in main.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol', 'li'],
                            recursive=True):
        if el.name in {'h2', 'h3', 'h4'}:
            sec = el.get_text(" ", strip=True)
            if sec.lower() in stop:
                break
            out.append(f"## {sec}")
        else:
            t = el.get_text(" ", strip=True)
            if t:
                out.append(t)
    return '\n\n'.join(out)

def extract_tables(soup: BeautifulSoup):
    main = soup.find('div', class_='mw-parser-output')
    if not main:
        return []
    tbls = []
    for tb in main.find_all('table'):
        cls = ' '.join(tb.get('class', [])).lower()
        if 'wikitable' in cls or 'sortable' in cls:
            hdr = [th.get_text(" ", strip=True) for th in tb.find_all('th')]
            rows = [[c.get_text(" ", strip=True)
                     for c in tr.find_all(['td', 'th'])]
                    for tr in tb.find_all('tr')[1:]]
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
            soup = BeautifulSoup(html_path.read_text(encoding='utf-8'),
                                 'html.parser')

            # ---------- TEXT ----------
            text = extract_text(soup)
            f_text.write(json.dumps({
                "id": new_id("text"),
                "title": title,
                "url": url,
                "content": text
            }, ensure_ascii=False) + '\n')

            # ---------- INFOBOX ----------
            ib = extract_infobox(soup)
            if ib:
                f_info.write(json.dumps({
                    "id": new_id("infobox"),
                    "title": title,
                    "url": url,
                    "content": json.dumps(ib, ensure_ascii=False)  # 统一存为字符串
                }, ensure_ascii=False) + '\n')

            # ---------- TABLES ----------
            for tbl in extract_tables(soup):
                f_tbl.write(json.dumps({
                    "id": new_id("table"),
                    "title": title,
                    "url": url,
                    "content": json.dumps(tbl, ensure_ascii=False)  # 统一存为字符串
                }, ensure_ascii=False) + '\n')

# ---------- CLI ----------
if __name__ == "__main__":
    html_dir = "wikipedia_expanded_html"
    process_html_folder(
        html_dir,
        t_out  = "numsports/corpus/text/expanded_corpus_text.jsonl",
        i_out  = "numsports/corpus/infobox/expanded_corpus_infobox.jsonl",
        tbl_out= "numsports/corpus/table/expanded_corpus_tables.jsonl"
    )
