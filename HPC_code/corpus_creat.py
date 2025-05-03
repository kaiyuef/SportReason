#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build corpus (text / infobox / table) from merged_dataset.jsonl
Auto‑download missing Wiki HTML, parse with broader rules.

依赖:
    pip install beautifulsoup4 tqdm requests
"""

import ast, json, re, os, time, hashlib, requests
from pathlib import Path
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import Dict, List

# ---------- 通用清洗 ----------
def sanitize_filename(name:str)->str:
    return re.sub(r'[<>:"/\\|?*\n\t]', '_', name).strip()
def sanitize_str(s): return re.sub(r'\s+', ' ', s).strip() if isinstance(s,str) else s
def sanitize_infobox(d:dict)->dict:
    return {k:(sanitize_infobox(v) if isinstance(v,dict) else sanitize_str(v))
            for k,v in d.items()} if isinstance(d,dict) else d
def sanitize_table(tbl:dict)->dict:
    c=tbl["content"]
    c["columns"]=[sanitize_str(x) for x in c.get("columns",[])]
    c["rows"]=[[sanitize_str(x) for x in row] for row in c.get("rows",[])]
    return tbl
def slug_from_url(url:str)->str:
    try: return unquote(urlparse(url).path.split("/wiki/")[-1])
    except: return ""

# ---------- JSONL 读取 ----------
def jsonl_reader(path:str):
    def balanced(buf:str)->bool:
        lvl,in_str,esc=0,False,False
        for ch in buf:
            if esc: esc=False; continue
            if ch=='\\': esc=True; continue
            if ch=='"': in_str=not in_str
            if in_str:  continue
            lvl += (ch=='{')-(ch=='}')
        return lvl==0
    with open(path,'r',encoding='utf-8') as fp:
        buf=""
        for line in fp:
            if not line.strip(): continue
            buf+=line
            if balanced(buf):
                try: yield json.loads(buf)
                except json.JSONDecodeError: yield ast.literal_eval(buf)
                buf=""

# ---------- HTML 解析 ----------
def _strip_sup_and_scripts(soup:BeautifulSoup):          ### 改进
    for tag in soup.find_all(['sup','style','script','span'], class_=['reference','sortkey']):
        tag.decompose()

def extract_infobox(soup:BeautifulSoup)->Dict:           ### 改进
    for table in soup.find_all('table'):
        cls = ' '.join(table.get('class',[])).lower()
        if 'infobox' in cls:
            _strip_sup_and_scripts(table)
            info,cur={},None
            for tr in table.find_all('tr'):
                th,td=tr.find('th'),tr.find('td')
                if th and not td:
                    cur=th.get_text(" ",strip=True)
                    info[cur]={}
                elif th and td:
                    key = th.get_text(" ",strip=True)
                    val = ' | '.join(td.stripped_strings)
                    info.setdefault(cur or key,{})[key]=val
            return info
    return {}

def extract_text(soup:BeautifulSoup)->str:               ### 改进
    _strip_sup_and_scripts(soup)
    cont=soup.find('div',class_='mw-parser-output')
    if not cont: return ""
    stop_sections={"references","external links","see also","further reading","notes"}
    lines=[]
    for el in cont.find_all(['h2','h3','h4','p','ul','ol','li'],recursive=True):
        if el.name in {'h2','h3','h4'}:
            sec = el.get_text(" ",strip=True).lower()
            if sec.lower() in stop_sections: break
            lines.append(f"## {el.get_text(' ',strip=True)}")
        else:
            txt = el.get_text(" ",strip=True)
            if txt: lines.append(txt)
    return '\n\n'.join(lines)

def extract_tables(soup:BeautifulSoup)->List[Dict]:      ### 改进
    _strip_sup_and_scripts(soup)
    tables=[]
    cont=soup.find('div',class_='mw-parser-output')
    if not cont: return tables
    for tb in cont.find_all('table'):
        cls=' '.join(tb.get('class',[])).lower()
        if 'wikitable' in cls or 'sortable' in cls:
            headers=[th.get_text(" ",strip=True) for th in tb.find_all('th')]
            rows=[[c.get_text(" ",strip=True) for c in tr.find_all(['td','th'])]
                  for tr in tb.find_all('tr')[1:]]
            tables.append({"columns":headers,"rows":rows})
    return tables

# ---------- 下载并缓存 wiki HTML ----------
def ensure_html(title:str,url:str,html_dir:str,dbg:list)->Path|None:
    slug = slug_from_url(url) or sanitize_filename(title.replace(' ','_'))
    fname = sanitize_filename(slug)+".html"
    path  = Path(html_dir)/fname
    if path.exists(): return path
    try:
        r = requests.get(url,timeout=10,headers={"User-Agent":"Mozilla/5.0"})
        if r.ok and "<html" in r.text.lower():
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(r.text,encoding='utf-8'); time.sleep(0.4)
            return path
        dbg.append(("download fail",f"{url} status {r.status_code}"))
    except Exception as e:
        dbg.append(("download err",f"{url} {e}"))
    return None



def extract_text_fallback(soup: BeautifulSoup) -> str:
    """
    最简单的降级方案：直接抓取页面所有 <p> 标签的文本并拼接。
    """
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return "\n\n".join([t for t in paras if t])

def parse_wiki(title: str, url: str, html_dir: str, dbg: list):
    path = ensure_html(title, url, html_dir, dbg)
    if not path:
        return None, None, []

    soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
    std_title = title or slug_from_url(url).replace('_', ' ')

    # —— 正文抽取：标准提取 + 空值降级 —— #
    raw_text = extract_text(soup)
    if not raw_text.strip():
        raw_text = extract_text_fallback(soup)
    text_d = {
        "title":   std_title,
        "url":     url,
        "content": sanitize_str(raw_text)
    }

    # —— Infobox 抽取（保持原逻辑） —— #
    info_d = {
        "title":   std_title,
        "url":     url,
        "content": sanitize_infobox(extract_infobox(soup))
    }

    # —— 表格抽取（保持原逻辑） —— #
    tables = [
        sanitize_table({"title": std_title, "url": url, "content": tbl})
        for tbl in extract_tables(soup)
    ]

    return text_d, info_d, tables


# ---------- 主流程 ----------
def build_corpora(in_jsonl:str,html_dir:str,
                  out_text:str,out_info:str,out_table:str):

    Path(out_text).parent.mkdir(parents=True,exist_ok=True)
    texts,infos,tables=[],[],[]
    seen_text,seen_info,seen_table={}, {}, {}
    cnt_text=cnt_info=cnt_table=0
    cache={}
    dbg=[]

    for sample in tqdm(jsonl_reader(in_jsonl),desc="building"):
        evs = sample.get("gold_evidences",[]) + sample.get("used_evidences",[])
        for ev in evs:
            if not isinstance(ev,dict): continue
            # url / title 补齐
            if not ev.get("url") and isinstance(ev.get("id"),str) and ev["id"].startswith("/wiki/"):
                ev["url"]="https://en.wikipedia.org"+ev["id"]
            if not ev.get("title") and ev.get("url"):
                slug=slug_from_url(ev["url"]); 
                if slug: ev["title"]=slug.replace('_',' ')
            title,url = ev.get("title",""), ev.get("url","")
            if not url: continue

            ev_type = ev.get("type","infobox").lower()
            if title not in cache:
                cache[title]=parse_wiki(title,url,html_dir,dbg)
            text_d,info_d,table_ls = cache[title]
            if text_d is None: continue
            key=(title,url)
            if ev_type=="text" and key not in seen_text:
                eid=f"text_{cnt_text:06d}"
                texts.append(dict(text_d,id=eid)); seen_text[key]=eid; cnt_text+=1
            elif ev_type=="table":
                for tbl in table_ls:
                    h=hashlib.sha1(json.dumps(tbl["content"],sort_keys=True).encode()).hexdigest()
                    k=(title,url,h)
                    if k not in seen_table:
                        eid=f"table_{cnt_table:06d}"
                        tbl["id"]=eid; tables.append(tbl); seen_table[k]=eid; cnt_table+=1
            else:
                if key not in seen_info:
                    eid=f"infobox_{cnt_info:06d}"
                    infos.append(dict(info_d,id=eid)); seen_info[key]=eid; cnt_info+=1

    def dump(p:str,data:List[dict]):
        with open(p,'w',encoding='utf-8') as f:
            for x in data: f.write(json.dumps(x,ensure_ascii=False)+"\n")
    dump(out_text,texts); dump(out_info,infos); dump(out_table,tables)

    print(f"✅ corpus 完成  text={len(texts)}, infobox={len(infos)}, table={len(tables)}")
    if dbg:
        print("⚠️  download/parse issues (first 5):")
        for tag,msg in dbg[:5]: print(f"  [{tag}] {msg}")

# ---------- CLI ----------
if __name__ == "__main__":
    build_corpora(
        in_jsonl ="regenerate_combine/reformatted_merged_dataset.jsonl",
        html_dir ="wikipedia_html",
        out_text ="regenerate_combine/regenerated_corpus/corpus_text.jsonl",
        out_info ="regenerate_combine/regenerated_corpus/corpus_infobox.jsonl",
        out_table="regenerate_combine/regenerated_corpus/corpus_tables.jsonl"
    )
