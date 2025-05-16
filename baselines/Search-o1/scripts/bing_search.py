#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal HTML → Text / Table / Infobox Extractor
=================================================
• Wikipedia: 专用规则
• 其他站点: readability‑lxml 抽正文 + pandas 解析 <table>
• snippet 匹配 + ±2 500 字符上下文
• 输出统一字符串，最长 8 000 字
"""

import os, re, time, json, string, concurrent
from typing import Optional, Tuple, Dict, List
from urllib.parse import urlparse
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import pdfplumber
import pandas as pd
from readability import Document
from nltk.tokenize import sent_tokenize  # ensure punkt model is downloaded

# ----------------------- HTTP Session ------------------------
HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/122.0.0.0 Safari/537.36'),
    'Referer': 'https://www.google.com/'
}

session = requests.Session()
session.headers.update(HEADERS)

# -------------------- Common Helpers -------------------------
def _strip(soup: BeautifulSoup):
    """删除引用脚注、脚本等噪声节点"""
    for t in soup.find_all(['sup', 'style', 'script', 'span'],
                           class_=['reference', 'sortkey']):
        t.decompose()

def remove_punct(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    inter = len(true_set & pred_set)
    if not inter:
        return 0.0
    prec = inter / len(pred_set)
    rec  = inter / len(true_set)
    return 2 * prec * rec / (prec + rec)

def is_wikipedia(url: str) -> bool:
    try:
        host = urlparse(url).netloc
        return host.endswith("wikipedia.org") and "/wiki/" in urlparse(url).path
    except Exception:
        return False

# -------------------- Wikipedia Parsing ----------------------
def parse_wiki_live(url: str, retry: int = 3, timeout: int = 10
                   ) -> Tuple[str, Dict, List[Dict]]:
    """
    返回 (正文, infobox:dict, tables:list[dict])
    """
    for att in range(1, retry + 1):
        try:
            r = session.get(url, timeout=timeout)
            if not r.ok or "<html" not in r.text.lower():
                raise RuntimeError(f"status {r.status_code}")
            soup = BeautifulSoup(r.text, "html.parser")
            _strip(soup)

            # -------- 正文 --------
            cont = soup.find("div", class_="mw-parser-output")
            stops = {"references", "external links", "see also",
                     "further reading", "notes"}
            paras = []
            for el in cont.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol', 'li'], recursive=True):
                txt = el.get_text(" ", strip=True)
                if el.name in {'h2', 'h3', 'h4'} and txt.lower() in stops:
                    break
                if txt:
                    paras.append(txt)
            raw_text = "\n\n".join(paras)

            # -------- Infobox -------
            infobox = {}
            for tb in soup.find_all("table"):
                if "infobox" in " ".join(tb.get("class", [])).lower():
                    _strip(tb)
                    cur = None
                    for tr in tb.find_all("tr"):
                        th, td = tr.find("th"), tr.find("td")
                        if th and not td:
                            cur = th.get_text(" ", strip=True)
                            infobox[cur] = {}
                        elif th and td:
                            k = th.get_text(" ", strip=True)
                            v = " | ".join(td.stripped_strings)
                            infobox.setdefault(cur or k, {})[k] = v
                    break

            # -------- Tables --------
            tables = []
            for tb in soup.find_all("table"):
                cls = " ".join(tb.get("class", [])).lower()
                if "wikitable" in cls or "sortable" in cls:
                    hdr = [th.get_text(" ", strip=True) for th in tb.find_all("th")]
                    rows = [[c.get_text(" ", strip=True)
                             for c in tr.find_all(['td', 'th'])]
                            for tr in tb.find_all("tr")[1:]]
                    tables.append({"columns": hdr, "rows": rows})
            return raw_text, infobox, tables
        except Exception as e:
            if att == retry:
                raise
            time.sleep(0.6 * 2 ** (att - 1))

# ------------------ Generic HTML Parsing ---------------------
INFO_KW = {"infobox", "sidebar", "summary", "card"}

def extract_main_text(html: str) -> str:
    """readability 抽正文，失败则退化到全页 text"""
    try:
        doc = Document(html)
        soup = BeautifulSoup(doc.summary(), "html.parser")
        return soup.get_text(" ", strip=True)
    except Exception:
        return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)

def extract_blocks(soup: BeautifulSoup) -> Tuple[Dict, List[Dict]]:
    """侧边栏 / 信息卡 & 所有表格 → JSON"""
    # ---- Infobox / Sidebar ----
    infobox = {}
    for box in soup.find_all(["table", "aside", "div"]):
        cls = " ".join(box.get("class", [])).lower()
        if any(k in cls for k in INFO_KW):
            _strip(box)
            cur = None
            for tr in box.find_all("tr"):
                th, td = tr.find("th"), tr.find("td")
                if th and not td:
                    cur = th.get_text(" ", strip=True)
                    infobox[cur] = {}
                elif th and td:
                    k = th.get_text(" ", strip=True)
                    v = " | ".join(td.stripped_strings)
                    infobox.setdefault(cur or k, {})[k] = v
            if infobox:
                break  # 取第一块即可

    # ---- Tables ----
    tables = []
    for tb in soup.find_all("table"):
        try:
            df = pd.read_html(str(tb), header=0)[0]  # 只取第一张
            tables.append({
                "columns": list(df.columns.astype(str)),
                "rows": df.astype(str).values.tolist()
            })
        except ValueError:
            continue
    return infobox, tables

# ------------------ Snippet 相关 ------------------------------
def extract_snippet_with_context(full_text: str, snippet: str,
                                 context_chars: int = 2500) -> Tuple[bool, str]:
    """句级 F1 匹配 + 上下文裁剪"""
    try:
        full_text = full_text[:50000]
        snippet_words = set(remove_punct(snippet.lower()).split())
        best_sent, best_f1 = None, 0.2
        for sent in sent_tokenize(full_text):
            s_words = set(remove_punct(sent.lower()).split())
            f1 = f1_score(snippet_words, s_words)
            if f1 > best_f1:
                best_f1, best_sent = f1, sent
        if not best_sent:
            return False, full_text[:context_chars * 2]
        p0, p1 = full_text.find(best_sent), full_text.find(best_sent) + len(best_sent)
        return True, full_text[max(0, p0 - context_chars):min(len(full_text), p1 + context_chars)]
    except Exception as e:
        return False, f"Failed to extract context: {e}"

def assemble_payload(text: str, infobox: Dict, tables: List[Dict],
                     snippet: Optional[str]) -> str:
    raw = text
    if snippet:
        hit, ctx = extract_snippet_with_context(text, snippet)
        raw = ctx if hit else text
    blocks = []
    if infobox:
        blocks.append("[INFOBOX]\n" + json.dumps(infobox, ensure_ascii=False))
    for i, tb in enumerate(tables, 1):
        blocks.append(f"[TABLE {i}]\n" + json.dumps(tb, ensure_ascii=False))
    return (raw + "\n\n" + "\n\n".join(blocks))[:8000]

# ------------------ PDF -------------------------------------------------
def extract_pdf_text(url: str) -> str:
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        with pdfplumber.open(BytesIO(r.content)) as pdf:
            txt = " ".join(p.extract_text() or "" for p in pdf.pages)
        return ' '.join(txt.split()[:600])
    except Exception as e:
        return f"Error parsing PDF: {e}"

# ------------------ 主入口 ----------------------------------------------
def extract_text_from_url(url: str, *, use_jina: bool = False,
                          jina_api_key: str | None = None,
                          snippet: Optional[str] = None) -> str:
    """
    返回统一字符串（正文/上下文 + [INFOBOX] + [TABLE]).
    """
    try:
        # ---------- 1. Jina Reader ----------
        if use_jina:
            headers = {
                'Authorization': f'Bearer {jina_api_key}' if jina_api_key else '',
                'X-Return-Format': 'markdown'
            }
            res = requests.get(f'https://r.jina.ai/{url}', headers=headers, timeout=20).text
            text = re.sub(r"\(https?:.*?\)|\[https?:.*?\]", "", res).replace('---', '-')
            return text[:8000]

        # ---------- 2. Wikipedia ----------
        if is_wikipedia(url):
            raw, info, tables = parse_wiki_live(url)
            return assemble_payload(raw, info, tables, snippet)

        # ---------- 3. 其他网页 ----------
        r = session.get(url, timeout=20, allow_redirects=True)
        r.raise_for_status()
        ctype = r.headers.get('Content-Type', '')
        if 'pdf' in ctype or url.lower().endswith('.pdf'):
            return extract_pdf_text(url)

        html = r.text
        main_text = extract_main_text(html)
        soup = BeautifulSoup(html, "html.parser")
        info, tables = extract_blocks(soup)
        return assemble_payload(main_text, info, tables, snippet)

    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except requests.exceptions.ConnectionError:
        return "Error: Connection problem"
    except Exception as e:
        return f"Unexpected error: {e}"

# ------------------ 并发抓取 -------------------------------------------
def fetch_page_content(urls: List[str], *, max_workers: int = 32,
                       use_jina: bool = False, jina_api_key: str | None = None,
                       snippets: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut2url = {pool.submit(extract_text_from_url, u,
                               use_jina=use_jina,
                               jina_api_key=jina_api_key,
                               snippet=snippets.get(u) if snippets else None): u
                   for u in urls}
        for fut in tqdm(concurrent.futures.as_completed(fut2url),
                        total=len(urls), desc="Fetching URLs"):
            url = fut2url[fut]
            try:
                results[url] = fut.result()
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            time.sleep(0.2)   # simple rate‑limit
    return results

# ------------------ Bing Search (SerpAPI) -------------------------------
def bing_web_search(query: str, subscription_key: str, endpoint: str,
                    market: str = 'en-US', language: str = 'en', timeout: int = 20
                   ) -> Dict:
    from serpapi.google_search import GoogleSearch
    params = {"engine": "bing", "q": query, "cc": market.split('-')[-1],
              "hl": language, "api_key": subscription_key}
    try:
        return GoogleSearch(params).get_dict()
    except Timeout:
        print(f"Bing search timed out for {query}")
        return {}
    except Exception as e:
        print(f"Bing search error: {e}")
        return {}

def extract_relevant_info(search_json: Dict) -> List[Dict]:
    info_list = []
    def _add(res: dict, idx: int, src: str):
        info_list.append({
            "id": idx, "title": res.get('name') or res.get('title', ''),
            "url": res.get('url') or res.get('link', ''),
            "site_name": res.get('siteName', ''),
            "date": res.get('datePublished', res.get('date', '')).split('T')[0],
            "snippet": res.get('snippet', ''),
            "context": ""
        })
    if 'webPages' in search_json:
        for i, v in enumerate(search_json['webPages'].get('value', []), 1):
            _add(v, i, 'webPages')
    elif 'organic_results' in search_json:
        for i, v in enumerate(search_json.get('organic_results', []), 1):
            _add(v, i, 'organic_results')
    return info_list

# ------------------ CLI 测试 -------------------------------------------
if __name__ == "__main__":
    QUERY = "Structure of dimethyl fumarate"
    SERP_KEY = os.getenv("BING_SEARCH_KEY") or "YOUR_SERPAPI_KEY"
    if SERP_KEY.startswith("YOUR"):
        raise ValueError("Please set environment variable BING_SEARCH_KEY")

    res = bing_web_search(QUERY, SERP_KEY, "")
    items = extract_relevant_info(res)

    for item in tqdm(items, desc="Enrich context"):
        txt = extract_text_from_url(item['url'], snippet=item['snippet'])
        hit, ctx = extract_snippet_with_context(txt, item['snippet'])
        item['context'] = ctx if hit else txt[:8000]

    print(json.dumps(items, indent=2, ensure_ascii=False))
