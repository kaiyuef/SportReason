#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-stop pipeline (text-chunk only):
 1) Fetch & parse Wikipedia → sentence-level chunking for TEXT only
 2) Quick existence check (rapidfuzz/hash)
 3) BGE-M3 + FAISS semantic fallback
 4) Sync gold_evidence_ids → corpus IDs

Table & Infobox are NOT chunked; saved in original JSON format.

Dependencies:
  pip install beautifulsoup4 rapidfuzz tqdm requests spacy transformers sentence-transformers faiss-cpu
  python -m spacy download en_core_web_sm
"""
import json
import re
import time
import hashlib
import itertools
import logging
import requests
import numpy as np
import faiss
from pathlib import Path
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup
from tqdm import tqdm
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import spacy

# ------------------ Configuration ------------------
CFG = dict(
    in_dataset      = "regenerate_combine/reformatted_merged_dataset.jsonl",
    html_dir        = "wikipedia_html",
    out_dir         = "regenerate_combine/final",
    model_name      = "BAAI/bge-m3",
    sim_thr         = 0.60,
    quick_ratio_thr = 85,
    max_tok_chunk   = 100,
    overlap_tok     = 0,
    batch_embed     = 512,
    record_range    = None,
    log_file        = "pipeline_chunk_text_only.log",
)

# Setup logging
tlogging = logging.getLogger()
tlogging.setLevel(logging.INFO)
thandler = logging.FileHandler(CFG['log_file'])
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
thandler.setFormatter(formatter)
tlogging.addHandler(thandler)

# ---------- Utilities ----------
def jsonl_iter(path):
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)

def save_jsonl(lst, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in lst:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def sanitize_fn(s):
    return re.sub(r'[<>:"/\\|?*\n\t]', '_', s).strip()

def slug(url):
    try:
        return unquote(urlparse(url).path.split("/wiki/")[-1])
    except:
        return ""

def flatten(o):
    if isinstance(o, dict):
        return " ".join(itertools.chain.from_iterable((str(k), flatten(v)) for k, v in o.items()))
    if isinstance(o, list):
        return " ".join(flatten(x) for x in o)
    return str(o)

def clean(txt):
    return re.sub(r"\s+", " ", txt).strip()

# ---------- Sentence Chunker for TEXT ----------
nlp = spacy.load("en_core_web_sm", disable=["ner","tagger","parser"])
nlp.add_pipe("sentencizer")
tokenizer = AutoTokenizer.from_pretrained(CFG['model_name'])

def sent_split(text):
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]

def token_len(txt):
    return len(tokenizer(txt, add_special_tokens=False)["input_ids"])

def chunk_text(text, max_tok, overlap):
    sents = sent_split(text)
    chunks, cur, tok_count = [], [], 0
    for sent in sents:
        l = token_len(sent)
        if not l:
            continue
        if cur and tok_count + l > max_tok:
            chunks.append(" ".join(cur))
            if overlap > 0:
                last_ids = tokenizer(" ".join(cur), add_special_tokens=False)["input_ids"][-overlap:]
                ov_text = tokenizer.decode(last_ids)
                cur, tok_count = [ov_text], token_len(ov_text)
            else:
                cur, tok_count = [], 0
        cur.append(sent)
        tok_count += l
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# ---------- Wikipedia Fetch & Parse ----------
def ensure_html(title, url):
    slug_name = slug(url) or sanitize_fn(title.replace(' ', '_'))
    path = Path(CFG['html_dir']) / f"{slug_name}.html"
    if path.exists():
        return path
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
        if r.ok and '<html' in r.text.lower():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text, encoding='utf-8')
            time.sleep(0.3)
            return path
    except Exception as e:
        logging.warning(f"download error {url}: {e}")
    return None

def _strip_misc(soup):
    for tag in soup.find_all(['sup','style','script','span'], class_=['reference','sortkey']):
        tag.decompose()

def parse_wiki(title, url):
    p = ensure_html(title, url)
    if not p:
        return "", {}, []
    soup = BeautifulSoup(p.read_text(encoding='utf-8'), 'html.parser')
    _strip_misc(soup)
    cont = soup.find('div', class_='mw-parser-output')
    # text
    lines = []
    stop = {"references","external links","see also","further reading","notes"}
    for el in cont.find_all(['h2','h3','h4','p','ul','ol','li'], recursive=True):
        if el.name in {'h2','h3','h4'} and el.get_text(" ", strip=True).lower() in stop:
            break
        lines.append(el.get_text(" ", strip=True))
    raw_text = "\n\n".join([l for l in lines if l])
    # infobox
    info = {}
    for tb in soup.find_all('table'):
        if 'infobox' in ' '.join(tb.get('class', [])).lower():
            _strip_misc(tb)
            cur = None
            for tr in tb.find_all('tr'):
                th, td = tr.find('th'), tr.find('td')
                if th and not td:
                    cur = th.get_text(" ", strip=True)
                    info[cur] = {}
                elif th and td:
                    k = th.get_text(" ", strip=True)
                    v = ' | '.join(td.stripped_strings)
                    info.setdefault(cur or k, {})[k] = v
            break
    # tables
    tables = []
    for tb in soup.find_all('table'):
        cls = ' '.join(tb.get('class', [])).lower()
        if 'wikitable' in cls or 'sortable' in cls:
            hdr = [th.get_text(" ", strip=True) for th in tb.find_all('th')]
            rows = [[c.get_text(" ", strip=True) for c in tr.find_all(['td','th'])]
                    for tr in tb.find_all('tr')[1:]]
            tables.append({'columns': hdr, 'rows': rows})
    return raw_text, info, tables

# ---------- Quick Match & FAISS Alignment ----------
class Embedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def encode(self, texts, bs):
        outs = []
        for i in range(0, len(texts), bs):
            outs.append(
                self.model.encode(texts[i:i+bs], normalize_embeddings=True, show_progress_bar=False)
            )
        return np.vstack(outs).astype('float32')

# ---------- Main Pipeline ----------
def main():
    # Load dataset
    dataset = list(jsonl_iter(CFG['in_dataset']))

    # Initialize containers
    text_docs, table_docs, info_docs = [], [], []
    text_map, table_map, info_map = {}, {}, {}
    unmatched = []
    id_counter = {'text':0, 'table':0, 'infobox':0}
    def new_id(kind):
        rid = f"{kind}_{id_counter[kind]:06d}"
        id_counter[kind] += 1
        return rid

    # Stage 1: HTML parse + chunk + quick match
    for idx, sample in enumerate(tqdm(dataset, desc="Stage1: Quick Match")):
        if CFG['record_range'] and not (CFG['record_range'][0] <= idx <= CFG['record_range'][1]):
            continue
        for ev in sample.get('gold_evidences', []):
            kind = ev.get('type','infobox').lower()
            title = ev.get('title') or slug(ev.get('url','')).replace('_',' ')
            url   = ev.get('url') or ('https://en.wikipedia.org'+ev.get('id',''))
            ev['title'], ev['url'] = title, url
            if kind == 'text':
                # parse and chunk if first time
                if title not in text_map:
                    raw, _, _ = parse_wiki(title, url)
                    chunks = chunk_text(raw, CFG['max_tok_chunk'], CFG['overlap_tok'])
                    for i, ch in enumerate(chunks,1):
                        cid = new_id('text')+f"_chunk_{i:02d}"
                        text_docs.append({'id':cid,'title':title,'url':url,'content':ch})
                        text_map.setdefault(title,[]).append(cid)
                # quick match
                hit=False
                for cid in text_map.get(title,[]):
                    cont = text_docs[int(cid.split('_')[1])]['content']
                    if cont and (ev['content'] in cont or fuzz.token_set_ratio(ev['content'], cont) >= CFG['quick_ratio_thr']):
                        ev['id']=cid; hit=True; break
                if not hit: unmatched.append((sample,ev))
            elif kind=='table':
                if title not in table_map:
                    _,_,tbls = parse_wiki(title,url)
                    for tbl in tbls:
                        h=hashlib.sha1(json.dumps(tbl,sort_keys=True).encode()).hexdigest()
                        tid=new_id('table')
                        table_docs.append({'id':tid,'title':title,'url':url,'content':json.dumps({'columns':[[c,[]] for c in tbl['columns']],'rows':tbl['rows']},ensure_ascii=False)})
                        table_map.setdefault(title,{})[h]=tid
                h_ev=hashlib.sha1(json.dumps(ev['content'],sort_keys=True).encode()).hexdigest()
                if h_ev in table_map.get(title,{}): ev['id']=table_map[title][h_ev]
                else: unmatched.append((sample,ev))
            else:
                if title not in info_map:
                    _,info,_=parse_wiki(title,url)
                    iid=new_id('infobox')
                    info_docs.append({'id':iid,'title':title,'url':url,'content':json.dumps(info,ensure_ascii=False)})
                    info_map[title]=iid
                tgt=json.loads(info_docs[int(info_map[title].split('_')[1])]['content'])
                flat_tgt=clean(flatten(tgt))
                if ev['content'] in flat_tgt or fuzz.token_set_ratio(ev['content'],flat_tgt)>=CFG['quick_ratio_thr']:
                    ev['id']=info_map[title]
                else: unmatched.append((sample,ev))

    # Stage 2: BGE + FAISS fallback
    if unmatched:
        emb=Embedder(CFG['model_name'])
        vec_text=emb.encode([clean(flatten(d['content'])) for d in text_docs],CFG['batch_embed'])
        vec_tab =emb.encode([clean(flatten(json.loads(d['content']) if isinstance(d['content'],str) else d['content'])) for d in table_docs],CFG['batch_embed'])
        vec_inf =emb.encode([clean(flatten(json.loads(d['content']) if isinstance(d['content'],str) else d['content'])) for d in info_docs],CFG['batch_embed'])
        idx_pack={
            'text':(faiss.IndexFlatIP(vec_text.shape[1]),text_docs,vec_text),
            'table':(faiss.IndexFlatIP(vec_tab.shape[1]), table_docs, vec_tab),
            'infobox':(faiss.IndexFlatIP(vec_inf.shape[1]), info_docs, vec_inf)
        }
        for k in idx_pack:
            idx_pack[k][0].add(idx_pack[k][2])
        for sample,ev in tqdm(unmatched,desc="Stage2: BGE Align"):
            k=ev.get('type','infobox').lower()
            ix,corpus,_=idx_pack[k]
            qv=emb.encode([clean(flatten(ev['content']))],CFG['batch_embed'])
            D,I=ix.search(qv,1)
            if D[0][0]>=CFG['sim_thr']:
                ev['id']=corpus[I[0][0]]['id']
            else:
                nid=new_id(k)
                corpus.append({'id':nid,'title':ev['title'],'url':ev['url'],'content':ev['content'],'added_by':'bge'})
                ix.add(qv); ev['id']=nid

    # Sync gold_evidence_ids
    for sam in dataset:
        sam['gold_evidence_ids']=[ev.get('id') for ev in sam.get('gold_evidences',[]) if ev.get('id')]

    # Save outputs
    out=Path(CFG['out_dir'])
    save_jsonl(text_docs, out/"corpus_text_final.jsonl")
    save_jsonl(table_docs,out/"corpus_tables_final.jsonl")
    save_jsonl(info_docs, out/"corpus_infobox_final.jsonl")
    save_jsonl(dataset,    out/"merged_dataset_final.jsonl")
    print(f"✅ Done! text={len(text_docs)} table={len(table_docs)} info={len(info_docs)}")

if __name__ == '__main__':
    main()
