#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia → corpus  pipeline  (Py 3.11 safe, FAISS block-add)
------------------------------------------------------------
• TEXT  → ≤100-token sentence chunks
• TABLE / INFOBOX 原样
• substring / RapidFuzz 先匹配，BGE-M3 + FAISS 补漏
"""

from __future__ import annotations
import json, re, time, random, hashlib, logging, requests, numpy as np, faiss, os, sys, torch
from pathlib import Path
from urllib.parse import urlparse, unquote
from collections.abc import Mapping, Iterable
from bs4 import BeautifulSoup
from tqdm import tqdm
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import spacy

# ------------------- Config -------------------
CFG = dict(
    in_dataset   = "numsports/dataset/reformatted_merged_dataset.jsonl",
    html_dir     = "numsports/wikipedia_html",
    out_dir      = "numsports/corpus/final",
    model_name   = "BAAI/bge-m3",
    sim_thr      = 0.60,
    quick_ratio_thr = 85,
    max_tok_chunk   = 100,
    overlap_tok     = 0,
    batch_embed     = 512,
    record_range    = None,          # (start, end) or None
    log_file        = "pipeline_chunk_text_only.log",
    retry_html      = 3,
    backoff_base    = 0.6,
    faiss_block     = 10_000
)

os.environ.setdefault(
    "TRANSFORMERS_CACHE", str(Path.home() / ".cache" / "huggingface")
)

# ---------------- Logging --------------------
logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(CFG["log_file"])
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(fh)
logger.addHandler(logging.StreamHandler(sys.stdout))

failed_pages: set[str] = set()
failed_evidences: list[dict] = []

# ---------------- Utils ----------------------
def jsonl_iter(path: str):
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                try:
                    yield json.loads(ln)
                except Exception as e:
                    logger.warning(f"malformed line skipped: {e}")

def safe_write(lst: list[dict], path: Path):
    tmp = Path(f"{path}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for obj in lst:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(path)

def flatten(obj: str | Mapping | Iterable) -> str:
    if isinstance(obj, Mapping):
        return " ".join(f"{k} {flatten(v)}" for k, v in obj.items())
    if isinstance(obj, Iterable) and not isinstance(obj, str):
        return " ".join(flatten(x) for x in obj)
    return str(obj)

ensure_str = lambda x: x if isinstance(x, str) else flatten(x)
slug  = lambda u: unquote(urlparse(u).path.split("/wiki/")[-1]) if "/wiki/" in u else ""
clean = lambda s: re.sub(r"\s+", " ", s).strip()

# --------------- NLP helpers -----------------
nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")
tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

def sent_split(t: str): return [s.text.strip() for s in nlp(t).sents if s.text.strip()]
def token_len(t: str):   return len(tokenizer(t, add_special_tokens=False)["input_ids"])

def chunk_text(text: str) -> list[str]:
    sents = sent_split(text)
    if not sents:
        return []
    max_tok = CFG["max_tok_chunk"]
    overlap = min(CFG["overlap_tok"], max_tok - 1)
    chunks, cur, tok = [], [], 0
    for s in sents:
        l = token_len(s)
        if cur and tok + l > max_tok:
            chunks.append(" ".join(cur))
            if overlap:
                tail_ids = tokenizer(" ".join(cur), add_special_tokens=False)["input_ids"][-overlap:]
                ov = tokenizer.decode(tail_ids)
                cur, tok = [ov], token_len(ov)
            else:
                cur, tok = [], 0
        cur.append(s)
        tok += l
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# -------------- HTML fetch ------------------
def canonical_key(title: str, url: str) -> str:
    url = re.sub(r"^https?://[^/]+", "https://en.wikipedia.org", url).split("#")[0].lower()
    return slug(url) or re.sub(r"[^\w\-]", "_", title.lower())

def ensure_html(title: str, url: str) -> Path | None:
    fp = Path(CFG["html_dir"]) / f"{canonical_key(title, url)}.html"
    if fp.exists():
        return fp
    for att in range(1, CFG["retry_html"] + 1):
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if r.ok and "<html" in r.text.lower():
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(r.text, encoding="utf-8")
                return fp
            logger.warning(f"[{att}] status {r.status_code}: {url}")
        except Exception as e:
            logger.warning(f"[{att}] download error {e}")
        time.sleep(CFG["backoff_base"] * (2 ** (att - 1)) + random.random() * 0.2)
    failed_pages.add(url)
    return None

# --------- Wiki parsing ---------------------
def _strip(soup):
    for t in soup.find_all(['sup', 'style', 'script', 'span'], class_=['reference', 'sortkey']):
        t.decompose()

def parse_wiki(title: str, url: str):
    fp = ensure_html(title, url)
    if not fp:
        return "", "", []
    try:
        soup = BeautifulSoup(fp.read_text(encoding="utf-8"), "html.parser")
    except Exception as e:
        logger.warning(f"parse error {fp}: {e}")
        return "", "", []
    _strip(soup)
    cont = soup.find("div", class_="mw-parser-output")
    lines, stop = [], {"references", "external links", "see also", "further reading", "notes"}
    for el in cont.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol', 'li'], recursive=True):
        txt = el.get_text(" ", strip=True)
        if el.name in {'h2', 'h3', 'h4'} and txt.lower() in stop:
            break
        lines.append(txt)
    raw = "\n\n".join([l for l in lines if l])

    # ---------- infobox ----------
    info = {}
    for tb in soup.find_all("table"):
        if "infobox" in " ".join(tb.get("class", [])).lower():
            _strip(tb)
            cur = None
            for tr in tb.find_all("tr"):
                th, td = tr.find("th"), tr.find("td")
                if th and not td:
                    cur = th.get_text(" ", strip=True)
                    info[cur] = {}
                elif th and td:
                    k = th.get_text(" ", strip=True)
                    v = " | ".join(td.stripped_strings)
                    info.setdefault(cur or k, {})[k] = v
            break

    # ---------- wikitable ----------
    tables = []
    for tb in soup.find_all("table"):
        cls = " ".join(tb.get("class", [])).lower()
        if "wikitable" in cls or "sortable" in cls:
            hdr = [th.get_text(" ", strip=True) for th in tb.find_all("th")]
            rows = [[c.get_text(" ", strip=True) for c in tr.find_all(['td', 'th'])]
                    for tr in tb.find_all("tr")[1:]]
            tables.append({"columns": hdr, "rows": rows})

    return raw, info, tables

# --------------- Embedder -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
class Embedder:
    def __init__(self, mdl):
        self.model = SentenceTransformer(mdl, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return self.model.encode(
            [t[:8192] for t in texts],
            normalize_embeddings=True,
            show_progress_bar=False
        )

# ------------ Helper ------------------------
valid_url = lambda u: u.startswith("http://") or u.startswith("https://")
id2url    = lambda eid: "https://en.wikipedia.org" + eid if eid.startswith("/wiki/") else ""

# --------------- Main -----------------------
def main():
    dataset = list(jsonl_iter(CFG["in_dataset"]))

    text_docs:   list[dict]        = []
    table_docs:  list[dict]        = []
    info_docs:   dict[str, dict]   = {}   # id → doc
    id2content:  dict[str, str]    = {}   # chunk id → text

    text_map, table_map, info_map = {}, {}, {}  # title → ids / hash
    unmatched: list[tuple[dict, dict]] = []

    id_counter = {'text': 0, 'table': 0, 'infobox': 0}
    gen_id = lambda k: f"{k}_{id_counter.__setitem__(k, id_counter[k] + 1) or id_counter[k] - 1:06d}"

    # ---------- Stage-1 : 快速匹配 ----------
    for si, sample in enumerate(tqdm(dataset, desc="Stage1")):
        rng = CFG["record_range"]
        if rng and not (rng[0] <= si <= rng[1]):
            if rng and si > rng[1]:
                break
            continue

        for ev in sample.get("gold_evidences", []):
            kind  = ev.get("type", "infobox").lower()
            title = ev.get("title") or slug(ev.get("url", "")).replace("_", " ")
            url_raw = ev.get("url", "")
            url = url_raw if valid_url(url_raw) else id2url(ev.get("id", ""))
            ev["title"], ev["url"] = title, url
            try:
                # ---------- TEXT ----------
                if kind == "text":
                    if title not in text_map and valid_url(url):
                        raw, _, _ = parse_wiki(title, url)
                        chunks = chunk_text(raw) or [ensure_str(ev["content"])]
                        ids = []
                        for i, ch in enumerate(chunks, 1):
                            cid = f"{gen_id('text')}_chunk_{i:02d}"
                            text_docs.append(
                                {'id': cid, 'title': title, 'url': url, 'content': ch}
                            )
                            id2content[cid] = ch
                            ids.append(cid)
                        text_map[title] = ids

                    hit = False
                    for cid in text_map.get(title, []):
                        txt  = id2content.get(cid, "")
                        left = ensure_str(ev["content"])
                        if left and (left in txt or
                                     fuzz.token_set_ratio(left, txt) >= CFG["quick_ratio_thr"]):
                            ev["id"] = cid
                            hit = True
                            break
                    if not hit:
                        unmatched.append((sample, ev))

                # ---------- TABLE ----------
                elif kind == "table":
                    if title not in table_map and valid_url(url):
                        _, _, tbls = parse_wiki(title, url)
                        for tbl in tbls:
                            h = hashlib.sha1(
                                json.dumps(
                                    {'columns': sorted(tbl['columns']), 'rows': tbl['rows']},
                                    sort_keys=True
                                ).encode()
                            ).hexdigest()
                            tid = gen_id('table')
                            table_docs.append(
                                {'id': tid, 'title': title, 'url': url,
                                 'content': json.dumps(tbl, ensure_ascii=False)}
                            )
                            table_map.setdefault(title, {})[h] = tid

                    tbl_ev = ev["content"] if isinstance(ev["content"], dict) else \
                             json.loads(ev["content"])
                    h_ev = hashlib.sha1(
                        json.dumps(
                            {'columns': sorted(tbl_ev['columns']), 'rows': tbl_ev['rows']},
                            sort_keys=True
                        ).encode()
                    ).hexdigest()
                    if h_ev in table_map.get(title, {}):
                        ev["id"] = table_map[title][h_ev]
                    else:
                        unmatched.append((sample, ev))

                # ---------- INFOBOX ----------
                else:
                    if title not in info_map and valid_url(url):
                        _, info, _ = parse_wiki(title, url)
                        iid = gen_id('infobox')
                        doc = {
                            'id': iid, 'title': title, 'url': url,
                            'content': json.dumps(info, ensure_ascii=False)
                        }
                        info_docs[iid] = doc
                        info_map[title] = iid

                    tgt = {}
                    if title in info_map:
                        iid = info_map[title]
                        tgt = json.loads(info_docs[iid]['content'])

                    left = ensure_str(ev["content"])
                    if tgt and (left in clean(flatten(tgt)) or
                                fuzz.token_set_ratio(left, clean(flatten(tgt))) >= CFG["quick_ratio_thr"]):
                        ev["id"] = info_map[title]
                    else:
                        unmatched.append((sample, ev))

            except Exception as e:
                logger.warning(f"quick pass error: {e}")
                unmatched.append((sample, ev))
                failed_evidences.append(ev)

    id2content.clear()

    # ---------- Stage-2 : 语义补漏 ----------
    if unmatched:
        emb = Embedder(CFG["model_name"])

        def add_block(docs, index, extr):
            for i in range(0, len(docs), CFG["faiss_block"]):
                vec = emb.encode([clean(flatten(extr(d))) for d in docs[i:i + CFG["faiss_block"]]])
                if vec.size:
                    index.add(vec)

        idx_t  = faiss.IndexFlatIP(emb.dim);  add_block(text_docs,  idx_t,  lambda d: d['content'])
        idx_tb = faiss.IndexFlatIP(emb.dim);  add_block(table_docs, idx_tb, lambda d: json.loads(d['content']))
        idx_i  = faiss.IndexFlatIP(emb.dim);  add_block(list(info_docs.values()), idx_i,
                                                       lambda d: json.loads(d['content']))

        if faiss.get_num_gpus():
            res = faiss.StandardGpuResources()
            idx_t  = faiss.index_cpu_to_gpu(res, 0, idx_t)
            idx_tb = faiss.index_cpu_to_gpu(res, 0, idx_tb)
            idx_i  = faiss.index_cpu_to_gpu(res, 0, idx_i)

        idx_pack = {
            'text':    (idx_t,  text_docs),
            'table':   (idx_tb, table_docs),
            'infobox': (idx_i,  list(info_docs.values()))
        }

        for sample, ev in tqdm(unmatched, desc="Stage2"):
            k = ev.get("type", "infobox").lower()
            ix, corp = idx_pack[k]
            try:
                qv = emb.encode([clean(flatten(ev["content"]))])
                if not qv.size:
                    raise ValueError("empty vec")
                D, I = ix.search(qv, 1)
                if D[0, 0] >= CFG["sim_thr"]:
                    ev["id"] = corp[I[0, 0]]["id"]
                else:
                    nid = gen_id(k)
                    new_doc = {
                        'id': nid, 'title': ev['title'], 'url': ev['url'],
                        'content': ev['content'], 'added_by': 'bge'
                    }
                    corp.append(new_doc)
                    ix.add(qv)
                    ev["id"] = nid
                    if k == "infobox":
                        info_docs[nid] = new_doc
            except Exception as e:
                logger.warning(f"faiss fallback {e}")
                nid = gen_id(k)
                new_doc = {
                    'id': nid, 'title': ev['title'], 'url': ev['url'],
                    'content': ev['content'], 'added_by': 'fallback'
                }
                corp.append(new_doc)
                ev["id"] = nid
                if k == "infobox":
                    info_docs[nid] = new_doc

    # ---------- save ------------------------
    for samp in dataset:
        samp["gold_evidence_ids"] = [
            ev["id"] for ev in samp.get("gold_evidences", []) if ev.get("id")
        ]

    out = Path(CFG["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    safe_write(text_docs,               out / "corpus_text_final.jsonl")
    safe_write(table_docs,              out / "corpus_tables_final.jsonl")
    safe_write(list(info_docs.values()), out / "corpus_infobox_final.jsonl")
    safe_write(dataset,                 out / "merged_dataset_final.jsonl")

    if failed_pages:
        Path(out / "failed_pages.txt").write_text("\n".join(sorted(failed_pages)))
    if failed_evidences:
        safe_write(failed_evidences, out / "failed_evidences.jsonl")

    logger.info(
        f"Done ✔ text={len(text_docs)} table={len(table_docs)} info={len(info_docs)}"
    )

if __name__ == "__main__":
    main()
