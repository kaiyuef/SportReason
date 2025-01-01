import json
from pyserini.search.lucene import LuceneSearcher

def retrieve_bm25(query, index_path, top_k=5):
    """
    BM25 Retriever 实现
    """
    searcher = LuceneSearcher(index_path)
    hits = searcher.search(query, k=top_k)
    results = []

    for hit in hits:
        doc = searcher.doc(hit.docid)
        if doc and doc.raw():
            raw_content = json.loads(doc.raw())
            results.append({
                "id": raw_content.get("id", hit.docid),
                "title": raw_content.get("title", "Untitled Document"),
                "content": raw_content.get("contents", "No content available."),
                "score": hit.score
            })
    return results

def retrieve_contriever(query, index_path, top_k=5):
    """
    Contriever Retriever 实现（这里使用占位代码）
    """
    # 需要集成实际的 Contriever 检索逻辑
    results = [
        {
            "id": "dummy-doc-1",
            "title": "Contriever Result 1",
            "content": "This is a dummy result from Contriever.",
            "score": 0.9
        },
        {
            "id": "dummy-doc-2",
            "title": "Contriever Result 2",
            "content": "This is another dummy result from Contriever.",
            "score": 0.8
        }
    ]
    return results
