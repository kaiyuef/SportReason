import json
from pyserini.encode import TctColBertQueryEncoder
from pyserini.search.lucene import LuceneSearcher, LuceneHnswDenseSearcher
from pyserini.search.faiss import FaissSearcher

def retrieve_bm25(query, index_path, top_k=100):
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

def retrieve_contriever(query, index_path, top_k=100, encoder_name="facebook/contriever-msmarco"):
    """
    使用 Pyserini 的 Contriever 检索文档
    :param query: 输入查询文本
    :param index_path: FAISS 索引路径
    :param top_k: 返回的最近邻数量
    :param encoder_name: Contriever 编码器名称，默认使用 'facebook/contriever-msmarco'
    :return: 检索结果列表
    """
    # 初始化 Pyserini 的密集向量检索器
    retriever = FaissSearcher(
        encoder_name=encoder_name,  # 使用 Contriever 编码器
        index_path=index_path      # 指定 FAISS 索引路径
    )

    # 执行检索
    hits = retriever.search(query, top_k)

    # 处理检索结果
    results = []
    for hit in hits:
        # 如果索引中存储了原始文档，提取其元数据
        if hit.raw:
            doc = json.loads(hit.raw)
            results.append({
                "id": doc.get("id", hit.docid),
                "title": doc.get("title", "Untitled Document"),
                "content": doc.get("contents", "No content available."),
                "score": hit.score
            })
        else:
            # 如果没有原始文档，则返回基本信息
            results.append({
                "id": hit.docid,
                "score": hit.score
            })

    return results

