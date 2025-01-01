import json
from pyserini.search.lucene import LuceneSearcher
from huggingface_hub import InferenceClient

API_KEY = "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"
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
client = InferenceClient(api_key=API_KEY)

# ============ 示例 Retriever 函数 =============

def retrieve_bm25(query, index_path, top_k=5):
    """
    基于 BM25 的检索器示例。
    """
    searcher = LuceneSearcher(index_path)
    hits = searcher.search(query, k=top_k)
    results = []

    for hit in hits:
        doc = searcher.doc(hit.docid)
        if doc and doc.raw():
            # 解析 JSON 文档
            raw_content = json.loads(doc.raw())
            results.append({
                "id": raw_content.get("id", hit.docid),
                "title": raw_content.get("title", "Untitled Document"),
                "content": raw_content.get("contents", "No content available."),
                "score": hit.score
            })
    return results


def retrieve_dummy(query, index_path, top_k=5):
    """
    一个假的/测试用的检索器示例。
    通常用于演示或者测试“Reader”部分功能，不进行实际检索。
    """
    # 这里直接返回一个“虚拟文档列表”
    results = [
        {
            "id": "dummy-doc-1",
            "title": "Dummy Document Title",
            "content": "This is a dummy content for testing the reader part.",
            "score": 0.0
        }
    ]
    return results


# ============ 示例 Reader 函数 =============

def reader_huggingface_inference(query, retrieved_docs):
    """
    使用 Hugging Face InferenceClient 生成答案的阅读器。
    """
    # 将检索到的文档内容整理到提示（prompt）中
    context = "\n\n".join(
        [
            f"Document {i+1} Title: {doc['title']}\n"
            f"Content: {doc['content'][:1000]}"  # 截断过长的内容以适应 token 限制
            for i, doc in enumerate(retrieved_docs)
        ]
    )

    prompt = (
        f"Answer the following question based on the context provided:\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\nAnswer:"
    )

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1000
        )
        answer = completion.choices[0].message["content"].strip()
    except Exception as e:
        answer = f"Error generating answer: {e}"

    return {
        "query": query,
        "answer": answer,
        "documents_used": len(retrieved_docs)
    }


def reader_dummy(query, retrieved_docs):
    """
    一个简单的阅读器示例，直接拼接文档内容返回。
    用于测试管道的整体流程，不会产生真正的“答案”推理。
    """
    concatenated_docs = "\n".join(
        [f"[{doc['title']}]: {doc['content']}" for doc in retrieved_docs]
    )
    return {
        "query": query,
        "answer": f"Dummy Reader: Here are the docs:\n\n{concatenated_docs}",
        "documents_used": len(retrieved_docs)
    }


# ============ Pipeline =============

def qa_pipeline(query, index_path, retriever_fn, reader_fn, top_k=5):
    """
    可以灵活切换 Retriever 和 Reader 的 QA 管道。
    :param query: 用户查询
    :param index_path: Lucene 索引路径或者其他需要的检索器配置路径
    :param retriever_fn: 外部传入的检索函数
    :param reader_fn: 外部传入的阅读/生成函数
    :param top_k: 检索的文档数
    :return: 最终答案（字典）
    """
    print(f"Query: {query}")
    print("\n[Retriever] Retrieving relevant documents...")

    # 使用传入的检索函数
    retrieved_docs = retriever_fn(query, index_path, top_k=top_k)

    # 输出检索结果
    for i, doc in enumerate(retrieved_docs):
        print(f"Document {i+1}:")
        print(f"  ID: {doc['id']}")
        print(f"  Title: {doc['title']}")
        print(f"  Score: {doc.get('score', 'N/A')}")
        print(f"  Content Preview: {doc['content'][:200]}...\n")

    print("[Reader] Generating answers from all retrieved documents...")

    # 使用传入的阅读器函数
    final_answer = reader_fn(query, retrieved_docs)

    # 输出最终答案
    print("\n[Final Answer]")
    print(f"Query: {final_answer['query']}")
    print(f"Answer: {final_answer['answer']}")
    print(f"Documents Used: {final_answer['documents_used']}\n")

    return final_answer


if __name__ == "__main__":
    # 示例配置
    index_path = "/Users/kevinfeng/numsports/indexes/lucene-index"
    query = "How old was Singh when he won the Master's Tournament?"

    # 可以灵活切换不同的 retriever 和 reader
    # 1. 使用 BM25 + Hugging Face Reader
    print("===== BM25 + HF Reader =====")
    qa_pipeline(
        query=query,
        index_path=index_path,
        retriever_fn=retrieve_bm25,
        reader_fn=reader_huggingface_inference,
        top_k=2
    )

    # 2. 使用 Dummy Retriever + Dummy Reader
    print("===== Dummy Retriever + Dummy Reader =====")
    qa_pipeline(
        query=query,
        index_path=index_path,
        retriever_fn=retrieve_dummy,
        reader_fn=reader_dummy,
        top_k=2
    )
