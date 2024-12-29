from pyserini.search.lucene import LuceneSearcher
from vllm import LLM, SamplingParams
import json
import os


# ---------- Step 1: Retriever ----------
def retrieve_docs(query, index_path, top_k=5):
    """
    使用 BM25 检索与查询相关的文档。
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


# ---------- Step 2: Reader ----------
def generate_answers(query, retrieved_docs, model_name="meta-llama/Llama-2-8b-chat-hf"):
    """
    使用 vLLM 模型生成答案。
    """
    llm = LLM(model=model_name)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
    answers = []

    for doc in retrieved_docs:
        prompt = (
            f"Answer the following question based on the context provided:\n\n"
            f"Document Title: {doc['title']}\n"
            f"Context:\n{doc['content']}\n\n"
            f"Question:\n{query}\n\nAnswer:"
        )
        outputs = llm.generate(prompt, sampling_params)
        answer = outputs[0].outputs[0].text.strip()
        answers.append({
            "doc_id": doc["id"],
            "doc_title": doc["title"],
            "answer": answer
        })

    return answers


# ---------- Step 3: Pipeline Integration ----------
def qa_pipeline(query, index_path, top_k=5, model_name="meta-llama/Llama-2-8b-chat-hf"):
    """
    检索和阅读结合的问答系统管道。
    """
    print(f"Query: {query}")

    # Step 1: 检索
    print("\n[Retriever] Retrieving relevant documents...")
    retrieved_docs = retrieve_docs(query, index_path, top_k=top_k)

    # 输出检索结果
    for i, doc in enumerate(retrieved_docs):
        print(f"Document {i+1}:")
        print(f"  ID: {doc['id']}")
        print(f"  Title: {doc['title']}")
        print(f"  Score: {doc['score']}")
        print(f"  Content Preview: {doc['content'][:200]}...\n")

    # Step 2: 阅读并生成答案
    print("[Reader] Generating answers from retrieved documents...")
    answers = generate_answers(query, retrieved_docs, model_name=model_name)

    # 输出最终答案
    print("\n[Final Answers]")
    for answer in answers:
        print(f"Document ID: {answer['doc_id']}")
        print(f"Document Title: {answer['doc_title']}")
        print(f"Answer: {answer['answer']}\n")

    return answers


# ---------- Step 4: Main Execution ----------
if __name__ == "__main__":
    # 配置路径
    index_path = "/Users/kevinfeng/numsports/indexes/lucene-index"
    query = "What are the top teams in FIFA World Cup history?"

    # 运行 QA 管道
    qa_pipeline(query, index_path, top_k=5, model_name="meta-llama/Llama-2-8b-chat-hf")
