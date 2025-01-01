from retrievers import retrieve_bm25, retrieve_contriever
from readers import HuggingFaceReader

retrievers = [
    {"name": "bm25", "function": retrieve_bm25},
    {"name": "contriever", "function": retrieve_contriever}
]

reader_models = [
    {"name": "huggingface-qwen", "model_name": "Qwen/Qwen2.5-Coder-32B-Instruct"},
    {"name": "huggingface-llama", "model_name": "meta-llama/Llama-2-7b-chat"}
]

def qa_pipeline(query, index_path, retriever_name, reader_model_name, top_k=5):
    retriever = next((r for r in retrievers if r["name"] == retriever_name), None)
    reader_model = next((m for m in reader_models if m["name"] == reader_model_name), None)

    if not retriever:
        raise ValueError(f"Retriever '{retriever_name}' not found!")
    if not reader_model:
        raise ValueError(f"Reader model '{reader_model_name}' not found!")

    print(f"Using Retriever: {retriever_name}")
    print(f"Using Reader Model: {reader_model_name}")

    print("\n[Retriever] Retrieving relevant documents...")
    retrieved_docs = retriever["function"](query, index_path, top_k)

    print("\n[Reader] Generating answers from all retrieved documents...")
    reader = HuggingFaceReader(model_name=reader_model["model_name"])
    final_answer = reader.generate_answer(query, retrieved_docs)

    print("\n[Final Answer]")
    print(f"Query: {final_answer['query']}")
    print(f"Answer: {final_answer['answer']}")
    print(f"Documents Used: {final_answer['documents_used']}\n")

    # 返回检索和生成的完整输出
    return {
        "query": final_answer["query"],
        "answer": final_answer["answer"],
        "documents_used": final_answer["documents_used"],
        "retrieved_docs": retrieved_docs  # 包含检索到的文档详细信息
    }
