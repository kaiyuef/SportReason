import argparse
import json
from pipeline import qa_pipeline
from dotenv import load_dotenv
load_dotenv()

def process_dataset(input_file, output_dir, index_path, retriever_name, reader_model_name, top_k=5):
    """
    处理数据集，执行 QA 流程，并将结果保存到新的 JSON 文件。
    """
    with open(input_file, "r") as infile:
        dataset = json.load(infile)

    results = []

    for entry in dataset[390:410]:
        seed_question = entry.get("seed_question", "")
        entry_id = entry.get("id", "")

        if not seed_question:
            print(f"Skipping entry {entry_id}: No seed_question found.")
            continue

        print(f"Processing ID {entry_id}: {seed_question}")

        # 调用 QA Pipeline
        output = qa_pipeline(
            query=seed_question,
            index_path=index_path,
            retriever_name=retriever_name,
            reader_model_name=reader_model_name,
            top_k=top_k
        )

        # 提取检索到的 evidence IDs
        retrieved_evidence_ids = [doc["id"] for doc in output.get("retrieved_docs", [])]

        # 将结果保存到结果列表中
        results.append({
            "id": entry_id,
            "seed_question": seed_question,
            "pipeline_output": {
                "query": output["query"],
                "answer": output["answer"],
                "documents_used": output["documents_used"],
                "retrieved_evidence_ids": retrieved_evidence_ids
            }
        })

    # 根据 retriever 和 reader 动态生成输出文件名
    output_file = f"{output_dir}/{retriever_name}_{reader_model_name}.json"

    # 保存结果到输出文件
    with open(output_file, "w") as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    '''
    index list: lucene-index feiss-index
    retriever list: bm25, contriever
    reader list: huggingface-qwen, huggingface-llama
    '''
    input_file = "merged_dataset.json"
    index_path = "indexes/feiss-index"
    retriever = "bm25"
    reader_model = "huggingface-llama"
    top_k = 5
    
    process_dataset(
        input_file=input_file,
        output_dir="answer_results",
        index_path=index_path,
        retriever_name=retriever,
        reranker_name="bge",
        reader_model_name=reader_model,
    )
