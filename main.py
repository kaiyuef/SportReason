import argparse
import json
from pipeline import qa_pipeline

def process_dataset(input_file, output_file, index_path, retriever_name, reader_model_name, top_k=5):
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

        # 将结果保存到结果列表中
        results.append({
            "id": entry_id,
            "seed_question": seed_question,
            "pipeline_output": output
        })

    # 保存结果到输出文件
    with open(output_file, "w") as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":

    input_file = "merged_dataset.json"
    output_file = "sample_dataset_results.json"
    index_path = "indexes/lucene-index"
    retriever = "bm25"
    reader_model = "huggingface-qwen"
    top_k = 5
    
    process_dataset(
        input_file=input_file,
        output_file=output_file,
        index_path=index_path,
        retriever_name=retriever,
        reader_model_name=reader_model,
        top_k=top_k
    )
