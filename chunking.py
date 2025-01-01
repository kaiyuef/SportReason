from tqdm import tqdm
import json
import pandas as pd
from transformers import AutoTokenizer
import spacy

# 参数配置
MAX_TOKENS_PER_CHUNK = 100  # 每个 chunk 的最大 token 数
OVERLAP_TOKENS = 0  # 每个 chunk 的重叠 token 数

# 加载 Hugging Face 分词器和 spacy 模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 替换为所需的分词器模型
nlp = spacy.load("en_core_web_sm")

# 从 JSON 文件加载数据
def load_data_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    valid_data = []
    for entry in data:
        if "id" in entry and "contents" in entry and isinstance(entry["contents"], str) and entry["contents"].strip():
            valid_data.append(entry)
        else:
            print(f"跳过无效条目: {entry}")
    return valid_data

# 分块处理
def chunk_dataset(dataset, tokenizer, max_tokens, overlap):
    chunks = []
    for entry in tqdm(dataset, desc="Processing entries"):
        content = entry["contents"]
        entry_id = entry["id"]

        # 如果 content 为空，跳过该 entry
        if not content.strip():
            print(f"跳过无效内容: ID={entry_id}")
            continue

        # 按句子分割
        sentences = split_into_sentences(content)

        # 按句子拼接到 chunk
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            sentence_tokens = tokenizer(sentence, add_special_tokens=False)["input_ids"]
            sentence_length = len(sentence_tokens)

            # 如果当前句子超出最大 token 限制，则保存当前 chunk
            if current_length + sentence_length > max_tokens:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "id": f"{entry_id}_chunk_{len(chunks) + 1}",
                    "original_id": entry_id,
                    "contents": chunk_text
                })
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length

        # 保存最后的 chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "id": f"{entry_id}_chunk_{len(chunks) + 1}",
                "original_id": entry_id,
                "contents": chunk_text
            })

    # 处理 chunk 重叠
    final_chunks = []
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        chunk_content = chunk["contents"]  # 从字典中提取 chunk 内容
        if not isinstance(chunk_content, str):
            raise TypeError(f"Expected a string for chunk_content, but got {type(chunk_content)}: {chunk_content}")

        if i > 0 and overlap > 0:
            prev_tokens = tokenizer(final_chunks[-1]["contents"], add_special_tokens=False)["input_ids"][-overlap:]
            current_tokens = tokenizer(chunk_content, add_special_tokens=False)["input_ids"]
            combined_tokens = prev_tokens + current_tokens
            chunk["contents"] = tokenizer.decode(combined_tokens)

        final_chunks.append(chunk)

    return final_chunks

# 分句工具
def split_into_sentences(text):
    """使用 spacy 分句"""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

# 主流程
def main():
    file_path = "text_content_updated.json"  # 替换为实际文件路径
    dataset = load_data_from_json(file_path)

    # 执行分块
    chunked_data = chunk_dataset(dataset, tokenizer, MAX_TOKENS_PER_CHUNK, OVERLAP_TOKENS)

    # 将处理好的 chunked 数据保存为 JSON 和 CSV
    output_json_path = "chunked_data.json"
    output_csv_path = "chunked_data.csv"

    # 保存为 JSON 文件（包含列表）
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(chunked_data, json_file, ensure_ascii=False, indent=2)

    # 保存为 CSV 文件
    chunked_df = pd.DataFrame(chunked_data)
    chunked_df.to_csv(output_csv_path, index=False, encoding="utf-8")

    print(f"完成！结果已保存为 {output_json_path} 和 {output_csv_path}。")

# 执行主函数
if __name__ == "__main__":
    main()
