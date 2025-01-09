from tqdm import tqdm
import json
from transformers import AutoTokenizer
import spacy

# 参数配置
MAX_TOKENS_PER_CHUNK = 100  # 每个 chunk 的最大 token 数
OVERLAP_TOKENS = 0         # 每个 chunk 的重叠 token 数

# 加载 Hugging Face 分词器和 spacy 模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 替换为所需的分词器模型
nlp = spacy.load("en_core_web_sm")

def load_data_from_jsonl(file_path):
    """
    读取 .jsonl 文件，每行都是一个独立的 JSON 对象。
    仅保留具备 "id" 与非空 "contents" 字段的记录。
    """
    valid_data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"跳过无法解析的行: {line}")
                continue

            if ("id" in entry 
                and "contents" in entry 
                and isinstance(entry["contents"], str) 
                and entry["contents"].strip()):
                valid_data.append(entry)
            else:
                print(f"跳过无效条目: {entry}")
    return valid_data

def split_into_sentences(text):
    """使用 spaCy 分句"""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def chunk_dataset(dataset, tokenizer, max_tokens, overlap):
    """
    将 dataset 的每条数据分句、再分块为不超过 max_tokens 的 chunk。
    若设置 overlap > 0，会让相邻 chunk 有一定 token 重叠。
    """
    chunks = []
    # 1) 遍历 dataset
    for entry in tqdm(dataset, desc="Processing entries"):
        content = entry["contents"]
        entry_id = entry["id"]

        if not content.strip():
            print(f"跳过无效内容: ID={entry_id}")
            continue

        # 按句子分割
        sentences = split_into_sentences(content)

        # 累积句子到 chunk
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            sentence_tokens = tokenizer(sentence, add_special_tokens=False)["input_ids"]
            sentence_length = len(sentence_tokens)

            # 如果当前句子加入后会超出最大 token 限制，先保存当前 chunk
            if current_length + sentence_length > max_tokens:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "id": f"{entry_id}_chunk_{len(chunks) + 1}",
                    "original_id": entry_id,
                    "contents": chunk_text
                })
                current_chunk = []
                current_length = 0

            # 加入当前句子
            current_chunk.append(sentence)
            current_length += sentence_length

        # 最后剩余的 chunk 也要保存
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "id": f"{entry_id}_chunk_{len(chunks) + 1}",
                "original_id": entry_id,
                "contents": chunk_text
            })

    # 2) 处理 chunk 间的重叠
    final_chunks = []
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        chunk_content = chunk["contents"]
        if not isinstance(chunk_content, str):
            raise TypeError(f"Expected string for chunk_content, got {type(chunk_content)}")

        if i > 0 and overlap > 0:
            # 取前一个 chunk 的最后 overlap 个 token，再跟当前 chunk 拼接
            prev_tokens = tokenizer(final_chunks[-1]["contents"], add_special_tokens=False)["input_ids"][-overlap:]
            current_tokens = tokenizer(chunk_content, add_special_tokens=False)["input_ids"]
            combined_tokens = prev_tokens + current_tokens
            chunk["contents"] = tokenizer.decode(combined_tokens)

        final_chunks.append(chunk)

    return final_chunks

def save_to_jsonl(data_list, file_path):
    """
    将 data_list 中的每个元素独立写成一行 JSON，输出到 .jsonl 文件
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # 1) 读取输入 .jsonl 文件
    input_jsonl_path = "expanded_text_content.jsonl"   # 替换为你的输入文件路径
    dataset = load_data_from_jsonl(input_jsonl_path)

    # 2) 分块处理
    chunked_data = chunk_dataset(dataset, tokenizer, MAX_TOKENS_PER_CHUNK, OVERLAP_TOKENS)

    # 3) 输出到 .jsonl 文件（去除 CSV）
    output_jsonl_path = "expanded_chunked_data.jsonl"
    save_to_jsonl(chunked_data, output_jsonl_path)

    print(f"处理完成！结果已保存为 {output_jsonl_path}。")

if __name__ == "__main__":
    main()
