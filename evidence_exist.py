import json
import re
import logging
from tqdm import tqdm

try:
    from rapidfuzz.fuzz import partial_ratio
except ImportError:
    def partial_ratio(a, b):
        return 100 if a in b else 0

logging.basicConfig(
    filename='gold_evidence.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

#######################
# 1. 递归扁平化函数
#######################
def flatten_to_string(data):
    if isinstance(data, dict):
        parts = []
        for k, v in data.items():
            parts.append(str(k))
            parts.append(flatten_to_string(v))
        return ' '.join(parts)
    elif isinstance(data, list):
        return ' '.join([flatten_to_string(item) for item in data])
    else:
        return str(data)

#######################
# 2. 清洗与模糊匹配
#######################
def clean_content(content):
    content_str = flatten_to_string(content)
    content_str = re.sub(r'[^\w\s]', '', content_str)
    content_str = re.sub(r'\s+', ' ', content_str)
    return content_str.strip().lower()

def is_content_similar(gold_obj, corpus_obj, threshold=70):
    gold_str_clean = clean_content(gold_obj)
    corpus_str_clean = clean_content(corpus_obj)
    score = partial_ratio(gold_str_clean, corpus_str_clean)
    return score, (score >= threshold)

def get_gold_type(gold_id):
    if gold_id.startswith("text"):
        return "text"
    elif gold_id.startswith("table"):
        return "table"
    elif gold_id.startswith("infobox"):
        return "infobox"
    return None

#######################
# 3. 核心更新逻辑
#######################
def update_corpus_with_gold_evidence(dataset, text_corpus, table_corpus, infobox_corpus):
    text_dict = {page["title"]: page for page in text_corpus if "title" in page}
    table_dict = {page["title"]: page for page in table_corpus if "title" in page}
    infobox_dict = {page["title"]: page for page in infobox_corpus if "title" in page}

    for data in tqdm(dataset, desc="Processing dataset"):
        for gold in data.get("gold_evidences", []):
            gold_id = gold.get("id", "")
            gold_content = gold.get("content", "")
            gold_type = get_gold_type(gold_id)
            if not gold_type:
                logging.warning(f"[SKIP] Unrecognized gold_id={gold_id}")
                continue

            title = gold.get("title", f"Untitled-{gold_id}")
            logging.info(f"[PROCESS] gold_id={gold_id}, title={title}, type={gold_type}")

            if gold_type == "text":
                corpus_list = text_corpus
                corpus_dict = text_dict
            elif gold_type == "table":
                corpus_list = table_corpus
                corpus_dict = table_dict
            else:
                corpus_list = infobox_corpus
                corpus_dict = infobox_dict

            if title in corpus_dict:
                # 如果标题已存在，进行相似度检查并更新内容
                page = corpus_dict[title]
                existing_contents = page.get("contents", "")

                score, is_sim = is_content_similar(gold_content, existing_contents, threshold=70)
                logging.info(
                    f"[EXIST] Found doc title='{title}' (id={page.get('id','NoID')}). "
                    f"Similarity score={score}, is_similar={is_sim}"
                )

                if is_sim:
                    # 更新 merged_dataset 中 gold 的 id
                    gold["id"] = page["id"]
                else:
                    # 不相似，则追加
                    combined = flatten_to_string(existing_contents) + "\n" + flatten_to_string(gold_content)
                    page["contents"] = combined
                    logging.info(f"[UPDATE] Appended new content to existing doc (title={title}).")
                    gold["id"] = page["id"]  # 更新 id
            else:
                # 如果标题不存在，对整个 corpus_list 进行相似度检查
                is_similar_found = False
                for page in corpus_list:
                    existing_contents = page.get("contents", "")
                    score, is_sim = is_content_similar(gold_content, existing_contents, threshold=70)
                    if is_sim:
                        is_similar_found = True
                        logging.info(
                            f"[SKIP] Found similar content in doc (id={page['id']}, title={page['title']}). "
                            f"Similarity score={score}. No new content added."
                        )
                        gold["id"] = page["id"]  # 更新 id
                        break

                if not is_similar_found:
                    # 未找到相似内容 -> 创建新的文档
                    new_id = f"{gold_type}_{len(corpus_list)}"
                    new_page = {
                        "id": new_id,
                        "title": title,
                        "url": gold.get("meta", {}).get("url", ""),
                        "contents": flatten_to_string(gold_content),
                        "added_by": "gold_evidence_script"
                    }
                    corpus_list.append(new_page)
                    corpus_dict[title] = new_page
                    logging.info(f"[NEW] Created new doc with title='{title}', id={new_id}")
                    gold["id"] = new_id  # 更新 id

    return dataset, text_corpus, table_corpus, infobox_corpus


#######################
# 4. 最终写回前，统一转字符串
#######################
def finalize_corpus_for_output(*corpus_lists):
    for corpus_list in corpus_lists:
        for doc in corpus_list:
            doc["contents"] = flatten_to_string(doc["contents"])

#######################
# 5. 新增：保存为 .jsonl
#######################
def save_jsonl(data_list, filename):
    """
    将列表 data_list 中的每个元素独立写成一行 JSON，保存到 filename。
    """
    with open(filename, "w", encoding="utf-8") as f:
        for item in data_list:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + "\n")

#######################
# 6. 主函数
#######################
if __name__ == "__main__":
    # 1) 读取原始数据
    with open("merged_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open("text_content.json", "r", encoding="utf-8") as f:
        text_corpus = json.load(f)

    with open("tables_content.json", "r", encoding="utf-8") as f:
        table_corpus = json.load(f)

    with open("infobox_content.json", "r", encoding="utf-8") as f:
        infobox_corpus = json.load(f)

    # 2) 更新
    dataset, text_corpus, table_corpus, infobox_corpus = update_corpus_with_gold_evidence(
        dataset, text_corpus, table_corpus, infobox_corpus
    )

    # 3) 统一转成字符串
    finalize_corpus_for_output(text_corpus, table_corpus, infobox_corpus)

    # 4) 保存更新后的数据
    save_jsonl(text_corpus, "text_content_updated.jsonl")
    save_jsonl(table_corpus, "tables_content_updated.jsonl")
    save_jsonl(infobox_corpus, "infobox_content_updated.jsonl")
    save_jsonl(dataset, "merged_dataset_updated.jsonl")

    print("Corpus update completed. Updated JSONL files were generated, including merged_dataset_updated.jsonl.")
