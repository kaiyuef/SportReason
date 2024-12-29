import re
import json
from tqdm import tqdm
import logging

# 如果需要使用 fuzzy matching
from rapidfuzz.fuzz import partial_ratio

logging.basicConfig(
    filename="gold_evidence_comparisons.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

######################################
#          1. 辅助函数定义
######################################

def get_gold_type(gold_id):
    """
    Determine the type based on the gold_evidence's id.
    """
    if gold_id.startswith("text"):
        return "text"
    elif gold_id.startswith("table"):
        return "table"
    elif gold_id.startswith("infobox"):
        return "infobox"
    else:
        return None

def clean_content(content):
    """
    Clean and normalize content by removing punctuation and extra whitespace.
    """
    content = re.sub(r'[^\w\s]', '', content)  # remove punctuation
    content = re.sub(r'\s+', ' ', content).strip()  # normalize whitespace
    return content.lower()

def extract_text_from_dict(data):
    """
    Recursively extract all keys and values from a dictionary or list,
    return them as a list of strings.
    """
    text_elements = []
    if isinstance(data, dict):
        for key, value in data.items():
            text_elements.append(str(key))
            text_elements.extend(extract_text_from_dict(value))
    elif isinstance(data, list):
        for item in data:
            text_elements.extend(extract_text_from_dict(item))
    else:
        text_elements.append(str(data))
    return text_elements


##############################
#     1.1 表格转字符串
##############################

def table_dict_to_string(table_dict):
    """
    将一个表格的 {columns: [...], rows: [...]} 字典结构
    转换为可读的单一字符串格式。
    """
    if not isinstance(table_dict, dict):
        return str(table_dict)

    columns = table_dict.get("columns", [])
    rows = table_dict.get("rows", [])

    lines = []
    # 将 columns 列表连成一行
    col_line = " | ".join(str(col) for col in columns)
    lines.append(f"Columns: {col_line}")

    # 依次处理每行
    for idx, row in enumerate(rows, start=1):
        row_line = " | ".join(str(cell) for cell in row)
        lines.append(f"Row{idx}: {row_line}")

    return "\n".join(lines)

def list_of_tables_to_string(table_list):
    """
    如果 table_list 中包含多个表格，则依次转换后，用空行拼接。
    """
    if not isinstance(table_list, list):
        return str(table_list)
    
    converted = []
    for i, tbl in enumerate(table_list, start=1):
        tbl_str = table_dict_to_string(tbl)
        converted.append(f"--- Table {i} ---\n{tbl_str}")
    return "\n\n".join(converted)

##############################
#     1.2 Infobox转字符串
##############################

def infobox_to_string(infobox_data):
    """
    将 infobox（字典或列表）递归提取为文本。
    """
    text_elements = extract_text_from_dict(infobox_data)
    return " ".join(text_elements)


##############################
#     1.3 文本转字符串
##############################

def text_to_string(text_data):
    """
    如果是列表，拼成一个多段落字符串；
    如果是字典，就尝试取其中的 'text'；
    否则直接转 str。
    """
    if isinstance(text_data, list):
        return "\n".join(str(item) for item in text_data)
    elif isinstance(text_data, dict):
        if "text" in text_data:
            return str(text_data["text"])
        else:
            # 如果没有 text 字段，就把整个 dict 转成字符串
            return json.dumps(text_data, ensure_ascii=False)
    else:
        return str(text_data)

##############################
#     1.4 flatten + clean
##############################

def flatten_content(content, content_type):
    """
    将content扁平化并清洗后用于匹配。
    - text -> 返回字符串
    - table -> 返回多个字符串（列表）
    - infobox -> 返回多个字符串（列表）
    """
    if content_type == "text":
        # 直接提取并清洗
        raw_string = text_to_string(content)
        return clean_content(raw_string)
    elif content_type == "table":
        # 可能有多个表格
        if isinstance(content, list):
            return [clean_content(table_dict_to_string(tbl)) for tbl in content]
        elif isinstance(content, dict):
            return [clean_content(table_dict_to_string(content))]
        else:
            return [clean_content(str(content))]
    elif content_type == "infobox":
        # 可能有多个 infobox
        if isinstance(content, list):
            return [clean_content(infobox_to_string(item)) for item in content]
        else:
            return [clean_content(infobox_to_string(content))]
    else:
        return ""

######################################
#   2. 相似度判断函数
######################################

def is_content_similar(gold_content, corpus_page, content_type="text", dataset_id=None, gold_id=None):
    """
    判断 gold_content 是否与 corpus_page 的 contents 相似；
    返回 (corpus_id, bool)。
    """
    global text_length, table_length, infobox_length

    corpus_id = corpus_page.get("id", "unknown_id")
    existing_content = corpus_page.get("contents", "")

    if content_type == "text":
        # Flatten gold
        gold_str = flatten_content(gold_content, "text")
        if not gold_str:
            return corpus_id, False
        
        # Flatten corpus
        corpus_str = flatten_content(existing_content, "text")
        if not corpus_str:
            return corpus_id, False

        # fuzzy matching
        match_score = partial_ratio(gold_str, corpus_str)
        is_match = (match_score >= 70)
        if not is_match:
            logging.info(
                f"[Text match] dataset={dataset_id} gold_id={gold_id}\n"
                f"Gold: {gold_str}\nCorpus: {corpus_str}\nScore={match_score}"
            )
        return corpus_id, is_match

    elif content_type == "table":
        # gold -> list of strings (maybe just one)
        gold_list = flatten_content(gold_content, "table")
        # corpus -> list of table strings
        corpus_list = flatten_content(existing_content, "table")

        # 取 gold_list[0] 进行比对
        if not gold_list:
            return corpus_id, False
        gold_table_text = gold_list[0]

        # 遍历
        for cstr in corpus_list:
            match_score = partial_ratio(gold_table_text, cstr)
            if match_score >= 50:
                return corpus_id, True
            else:
                logging.info(
                    f"[Table match] dataset={dataset_id} gold_id={gold_id}\n"
                    f"Gold: {gold_table_text}\nCorpus: {cstr}\nScore={match_score}"
                )
        return corpus_id, False

    else:  # infobox
        gold_list = flatten_content(gold_content, "infobox")
        corpus_list = flatten_content(existing_content, "infobox")

        if not gold_list:
            return corpus_id, False
        gold_infobox_text = gold_list[0]

        for cstr in corpus_list:
            match_score = partial_ratio(gold_infobox_text, cstr)
            if match_score >= 50:
                return corpus_id, True
            else:
                logging.info(
                    f"[Infobox match] dataset={dataset_id} gold_id={gold_id}\n"
                    f"Gold: {gold_infobox_text}\nCorpus: {cstr}\nScore={match_score}"
                )
        return corpus_id, False

######################################
#   3. 将最终contents转成 string
######################################

def finalize_page_contents_text(page):
    """
    最终让 text_corpus 中的 contents 变为单字符串。
    """
    contents = page.get("contents", "")
    # 统一转字符串
    return text_to_string(contents)

def finalize_page_contents_table(page):
    """
    最终让 table_corpus 中的 contents 变为单字符串。
    """
    contents = page.get("contents", "")
    # 如果是列表(多个表)，用 list_of_tables_to_string
    if isinstance(contents, list):
        return list_of_tables_to_string(contents)
    elif isinstance(contents, dict):
        return table_dict_to_string(contents)
    else:
        return str(contents)

def finalize_page_contents_infobox(page):
    """
    最终让 infobox_corpus 中的 contents 变为单字符串。
    """
    contents = page.get("contents", "")
    if isinstance(contents, dict) or isinstance(contents, list):
        return infobox_to_string(contents)
    else:
        return str(contents)

def finalize_corpus(text_corpus, table_corpus, infobox_corpus):
    """
    依次遍历每种语料库，将其 contents 转为字符串，方便 BM25 之类检索。
    """
    # text
    for p in text_corpus:
        p["contents"] = finalize_page_contents_text(p)
    
    # table
    for p in table_corpus:
        p["contents"] = finalize_page_contents_table(p)
    
    # infobox
    for p in infobox_corpus:
        p["contents"] = finalize_page_contents_infobox(p)
    
    return text_corpus, table_corpus, infobox_corpus

######################################
#   4. 核心更新函数
######################################

def update_corpus_with_gold_evidence(dataset, text_corpus, table_corpus, infobox_corpus):
    """
    对数据集中每条 gold_evidence，与现有语料对比/合并/更新。
    """
    global text_length, table_length, infobox_length

    # 初始化全局计数器
    text_length = len(text_corpus) if isinstance(text_corpus, list) else 0
    infobox_length = len(infobox_corpus) if isinstance(infobox_corpus, list) else 0
    table_length = 0
    if isinstance(table_corpus, list):
        for element in table_corpus:
            # 如果 table_corpus里每个 element["contents"] 是 list，就累加
            if "contents" in element and isinstance(element["contents"], list):
                table_length += len(element["contents"])

    # 建立快速查找
    text_corpus_dict = {}
    for page in text_corpus:
        if "title" in page:
            text_corpus_dict[page["title"]] = page

    table_corpus_dict = {}
    for page in table_corpus:
        if "title" in page:
            table_corpus_dict[page["title"]] = page

    infobox_corpus_dict = {}
    for idx, page in enumerate(infobox_corpus):
        title = page.get("title", f"Untitled-{idx}")
        infobox_corpus_dict[title] = page

    # 遍历 dataset
    for data in tqdm(dataset, desc="Processing dataset entries"):
        dataset_id = data["id"]
        dataset_name = data.get("seed_dataset", "Unknown")

        for gold in tqdm(data["gold_evidences"], desc=f"Gold evidences for ID={dataset_id}", leave=False):
            gold_id = gold.get("id", "")
            gold_content = gold.get("contents", {})
            gold_type = get_gold_type(gold_id)

            if gold_type == "text":
                corpus = text_corpus
                corpus_dict = text_corpus_dict
                content_type = "text"
            elif gold_type == "table":
                corpus = table_corpus
                corpus_dict = table_corpus_dict
                content_type = "table"
            elif gold_type == "infobox":
                corpus = infobox_corpus
                corpus_dict = infobox_corpus_dict
                content_type = "infobox"
            else:
                continue  # 未知类型

            title = gold.get("title", f"Untitled-{gold_id}")

            # 特殊处理 infobox 无标题
            if gold_type == "infobox" and (dataset_name == 'temptableqa' or not title):
                title = f"Untitled-{gold_id}"

            if title in corpus_dict:
                # 已存在
                corpus_page = corpus_dict[title]
                corpus_id, match = is_content_similar(gold_content, corpus_page, content_type, dataset_id, gold_id)

                match_str = "True" if match else "False"
                if "used_in" not in corpus_page:
                    corpus_page["used_in"] = []
                used_in_entry = {"dataset": dataset_name, "id": corpus_id, "match": match_str}
                if used_in_entry not in corpus_page["used_in"]:
                    corpus_page["used_in"].append(used_in_entry)

                if not match:
                    # 合并/追加 gold_content 到 corpus_page["contents"]
                    existing = corpus_page.get("contents", "")
                    if not isinstance(existing, str):
                        # 不是字符串的话直接转str
                        existing = str(existing)
                    new_content_str = str(gold_content)
                    # 简单拼接
                    combined = existing + "\n" + new_content_str
                    corpus_page["contents"] = combined
            else:
                # 不存在 -> 新建
                if content_type == "text":
                    vari = text_length
                    text_length += 1
                elif content_type == "table":
                    vari = table_length
                    table_length += 1
                else:
                    vari = infobox_length
                    infobox_length += 1

                new_entry = {
                    "title": title if title else None,
                    "url": gold.get("meta", {}).get("url", ""),
                    "contents": gold_content,  # 先直接放这里, 等最后再统一转字符串
                    "used_in": [{
                        "dataset": dataset_name,
                        "id": f"{content_type}_{vari}",
                        "match": "Not found"
                    }],
                    "added_by": "gold_evidence_script",
                }

                corpus.append(new_entry)
                corpus_dict[title] = new_entry

    return text_corpus, table_corpus, infobox_corpus


######################################
#   5. 主程序
######################################

if __name__ == "__main__":
    # 读取 dataset
    with open("merged_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 读取三种 corpus
    with open("text_content.json", "r", encoding="utf-8") as f:
        text_corpus = json.load(f)

    with open("tables_content.json", "r", encoding="utf-8") as f:
        table_corpus = json.load(f)

    with open("infobox_content.json", "r", encoding="utf-8") as f:
        infobox_corpus = json.load(f)

    # 更新语料
    text_corpus, table_corpus, infobox_corpus = update_corpus_with_gold_evidence(
        dataset, text_corpus, table_corpus, infobox_corpus
    )

    # 最终把contents转成字符串
    text_corpus, table_corpus, infobox_corpus = finalize_corpus(
        text_corpus, table_corpus, infobox_corpus
    )

    # 写回文件
    with open("text_content_updated.json", "w", encoding="utf-8") as f:
        json.dump(text_corpus, f, ensure_ascii=False, indent=4)

    with open("tables_content_updated.json", "w", encoding="utf-8") as f:
        json.dump(table_corpus, f, ensure_ascii=False, indent=4)

    with open("infobox_content_updated.json", "w", encoding="utf-8") as f:
        json.dump(infobox_corpus, f, ensure_ascii=False, indent=4)

    print("Corpus update completed.")
