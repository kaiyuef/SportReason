import json
import logging
import os
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from rapidfuzz.fuzz import partial_ratio

# 如果使用 fuzzy matching，需要 import，例如：
# from rapidfuzz.fuzz import partial_ratio

logging.basicConfig(
    filename='process_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def sanitize_filename(filename):
    invalid_chars = r'[<>:"/\\|?*\n\t]'
    sanitized = re.sub(invalid_chars, '_', filename)
    return sanitized.strip()

def extract_infobox(soup):
    infobox = soup.find('table', {'class': 'infobox'})
    if infobox is None:
        return None
    
    rows = infobox.find_all('tr')
    infobox_data = {}
    current_section = None
    
    for row in rows:
        header = row.find('th')
        data = row.find('td')
        
        if header and data:
            header_text = header.text.strip()
            data_text = ' | '.join([line for line in data.stripped_strings])
            if header_text:
                if current_section is None:
                    current_section = header_text
                    infobox_data[current_section] = {header_text: data_text}
                else:
                    infobox_data[current_section][header_text] = data_text
        elif header:
            current_section = header.text.strip()
            infobox_data[current_section] = {}
    
    return infobox_data

def calculate_jaccard_similarity(text1, text2):
    if isinstance(text1, list):
        text1 = " ".join(text1)
    if isinstance(text2, list):
        text2 = " ".join(text2)
    
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def clean_content(content):
    """
    Clean and normalize content by removing punctuation and extra whitespace.
    """
    content = re.sub(r'[^\w\s]', '', content)
    content = re.sub(r'\s+', ' ', content).strip()
    return content.lower()

def flatten_content(content, content_type):
    """此处可根据需要保留或修改。"""
    if content_type == "text":
        return clean_content(content) if isinstance(content, str) else ""
    elif content_type == "infobox":
        return clean_content(str(content))
    elif content_type == "table":
        # 如果是字符串，就直接清洗
        if isinstance(content, str):
            return clean_content(content)
        # 如果是 dict, 可能包含表格的各种字段
        elif isinstance(content, dict):
            combined = []
            # 提取标题或单元格
            for k, v in content.items():
                combined.append(str(k))
                combined.append(str(v))
            return clean_content(" ".join(combined))
    return ""

def is_content_similar(gold_content, corpus_content, content_type="text", dataset_id=None, gold_id=None):
    """
    判断 gold_content 和 corpus_content 是否相似的示例。
    如果不需要可删除或修改。
    """
    gold_text = flatten_content(gold_content, content_type)
    if not gold_text:
        return False

    corpus_text = flatten_content(corpus_content, content_type)
    if not corpus_text:
        return False

    # 这里可以用任何相似度逻辑，以下仅示例
    sim_score = calculate_jaccard_similarity(gold_text, corpus_text)
    return sim_score >= 0.5

def extract_text_from_html(soup):
    """
    更完善地提取文本：标题段落 + p + ul + ol。
    """
    content_div = soup.find('div', class_='mw-parser-output')
    if not content_div:
        return ""

    skip_sections = {"References", "External links", "See also", "Further reading", "Notes"}
    lines = []
    skip_rest = False

    for element in content_div.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol'], recursive=True):
        if skip_rest:
            break

        if element.name in ['h2', 'h3', 'h4']:
            section_title = element.get_text(strip=True)
            if section_title in skip_sections:
                skip_rest = True
                continue
            lines.append(f"## {section_title}")
        else:
            paragraph_text = element.get_text(separator=" ", strip=True)
            if paragraph_text:
                lines.append(paragraph_text)

    return "\n\n".join(lines)

def get_wikipedia_content_from_file(title, url, html_dir):
    """
    返回: text_data, infobox_data, tables_data_list
      - text_data: {title, url, contents, used_in}
      - infobox_data: {title, url, contents, used_in}
      - tables_data_list: [ {title, url, id, contents, used_in}, ...]  <-- 一张表一个元素
    """
    valid_filename = sanitize_filename(title).replace(' ', '_') + '.html'
    file_path = os.path.join(html_dir, valid_filename)

    if not os.path.exists(file_path):
        logging.error(f"HTML file for {title} not found at {file_path}")
        return None, None, []

    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1) 提取正文文本
    text_string = extract_text_from_html(soup)
    text_data = {
        "title": title,
        "url": url,
        "contents": text_string,
        "used_in": []
    }

    # 2) 提取infobox
    infobox_dict = extract_infobox(soup)
    infobox_data = {
        "title": title,
        "url": url,
        "contents": infobox_dict if infobox_dict else {},
        "used_in": []
    }

    # 3) 提取所有 table，并让每一个 table 成为独立的 element
    tables_data_list = []
    content_div = soup.find('div', class_='mw-parser-output')
    if content_div:
        # 找到所有带wikitable类的表
        tables = content_div.find_all('table', {'class': 'wikitable'})
        for idx, table in enumerate(tables):
            headers = [
                header_cell.get_text(separator=" ", strip=True) 
                for header_cell in table.find_all('th')
            ]
            rows = []
            table_rows = table.find_all('tr')
            # 从第二行开始解析数据
            for row in table_rows[1:]:
                row_cells = [
                    cell.get_text(separator=" ", strip=True)
                    for cell in row.find_all(['td', 'th'])
                ]
                rows.append(row_cells)

            # 给每个表一个独立 ID
            table_item = {
                "title": title,
                "url": url,
                "id": f"{title}_table_{idx}",  # 或者其他 ID 规则
                "contents": {
                    "columns": headers,
                    "rows": rows
                },
                "used_in": []
            }
            tables_data_list.append(table_item)

    return text_data, infobox_data, tables_data_list

def process_dataset(input_file, text_output_file, infobox_output_file, tables_output_file, html_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    text_results = []
    infobox_results = []
    tables_results = []  # 用于存放所有表格的独立元素

    for idx, element in tqdm(enumerate(dataset), total=len(dataset), desc="Processing datasets"):
        for evidence in element['gold_evidences']:
            url = evidence.get('meta', {}).get('url', '')
            title = evidence.get('title', '')

            if not url or not title:
                continue

            # 注意 get_wikipedia_content_from_file 第三个返回值现在是 "列表"
            text_data, infobox_data, tables_data_list = get_wikipedia_content_from_file(title, url, html_dir)

            # 如果文件不存在，会返回 None, None, []
            # 做个判空判断
            if text_data is None:
                continue

            # 判断与 gold_content 相似的部分 (可选)
            # 这里只是个演示，如果你想对 text_data, tables_data_list 依次做 is_content_similar 等操作的话，可以自己改写
            for gold_content in [evidence]:
                # 1) text部分
                match_text = is_content_similar(gold_content, text_data["contents"], content_type="text")
                if match_text:
                    match_status_text = "True"
                else:
                    match_status_text = "False"
                text_data["used_in"].append({
                    "dataset": element.get("seed_dataset", "unknown"),
                    "id": evidence.get("id", "unknown"),
                    "match": match_status_text
                })

                # 2) 对每个table分别判断
                for table_item in tables_data_list:
                    match_table = is_content_similar(gold_content, table_item["contents"], content_type="table")
                    match_status_table = "True" if match_table else "False"
                    table_item["used_in"].append({
                        "dataset": element.get("seed_dataset", "unknown"),
                        "id": evidence.get("id", "unknown"),
                        "match": match_status_table
                    })
            
            # 加入去重逻辑（避免重复）
            if text_data and text_data not in text_results:
                text_results.append(text_data)
            if infobox_data and infobox_data not in infobox_results:
                infobox_results.append(infobox_data)

            # 把 tables_data_list 的每个表都追加到 tables_results
            for tbl in tables_data_list:
                if tbl not in tables_results:
                    tables_results.append(tbl)

    # 写入输出文件
    with open(text_output_file, 'w', encoding='utf-8') as f:
        json.dump(text_results, f, ensure_ascii=False, indent=4)

    with open(infobox_output_file, 'w', encoding='utf-8') as f:
        json.dump(infobox_results, f, ensure_ascii=False, indent=4)

    with open(tables_output_file, 'w', encoding='utf-8') as f:
        json.dump(tables_results, f, ensure_ascii=False, indent=4)

    logging.info("Data processing complete. Results saved to output files.")

# 使用示例
if __name__ == "__main__":
    input_file = "merged_dataset.json"
    text_output_file = "text_content.json"
    infobox_output_file = "infobox_content.json"
    tables_output_file = "tables_content.json"
    html_directory = "wikipedia_html"

    process_dataset(input_file, text_output_file, infobox_output_file, tables_output_file, html_directory)
