import json
import os
import re
from bs4 import BeautifulSoup
from tqdm import tqdm

def sanitize_filename(filename):
    """
    去除文件名中的非法字符。
    """
    invalid_chars = r'[<>:"/\\|?*\n\t]'
    sanitized = re.sub(invalid_chars, '_', filename)
    return sanitized.strip()

def sanitize_content_string(content_str):
    """
    清洗字符串中的多余空白和换行符，将其统一替换为一个空格，并去除首尾空格。
    """
    if not isinstance(content_str, str):
        return content_str
    # 用正则去掉所有连续的空白字符（包括 \n \r \t 等），并替换为单个空格
    return re.sub(r'\s+', ' ', content_str).strip()

def sanitize_infobox_dict(info_dict):
    """
    递归清洗 infobox 字典中的字符串内容。
    """
    if not isinstance(info_dict, dict):
        return info_dict
    
    cleaned_dict = {}
    for key, value in info_dict.items():
        if isinstance(value, dict):
            # 如果 value 还是一个字典，则递归处理
            cleaned_dict[key] = sanitize_infobox_dict(value)
        elif isinstance(value, str):
            # 如果 value 是字符串，调用字符串清洗
            cleaned_dict[key] = sanitize_content_string(value)
        else:
            cleaned_dict[key] = value
    return cleaned_dict

def sanitize_table_item(tbl_item):
    """
    对表格的 columns 和 rows 里所有字符串内容进行清洗。
    """
    if not tbl_item or not isinstance(tbl_item, dict):
        return tbl_item
    
    contents = tbl_item.get("contents", {})
    if not isinstance(contents, dict):
        return tbl_item
    
    # 清洗 columns
    columns = contents.get("columns", [])
    columns_cleaned = [sanitize_content_string(col) for col in columns]
    
    # 清洗 rows
    rows = contents.get("rows", [])
    rows_cleaned = []
    for row in rows:
        cleaned_row = [sanitize_content_string(cell) for cell in row]
        rows_cleaned.append(cleaned_row)
    
    tbl_item["contents"]["columns"] = columns_cleaned
    tbl_item["contents"]["rows"] = rows_cleaned
    return tbl_item

def extract_infobox(soup):
    """
    从 HTML 中查找 class=infobox 的表格，并解析其中的内容。
    返回一个 dict，每个 section 内部是 {表头: 数据} 的形式。
    """
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

def extract_text_from_html(soup):
    """
    从 HTML 中提取正文文本：考虑 h2/h3/h4/p/ul/ol 等标签。
    可以根据需要添加或删除跳过的 section。
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
    给定条目标题和对应 HTML 目录，从文件中读取 HTML 并提取：
      1) 正文文本 text_data
      2) infobox_data
      3) tables_data_list（所有 wikitable）
    """
    valid_filename = sanitize_filename(title).replace(' ', '_') + '.html'
    file_path = os.path.join(html_dir, valid_filename)

    if not os.path.exists(file_path):
        return None, None, []

    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1) 正文
    text_string = extract_text_from_html(soup)
    # 在这里对提取到的正文进行清洗
    text_string = sanitize_content_string(text_string)
    text_data = {
        "title": title,
        "url": url,
        "contents": text_string
    }

    # 2) infobox
    infobox_dict = extract_infobox(soup)
    if infobox_dict is None:
        infobox_dict = {}
    # 对 infobox 中的值进行清洗
    infobox_dict = sanitize_infobox_dict(infobox_dict)
    infobox_data = {
        "title": title,
        "url": url,
        "contents": infobox_dict
    }

    # 3) tables
    tables_data_list = []
    content_div = soup.find('div', class_='mw-parser-output')
    if content_div:
        tables = content_div.find_all('table', {'class': 'wikitable'})
        for idx, table in enumerate(tables):
            headers = [
                header_cell.get_text(separator=" ", strip=True) 
                for header_cell in table.find_all('th')
            ]
            rows = []
            table_rows = table.find_all('tr')
            # 从第二行开始解析
            for row in table_rows[1:]:
                row_cells = [
                    cell.get_text(separator=" ", strip=True)
                    for cell in row.find_all(['td', 'th'])
                ]
                rows.append(row_cells)

            table_item = {
                "title": title,
                "url": url,
                "contents": {
                    "columns": headers,
                    "rows": rows
                }
            }
            # 表格内容清洗
            table_item = sanitize_table_item(table_item)
            tables_data_list.append(table_item)

    return text_data, infobox_data, tables_data_list

def process_dataset(input_file, 
                    text_output_file, 
                    infobox_output_file, 
                    tables_output_file, 
                    html_dir):
    """
    读取 input_file 的数据集，依次根据 gold_evidences 的标题，从本地的 HTML 文件中
    提取正文、infobox、table，然后写入相应的输出文件。

    为每个提取的 element（text/infobox/table）分配一个独立的 id。
    并使用cache，避免重复读取同一个HTML文件。
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    text_results = []
    infobox_results = []
    tables_results = []

    text_id_count = 0
    infobox_id_count = 0
    table_id_count = 0

    # 用于缓存已读取过的 title -> (text_data, infobox_data, tables_data_list)
    cache_dict = {}

    for element in tqdm(dataset, desc="Processing datasets"):
        for evidence in element.get('gold_evidences', []):
            url = evidence.get('meta', {}).get('url', '')
            title = evidence.get('title', '')

            if not url or not title:
                continue

            # 判断是否已在缓存中
            if title in cache_dict:
                text_data, infobox_data, tables_data_list = cache_dict[title]
            else:
                # 若不在缓存中，则读取文件并存入缓存
                text_data, infobox_data, tables_data_list = get_wikipedia_content_from_file(
                    title, url, html_dir
                )
                cache_dict[title] = (text_data, infobox_data, tables_data_list)

            if text_data is None:
                # 文件不存在或读取失败
                continue

            # 1) 给 text_data 分配 ID 并去重
            if text_data not in text_results:
                text_data["id"] = f"text_{text_id_count}"
                text_id_count += 1
                text_results.append(text_data)

            # 2) 给 infobox_data 分配 ID 并去重
            if infobox_data not in infobox_results:
                infobox_data["id"] = f"infobox_{infobox_id_count}"
                infobox_id_count += 1
                infobox_results.append(infobox_data)

            # 3) 给 tables_data_list 里的每个表分配 ID，并去重
            for tbl in tables_data_list:
                if tbl not in tables_results:
                    tbl["id"] = f"table_{table_id_count}"
                    table_id_count += 1
                    tables_results.append(tbl)

    # 写入文件
    with open(text_output_file, 'w', encoding='utf-8') as f:
        json.dump(text_results, f, ensure_ascii=False, indent=4)

    with open(infobox_output_file, 'w', encoding='utf-8') as f:
        json.dump(infobox_results, f, ensure_ascii=False, indent=4)

    with open(tables_output_file, 'w', encoding='utf-8') as f:
        json.dump(tables_results, f, ensure_ascii=False, indent=4)

# 示例主函数
if __name__ == "__main__":
    input_file = "merged_dataset.json"
    text_output_file = "text_content.json"
    infobox_output_file = "infobox_content.json"
    tables_output_file = "tables_content.json"
    html_directory = "wikipedia_html"

    process_dataset(
        input_file, 
        text_output_file, 
        infobox_output_file, 
        tables_output_file, 
        html_directory
    )
