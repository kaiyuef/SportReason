import os
import json
from bs4 import BeautifulSoup
from tqdm import tqdm

# 处理HTML文件夹中的文件
def sanitize_filename(filename):
    """
    去除文件名中的非法字符。
    """
    invalid_chars = r'[<>:"/\\|?*\n\t]'
    sanitized = re.sub(invalid_chars, '_', filename)
    return sanitized.strip()

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
            header_text = str(header.text.strip())
            data_text = str(' | '.join([line for line in data.stripped_strings]))
            if header_text:
                if current_section is None:
                    current_section = header_text
                    infobox_data[current_section] = {header_text: data_text}
                else:
                    infobox_data[current_section][header_text] = data_text
        elif header:
            current_section = str(header.text.strip())
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
    text_data = {
        "title": title,
        "url": url,
        "contents": text_string
    }

    # 2) infobox
    infobox_dict = extract_infobox(soup)
    infobox_data = {
        "title": title,
        "url": url,
        "contents": infobox_dict if infobox_dict else {}
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
            tables_data_list.append(table_item)

    return text_data, infobox_data, tables_data_list

        
def process_html_folder(html_dir, 
                        text_output_file, 
                        infobox_output_file, 
                        tables_output_file):
    """
    遍历 HTML 文件夹中所有 HTML 文件，提取 text_data, infobox_data, tables_data 并保存到 JSON 文件。
    """
    text_results = []
    infobox_results = []
    tables_results = []

    text_id_count = 0
    infobox_id_count = 0
    table_id_count = 0

    for file_name in tqdm(os.listdir(html_dir), desc="Processing HTML files"):
        if not file_name.endswith('.html'):
            continue

        file_path = os.path.join(html_dir, file_name)
        title = str(file_name.replace('_', ' ').replace('.html', ''))
        url = str(f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")

        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # 提取正文
        text_string = extract_text_from_html(soup)
        text_data = {
            "title": title,
            "url": url,
            "contents": str(text_string),
            "id": f"expanded_text_{text_id_count}"
        }
        text_results.append(text_data)
        text_id_count += 1

        # 提取 infobox
        infobox_dict = extract_infobox(soup)
        if infobox_dict:
            infobox_data = {
                "title": title,
                "url": url,
                "contents": str({key: str(value) for key, value in infobox_dict.items()}),
                "id": f"expanded_infobox_{infobox_id_count}"
            }
            infobox_results.append(infobox_data)
            infobox_id_count += 1
            

        # 提取表格
        content_div = soup.find('div', class_='mw-parser-output')
        if content_div:
            tables = content_div.find_all('table', {'class': 'wikitable'})
            for idx, table in enumerate(tables):
                headers = [
                    str(header_cell.get_text(separator=" ", strip=True))
                    for header_cell in table.find_all('th')
                ]
                rows = []
                table_rows = table.find_all('tr')
                for row in table_rows[1:]:
                    row_cells = [
                        str(cell.get_text(separator=" ", strip=True))
                        for cell in row.find_all(['td', 'th'])
                    ]
                    rows.append(row_cells)

                table_item = {
                    "title": title,
                    "url": url,
                    "contents": str({
                        "columns": headers,
                        "rows": rows
                    }),
                    "id": f"expanded_table_{table_id_count}"
                }
                tables_results.append(table_item)
                table_id_count += 1

    # 写入 JSON 文件
    with open(text_output_file, 'w', encoding='utf-8') as f:
        json.dump(text_results, f, ensure_ascii=False, indent=4)

    with open(infobox_output_file, 'w', encoding='utf-8') as f:
        json.dump(infobox_results, f, ensure_ascii=False, indent=4)

    with open(tables_output_file, 'w', encoding='utf-8') as f:
        json.dump(tables_results, f, ensure_ascii=False, indent=4)

# 主函数
if __name__ == "__main__":
    html_directory = "wikipedia_expanded_html"
    text_output_file = "expanded_text_content.json"
    infobox_output_file = "expanded_infobox_content.json"
    tables_output_file = "expanded_tables_content.json"

    process_html_folder(
        html_dir=html_directory,
        text_output_file=text_output_file,
        infobox_output_file=infobox_output_file,
        tables_output_file=tables_output_file
    )
