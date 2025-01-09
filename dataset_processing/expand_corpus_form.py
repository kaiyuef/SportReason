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

def process_html_folder(html_dir, 
                        text_output_file, 
                        infobox_output_file, 
                        tables_output_file):
    """
    遍历 HTML 文件夹中所有 HTML 文件，提取 text_data, infobox_data, tables_data 并保存到 JSONL 文件。
    """
    text_id_count = 0
    infobox_id_count = 0
    table_id_count = 0

    with open(text_output_file, 'w', encoding='utf-8') as text_out, \
         open(infobox_output_file, 'w', encoding='utf-8') as infobox_out, \
         open(tables_output_file, 'w', encoding='utf-8') as tables_out:

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
            text_out.write(json.dumps(text_data, ensure_ascii=False) + '\n')
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
                infobox_out.write(json.dumps(infobox_data, ensure_ascii=False) + '\n')
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
                    tables_out.write(json.dumps(table_item, ensure_ascii=False) + '\n')
                    table_id_count += 1

# 主函数
if __name__ == "__main__":
    html_directory = "wikipedia_expanded_html"
    text_output_file = "expanded_text_content.jsonl"
    infobox_output_file = "expanded_infobox_content.jsonl"
    tables_output_file = "expanded_tables_content.jsonl"

    process_html_folder(
        html_dir=html_directory,
        text_output_file=text_output_file,
        infobox_output_file=infobox_output_file,
        tables_output_file=tables_output_file
    )
