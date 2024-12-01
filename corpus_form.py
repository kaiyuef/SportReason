import json
import logging
import os
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

# 配置日志记录
logging.basicConfig(filename='process_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 函数清单：包括sanitize_filename, extract_infobox, calculate_jaccard_similarity, is_content_similar, flatten_content, clean_content

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
            header_text = header.text
            data_text = ' | '.join([line for line in data.stripped_strings])
            if header_text:
                if current_section is None:
                    current_section = header_text
                    infobox_data[current_section] = {header_text: data_text}
                else:
                    infobox_data[current_section][header_text] = data_text
        elif header:
            current_section = header.text
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

def is_content_similar(gold_content, corpus_content, content_type="text", dataset_id=None, gold_id=None):
    gold_text = flatten_content(gold_content, content_type)
    if not gold_text:
        return False

    if content_type == "text":
        corpus_text = flatten_content(corpus_content, content_type)
        if isinstance(corpus_text, list):
            corpus_text = " ".join(corpus_text)
        if isinstance(gold_text, list):
            gold_text = " ".join(gold_text)

        gold_text_cleaned = clean_content(gold_text)
        corpus_text_cleaned = clean_content(corpus_text)

        match_score = partial_ratio(gold_text_cleaned, corpus_text_cleaned)
        is_match = match_score >= 70
        return is_match
    else:
        gold_cleaned_text = gold_text
        corpus_cleaned_texts = flatten_content(corpus_content, content_type)

        for corpus_item in corpus_cleaned_texts:
            similarity_score = calculate_jaccard_similarity(gold_cleaned_text, corpus_item)
            if similarity_score >= 0.5:
                return True
        return False

# 清理和扁平化内容
def clean_content(content):
    """
    Clean and normalize content by removing punctuation and extra whitespace.
    """
    # Remove punctuation
    content = re.sub(r'[^\w\s]', '', content)
    # Normalize whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    return content.lower()
def clean_text_content(content):
    """
    Clean and prepare text content for matching.
    """
    if isinstance(content, dict):
        text = content.get("text", "")
        return clean_content(text) if isinstance(text, str) else ""
    elif isinstance(content, list):
        return clean_content(" ".join(content))
    elif isinstance(content, str):
        return clean_content(content)
    return ""

def clean_table_content(content):
    """
    Extract all text from table content, clean it, and return as a single string.
    """
    text_elements = extract_text_from_dict(content)
    combined_text = " ".join(text_elements)
    return clean_content(combined_text)

def clean_infobox_content(content):
    """
    Extract all text from infobox content, clean it, and return as a single string.
    """
    text_elements = extract_text_from_dict(content)
    combined_text = " ".join(text_elements)
    return clean_content(combined_text)
def flatten_content(content, content_type):
    """
    Flatten and clean content based on its type.
    """
    if content_type == "text":
        return clean_text_content(content)
    elif content_type == "table":
        if isinstance(content, list):
            # Return as a list of cleaned strings
            return [clean_table_content(item) for item in content]
        else:
            return [clean_table_content(content)]  # Ensure consistent list structure
    elif content_type == "infobox":
        if isinstance(content, list):
            return [clean_infobox_content(item) for item in content]
        else:
            return [clean_infobox_content(content)]
    return ""


def get_wikipedia_content_from_file(title, url, html_dir):
    valid_filename = sanitize_filename(title).replace(' ', '_') + '.html'
    file_path = os.path.join(html_dir, valid_filename)

    if not os.path.exists(file_path):
        logging.error(f"HTML file for {title} not found at {file_path}")
        return None, None, None

    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    content_div = soup.find('div', class_='mw-parser-output')

    text_content = []
    skip_sections = {"References", "External links", "See also", "Further reading", "Notes"}

    for element in content_div.find_all(['p', 'ul', 'ol', 'h2', 'h3', 'h4']):
        if element.name in ['h2', 'h3', 'h4']:
            section_title = element.get_text()
            if section_title.strip() in skip_sections:
                break
            text_content.append(f"## {section_title}")
        elif element.name in ['p', 'ul', 'ol']:
            paragraph_text = element.get_text()
            if paragraph_text:
                text_content.append(paragraph_text)

    text_data = {"title": title, "url": url, "content": text_content, "used_in": []}
    infobox_data = extract_infobox(soup)
    infobox = {"title": title, "url": url, "content": infobox_data if infobox_data else {}, "used_in": []}

    tables = []
    for table in content_div.find_all('table', {'class': 'wikitable'}):
        headers = [header_cell.get_text() for header_cell in table.find_all('th')]
        rows = [[cell.get_text() for cell in row.find_all(['td', 'th'])] for row in table.find_all('tr')[1:]]
        tables.append({'columns': headers, 'rows': rows})

    tables_data = {"title": title, "url": url, "content": tables, "used_in": []}

    return text_data, infobox, tables_data

def process_dataset(input_file, text_output_file, infobox_output_file, tables_output_file, html_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    text_results = []
    infobox_results = []
    tables_results = []

    for idx, element in tqdm(enumerate(dataset), total=len(dataset), desc="Processing datasets"):
        for evidence in element['gold_evidences']:
            url = evidence.get('meta', {}).get('url', '')
            title = evidence.get('title', '')

            if not url or not title:
                continue

            text_data, infobox_data, tables_data = get_wikipedia_content_from_file(title, url, html_dir)

            for gold_content, corpus_content in [(evidence, text_data), (evidence, tables_data)]:
                match = is_content_similar(gold_content, corpus_content)
                # 更新used_in字段
                if match:
                    match_status = "True"
                else:
                    match_status = "False"

                if corpus_content:
                    corpus_content["used_in"].append({
                        "dataset": element.get("seed_dataset", "unknown"),
                        "id": evidence.get("id", "unknown"),
                        "match": match_status
                    })

            if text_data and text_data not in text_results:
                text_results.append(text_data)
            if infobox_data and infobox_data not in infobox_results:
                infobox_results.append(infobox_data)
            if tables_data and tables_data not in tables_results:
                tables_results.append(tables_data)

    # 写入输出文件
    with open(text_output_file, 'w', encoding='utf-8') as f:
        json.dump(text_results, f, ensure_ascii=False, indent=4)

    with open(infobox_output_file, 'w', encoding='utf-8') as f:
        json.dump(infobox_results, f, ensure_ascii=False, indent=4)

    with open(tables_output_file, 'w', encoding='utf-8') as f:
        json.dump(tables_results, f, ensure_ascii=False, indent=4)

    logging.info("Data processing complete. Results saved to output files.")

# 使用示例
input_file = "merged_dataset.json"
text_output_file = "text_content.json"
infobox_output_file = "infobox_content.json"
tables_output_file = "tables_content.json"
html_directory = "wikipedia_html"

process_dataset(input_file, text_output_file, infobox_output_file, tables_output_file, html_directory)
