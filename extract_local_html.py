import json
import logging
import os
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import html2text

# 配置日志记录
logging.basicConfig(filename='process_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def sanitize_filename(filename):
    invalid_chars = r'[<>:"/\\|?*\n\t]'
    sanitized = re.sub(invalid_chars, '_', filename)
    sanitized = sanitized.strip()
    return sanitized

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


def extract_tables(content_div, title):
    """
    Extract tables content using BeautifulSoup and attach a unique ID for each table.

    Parameters:
        content_div (Tag): The main content division of the page.
        title (str): Title of the Wikipedia page.

    Returns:
        list: List of dictionaries representing tables with headers, rows, and used_in IDs.
    """
    tables = []
    table_count = 1  # To generate unique IDs for each table
    for table in content_div.find_all('table', {'class': 'wikitable'}):
        headers = []
        rows = []
        header_rows = table.find_all('tr')

        # Extract headers
        for header_cell in header_rows[0].find_all(['th', 'td']):
            headers.append(header_cell.get_text(strip=True))

        # Extract data rows
        for row in header_rows[1:]:
            cells = []
            for cell in row.find_all(['td', 'th']):
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                cell_text = cell.get_text(strip=True)
                cells.extend([cell_text] * colspan)
            rows.append(cells)

        # Generate a unique ID for this table
        table_id = f"{title.replace(' ', '_')}_table_{table_count}"
        table_count += 1

        tables.append({
            'columns': headers,
            'rows': rows,
            'used_in': [table_id]  # Attach the unique ID for this table
        })
    return tables

#def get_wikipedia_content_from_file2(title, url, html_dir):
    """
    Extract content from a local Wikipedia HTML file.

    Parameters:
        title (str): Title of the Wikipedia page.
        url (str): URL of the Wikipedia page.
        html_dir (str): Directory containing the HTML files.

    Returns:
        tuple: (text_data, infobox_data, tables_data)
    """
    # Sanitize and construct the file path
    valid_filename = title.replace(' ', '_') + '.html'
    file_path = os.path.join(html_dir, valid_filename)

    if not os.path.exists(file_path):
        logging.error(f"HTML file for {title} not found at {file_path}")
        return None, None, None

    # Read the HTML content
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Debug: Log all div elements for inspection
    all_divs = [div.get('class') for div in soup.find_all('div')]
    logging.debug(f"All div classes: {all_divs}")

    # Locate the main content area
    content_div = soup.find('div', class_='mw-parser-output')
    if not content_div:
        logging.warning(f"'mw-parser-output' not found for {title}. Attempting fallback selectors.")
        content_div = soup.find('main', {'id': 'content'}) or soup.find('div', {'id': 'bodyContent'})
        if not content_div:
            logging.error(f"No suitable content section found for {title}")
            return None, None, None

    # Debug: Log content-div children for inspection
    logging.debug(f"Content div children tags: {[child.name for child in content_div.children]}")

    # Extract main text content
    text_content = []
    skip_sections = {"References", "External links", "See also", "Further reading", "Notes"}

    for element in content_div.find_all(['p', 'ul', 'ol', 'h2', 'h3', 'h4']):
        if element.name in ['h2', 'h3', 'h4']:
            section_title = element.get_text()
            if section_title.strip() in skip_sections:
                logging.info(f"Skipping section: {section_title}")
                break
            text_content.append(f"## {section_title}")
        elif element.name in ['p', 'ul', 'ol']:
            paragraph_text = element.get_text(strip=True)
            if paragraph_text:
                text_content.append(paragraph_text)

    # Debug: Log extracted text content
    logging.debug(f"Extracted text content: {text_content}")

    text_data = {
        "title": title,
        "url": url,
        "content": text_content,
        "used_in": []
    }

    # Extract infobox data
    infobox_div = soup.find('table', class_='infobox')
    infobox_data = {}
    if infobox_div:
        for row in infobox_div.find_all('tr'):
            header = row.find('th')
            value = row.find('td')
            if header and value:
                infobox_data[header.get_text(strip=True)] = value.get_text(strip=True)
    else:
        logging.info(f"No infobox found for {title}")

    # Debug: Log extracted infobox data
    logging.debug(f"Extracted infobox data: {infobox_data}")

    infobox = {
        "title": title,
        "url": url,
        "content": infobox_data,
        "used_in": []
    }

    # Extract tables data
    tables = []
    for table in content_div.find_all('table', class_='wikitable'):
        headers = [header.get_text(strip=True) for header in table.find_all('th')]
        rows = []
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
            rows.append(cells)

        tables.append({
            'columns': headers,
            'rows': rows
        })

    # Debug: Log extracted table data
    logging.debug(f"Extracted tables data: {tables}")

    tables_data = {
        "title": title,
        "url": url,
        "content": tables,
        "used_in": []
    }

    return text_data, infobox, tables_data

def get_wikipedia_content_from_file(title, url, html_dir):
    """
    Extracts content from a local Wikipedia HTML file using html2text and BeautifulSoup.

    Parameters:
        title (str): Title of the Wikipedia page.
        url (str): URL of the Wikipedia page.
        html_dir (str): Directory containing the HTML files.

    Returns:
        tuple: (text_data, infobox_data, tables_data)
    """
    # Sanitize and construct the file path
    valid_filename = title.replace(' ', '_') + '.html'
    file_path = os.path.join(html_dir, valid_filename)

    if not os.path.exists(file_path):
        logging.error(f"HTML file for {title} not found at {file_path}")
        return None, None, None

    # Read the HTML content
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Use BeautifulSoup for structured content extraction
    soup = BeautifulSoup(html_content, 'html.parser')
    content_div = soup.find('div', class_='mw-parser-output')

    if not content_div:
        logging.error("No content div found; unable to extract tables.")
        return None, None, None

    # Extract infobox content and remove it from the content
    infobox_data = extract_infobox(soup)
    if infobox_data:
        infobox_div = soup.find('table', class_='infobox')
        if infobox_div:
            infobox_div.decompose()  # Remove infobox from the content

    infobox = {
        "title": title,
        "url": url,
        "content": infobox_data if infobox_data else {},
        "used_in": []
    }

    # Extract tables data and remove them from the content
    tables = []
    for table in content_div.find_all('table', class_='wikitable'):
        headers = [header.get_text(strip=True) for header in table.find_all('th')]
        rows = []
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
            rows.append(cells)

        tables.append({
            'columns': headers,
            'rows': rows
        })

        # Remove the table from the content
        table.decompose()

    # Debug: Log extracted table data
    logging.debug(f"Extracted tables data: {tables}")

    tables_data = {
        "title": title,
        "url": url,
        "content": tables,
        "used_in": []
    }

    # Use html2text for main text extraction
    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.ignore_emphasis = False
    converter.body_width = 0  # No wrapping

    # Get the updated HTML content after removing infobox and tables
    updated_html_content = str(content_div)
    markdown_text = converter.handle(updated_html_content)

    # Process main text
    sections = markdown_text.split('\n\n')
    main_text = [section.strip() for section in sections if section.strip()]

    text_data = {
        "title": title,
        "url": url,
        "content": main_text,
        "used_in": []
    }

    return text_data, infobox, tables_data


def process_dataset(input_file, text_output_file, infobox_output_file, tables_output_file, html_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    processed_titles = set()
    text_results = []
    infobox_results = []
    tables_results = []
    
    # 初始化计数器
    text_counter = 1
    infobox_counter = 1
    table_counter = 1
    
    for idx, element in tqdm(enumerate(dataset[:]), total=len(dataset[:]), desc="Processing datasets"):
        if element.get('seed_dataset') == 'temptableqa':
            logging.info(f"Skipping seed_dataset: {element.get('seed_dataset')} for element {idx}")
            continue  # 跳过此元素

        for evidence in element['gold_evidences']:
            url = evidence.get('meta', {}).get('url', '')
            title = evidence.get('title', '')

            if not url or not title:
                continue

            # 使用标题来避免重复处理
            if title in processed_titles:
                continue

            try:
                text_data, infobox_data, tables_data = get_wikipedia_content_from_file(title, url, html_dir)
                
                if not text_data and not infobox_data and not tables_data:
                    logging.error(f"No content extracted for URL {url} with title '{title}'")
                    continue

                # 为文本数据添加 ID
                if text_data:
                    text_data["id"] = f"text_{text_counter}"
                    text_counter += 1
                    text_results.append(text_data)

                # 为每个信息框添加唯一 ID
                if infobox_data:
                    infobox_data["id"] = f"infobox_{infobox_counter}"
                    infobox_counter += 1
                    infobox_results.append(infobox_data)

                # 为每个表格添加唯一 ID
                if tables_data:
                    for table in tables_data["content"]:
                        table["id"] = f"table_{table_counter}"
                        table_counter += 1
                    tables_results.append(tables_data)

                processed_titles.add(title)
            except Exception as e:
                logging.error(f"Error processing title '{title}': {e}")

    # 写入输出文件
    with open(text_output_file, 'w', encoding='utf-8') as f:
        json.dump(text_results, f, ensure_ascii=False, indent=4)

    with open(infobox_output_file, 'w', encoding='utf-8') as f:
        json.dump(infobox_results, f, ensure_ascii=False, indent=4)

    with open(tables_output_file, 'w', encoding='utf-8') as f:
        json.dump(tables_results, f, ensure_ascii=False, indent=4)

    logging.info("Data processing complete. Results saved to output files.")


# 使用示例
input_file = "merged_dataset.json"  # 替换为你的输入数据集文件
text_output_file = "text_content.json"
infobox_output_file = "infobox_content.json"
tables_output_file = "tables_content.json"
html_directory = "wikipedia_html"  # 下载的HTML文件目录

process_dataset(input_file, text_output_file, infobox_output_file, tables_output_file, html_directory)
# print(get_wikipedia_content_from_file('2015_Formula_One_World_Championship', 'https://en.wikipedia.org/wiki/2015_Formula_One_World_Championship', 'wikipedia_html'))
