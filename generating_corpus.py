import re
import json
import logging
import os
from bs4 import BeautifulSoup
from tqdm import tqdm
from nltk import download
from rapidfuzz.fuzz import partial_ratio

# Ensure necessary NLTK resources are downloaded
download('punkt')

# Configure logging
logging.basicConfig(
    filename="process_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def sanitize_filename(filename):
    invalid_chars = r'[<>:"/\\|?*\n\t]'
    return re.sub(invalid_chars, '_', filename).strip()


def extract_infobox(soup):
    infobox = soup.find('table', {'class': 'infobox'})
    if not infobox:
        return None
    rows = infobox.find_all('tr')
    infobox_data = {}
    current_section = None
    for row in rows:
        header = row.find('th')
        data = row.find('td')
        if header and data:
            header_text = header.text
            data_text = ' | '.join(data.stripped_strings)
            if current_section is None:
                current_section = header_text
                infobox_data[current_section] = {header_text: data_text}
            else:
                infobox_data[current_section][header_text] = data_text
        elif header:
            current_section = header.text
            infobox_data[current_section] = {}
    return infobox_data


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
    infobox = {"title": title, "url": url, "content": infobox_data or {}, "used_in": []}
    tables = []
    for table in content_div.find_all('table', {'class': 'wikitable'}):
        headers = [cell.get_text() for cell in table.find_all('th')]
        rows = [[cell.get_text() for cell in row.find_all(['td', 'th'])] for row in table.find_all('tr')[1:]]
        tables.append({'columns': headers, 'rows': rows})
    tables_data = {"title": title, "url": url, "content": tables, "used_in": []}
    return text_data, infobox, tables_data


def process_dataset(input_file, text_output_file, infobox_output_file, tables_output_file, html_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    processed_titles = set()
    text_results = []
    infobox_results = []
    tables_results = []

    text_counter = 1
    infobox_counter = 1
    table_counter = 1

    for idx, element in tqdm(enumerate(dataset), total=len(dataset), desc="Processing datasets"):
        if element.get('seed_dataset') == 'temptableqa':
            continue

        for evidence in element['gold_evidences']:
            url = evidence.get('meta', {}).get('url', '')
            title = evidence.get('title', '')
            if not url or not title or title in processed_titles:
                continue

            try:
                text_data, infobox_data, tables_data = get_wikipedia_content_from_file(title, url, html_dir)
                if text_data:
                    text_data["id"] = f"text_{text_counter}"
                    text_counter += 1
                    text_results.append(text_data)
                if infobox_data:
                    infobox_data["id"] = f"infobox_{infobox_counter}"
                    infobox_counter += 1
                    infobox_results.append(infobox_data)
                if tables_data:
                    for table in tables_data["content"]:
                        table["id"] = f"table_{table_counter}"
                        table_counter += 1
                    tables_results.append(tables_data)
                processed_titles.add(title)
            except Exception as e:
                logging.error(f"Error processing title '{title}': {e}")

    with open(text_output_file, 'w', encoding='utf-8') as f:
        json.dump(text_results, f, ensure_ascii=False, indent=4)
    with open(infobox_output_file, 'w', encoding='utf-8') as f:
        json.dump(infobox_results, f, ensure_ascii=False, indent=4)
    with open(tables_output_file, 'w', encoding='utf-8') as f:
        json.dump(tables_results, f, ensure_ascii=False, indent=4)


# Integration with gold evidence matching
def update_corpus_with_gold_evidence(dataset, text_corpus, table_corpus, infobox_corpus):
    text_corpus_dict = {page["title"]: page for page in text_corpus}
    table_corpus_dict = {page["title"]: page for page in table_corpus}
    infobox_corpus_dict = {page.get("title", f"Untitled-{idx}"): page for idx, page in enumerate(infobox_corpus)}

    for data in tqdm(dataset, desc="Updating corpora"):
        dataset_id = data["id"]
        dataset_name = data.get("seed_dataset", "Unknown")
        for gold in data["gold_evidences"]:
            gold_id = gold.get("id", "")
            gold_content = gold.get("content", {})
            gold_type = gold_id.split('_')[0]

            if gold_type == "text":
                corpus = text_corpus
                corpus_dict = text_corpus_dict
            elif gold_type == "table":
                corpus = table_corpus
                corpus_dict = table_corpus_dict
            elif gold_type == "infobox":
                corpus = infobox_corpus
                corpus_dict = infobox_corpus_dict
            else:
                continue

            title = gold.get("title", f"Untitled-{gold_id}")
            if title in corpus_dict:
                corpus_page = corpus_dict[title]
                existing_content = corpus_page["content"]
                # Check for matching and update
                # Add further processing here as per requirements

    return text_corpus, table_corpus, infobox_corpus


# Example usage
html_dir = "wikipedia_html"
process_dataset("merged_dataset.json", "text_content.json", "infobox_content.json", "tables_content.json", html_dir)
