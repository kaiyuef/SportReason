import json
import requests
from bs4 import BeautifulSoup
import re

def clean_text(text):
    text = (text
            .replace('\u00a0', ' ')
            .replace('\u2013', '-')
            .replace('\u2014', '-')
            .replace('\u200b', ''))
    text = re.sub(r'\s*\|\s*', ' ', text)
    text = re.sub(r'\(\s*\|\s*', '(', text)
    text = re.sub(r'\|\s*\)', ')', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_infobox(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve page. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    infobox = soup.find('table', {'class': 'infobox'})
    if not infobox:
        print("No infobox found on this page.")
        return None

    infobox_data = {}
    rows = infobox.find_all('tr')
    current_section = None

    for row in rows:
        header = row.find('th')
        data = row.find('td')
        
        if header and data:
            header_text = clean_text(header.text.strip())
            data_text = clean_text(' | '.join([line.strip() for line in data.stripped_strings]))
            if current_section is None:
                current_section = header_text
                infobox_data[current_section] = {header_text: data_text}
            else:
                infobox_data[current_section][header_text] = data_text
        elif header: 
            current_section = clean_text(header.text.strip())
            infobox_data[current_section] = {}

    return infobox_data

def save_updated_infobox_to_new_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as f:
        data = json.load(f)

    updated_data = []
    for item in data:
        new_item = item.copy()
        for evidence in new_item['gold_evidences']:
            if 'infobox' in evidence['id']:
                url = evidence['meta']['url']
                infobox_data = extract_infobox(url)
                if infobox_data:
                    evidence['content'] = infobox_data
        updated_data.append(new_item)

    with open(output_file_path, 'w') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=4)

# Specify the input and output file paths
input_file = 'processed/tanq/pretty_tanq.json'
output_file = 'processed/tanq/updated_pretty_tanq.json'
save_updated_infobox_to_new_file(input_file, output_file)
