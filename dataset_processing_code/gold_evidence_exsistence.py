import re
import json
from tqdm import tqdm
import logging
import nltk

from rapidfuzz.fuzz import partial_ratio
from itertools import count
# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')


# Configure logging
logging.basicConfig(
    filename="gold_evidence_comparisons.log",
    level=logging.INFO,  # Set to INFO level to capture the comparison logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
def calculate_jaccard_similarity(text1, text2):
    """
    Calculate Jaccard similarity between two strings or lists of strings.
    If input is a list, join elements into a single string.
    """
    if isinstance(text1, list):
        text1 = " ".join(text1)
    if isinstance(text2, list):
        text2 = " ".join(text2)
    
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:  # Avoid division by zero
        return 0.0
    return len(intersection) / len(union)


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
    # Remove punctuation
    content = re.sub(r'[^\w\s]', '', content)
    # Normalize whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    return content.lower()

def extract_text_from_dict(data):
    """
    Recursively extract all keys and values from a dictionary or list
    and return them as a concatenated string.
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


def is_content_similar(gold_content, corpus_content, content_type="text", dataset_id=None, gold_id=None):
    """
    Determine if gold_content exists within corpus_content using fuzzy matching.
    Enhanced to include dataset_id and gold_id in logs for better traceability.
    """
    gold_text = flatten_content(gold_content, content_type)
    global table_length
    global text_length
    global infobox_length
    print(text_length, infobox_length, table_length)

    if not gold_text:
        return False

    if content_type == "text":
        # Clean and flatten corpus content
        try:
            corpus_id = corpus_content['id']
        except:
            text_length+=1
            return f'text_{text_length}', False
        corpus_content = corpus_content['content']
        corpus_text = flatten_content(corpus_content, content_type)
        
        
        # Handle cases where content is a list
        if isinstance(corpus_text, list):
            corpus_text = " ".join(corpus_text)  # Combine into single string
        if isinstance(gold_text, list):
            gold_text = " ".join(gold_text)      # Combine into single string

        # Clean both texts
        gold_text_cleaned = clean_content(gold_text)
        corpus_text_cleaned = clean_content(corpus_text)

        # Perform fuzzy matching
        match_score = partial_ratio(gold_text_cleaned, corpus_text_cleaned)
        is_match = match_score >= 70  # Set a threshold for similarity
        if is_match == False:
            logging.info(
                f"Dataset ID: {dataset_id}, Gold ID: {gold_id}\n"
                f"Fuzzy matching gold text against corpus text:\n"
                f"Gold: {gold_text_cleaned}\nCorpus: {corpus_text_cleaned}\n"
                f"Score: {match_score}\nResult: {is_match}"
            )
        return corpus_id, is_match
    elif content_type == "table":
        # Handle table logic as before
        gold_cleaned_text = gold_text
        corpus_cleaned_texts = flatten_content(corpus_content['content'], content_type)

        acc = 0
        for corpus_item in corpus_cleaned_texts:
            
            similarity_score = calculate_jaccard_similarity(gold_cleaned_text, corpus_item)
            try:
                corpus_id = corpus_content['content'][acc]['id']
            except:
                table_length+=1
                return f'table_{table_length}', False
            acc+=1
            if similarity_score <= 0.5:
                logging.info(
                    f"Dataset ID: {dataset_id}, Gold ID: {gold_id}\n"
                    f"Comparing gold {content_type} to corpus {content_type}:\n"
                    f"Gold: {gold_cleaned_text}\nCorpus: {corpus_item}\n"
                    f"Score: {similarity_score:.4f}"
                )
            if similarity_score >= 0.5:
                return corpus_id, True
        logging.info("Match result: False")
        
        table_length+=1
        return f'table_{table_length}', False
    else:
        # Handle other types (e.g., infoboxes)
        gold_cleaned_text = gold_text
        try:
            corpus_id = corpus_content['id']
        except:
            infobox_length+=1
            return f'infobox_{infobox_length}', False
        corpus_content = corpus_content['content']
        corpus_cleaned_texts = flatten_content(corpus_content, content_type)
        

        for corpus_item in corpus_cleaned_texts:
            similarity_score = calculate_jaccard_similarity(gold_cleaned_text, corpus_item)
            if similarity_score <= 0.5:
                logging.info(
                    f"Dataset ID: {dataset_id}, Gold ID: {gold_id}\n"
                    f"Comparing gold {content_type} to corpus {content_type}:\n"
                    f"Gold: {gold_cleaned_text}\nCorpus: {corpus_item}\n"
                    f"Score: {similarity_score:.4f}"
                )
            if similarity_score >= 0.5:
                return corpus_id, True
        logging.info("Match result: False")
        infobox_length+=1
        return f'infobox_{infobox_length}', False



def update_corpus_with_gold_evidence(dataset, text_corpus, table_corpus, infobox_corpus):
    """
    Update the text, table, and infobox corpora with gold evidence from the dataset.
    """
    #update the dataset_length
    global text_length
    global infobox_length
    global table_length
    
    text_length = len(text_corpus)
    infobox_length = len(infobox_corpus)
    for element in table_corpus:
        table_length += len(element['content'])
    
    
    # Create dictionaries for quick lookup
    text_corpus_dict = {page["title"]: page for page in text_corpus}
    table_corpus_dict = {page["title"]: page for page in table_corpus}
    # For infoboxes, some entries may not have titles
    infobox_corpus_dict = {page.get("title", f"Untitled-{idx}"): page for idx, page in enumerate(infobox_corpus)}

    for data in tqdm(dataset, desc="Processing dataset entries"):
        dataset_id = data["id"]  # Track dataset ID
        dataset_name = data.get("seed_dataset", "Unknown")  # Get dataset name dynamically

        for gold in tqdm(data["gold_evidences"], desc=f"Processing gold evidences for ID {dataset_id}", leave=False):
            gold_id = gold.get("id", "")
            gold_content = gold.get("content", {})
            gold_type = get_gold_type(gold_id)

            # Determine the target corpus and content type
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
                continue  # Skip unknown types

            # Use the title to identify entries in the corpus
            title = gold.get("title", f"Untitled-{gold_id}")

            # Handle infoboxes without titles (e.g., from 'temptableqa')
            if gold_type == "infobox" and (dataset_name == 'temptableqa' or not title):
                title = f"Untitled-{gold_id}"

            if title in corpus_dict:
                # Entry exists in the corpus
                corpus_page = corpus_dict[title]
                existing_content = corpus_page

                # Check if the content matches
                corpus_id, match = is_content_similar(gold_content, existing_content, content_type, dataset_id, gold_id)
                existing_content = corpus_page['content']

                # Convert 'match' to string 'True' or 'False'
                match_str = "True" if match else "False"

                # Update 'used_in'
                if "used_in" not in corpus_page:
                    corpus_page["used_in"] = []
                used_in_entry = {"dataset": dataset_name, "id": corpus_id, "match": match_str}
                if used_in_entry not in corpus_page["used_in"]:
                    corpus_page["used_in"].append(used_in_entry)

                if not match:
                    # Merge the contents appropriately
                    if content_type == "text":
                        # Append new text if not already present
                        text = gold_content.get("text", "")
                        existing_content.append(text)
                    elif content_type == "table":
                        # Append new table if not already present
                        existing_content.append(gold_content)
                    elif content_type == "infobox":
                        # Append new infobox if not already present
                        existing_content = gold_content
            else:
                # Entry does not exist in the corpus
                if content_type == "text":
                    vari = text_length
                    text_length += 1
                    new_content = []
                    text = gold_content.get("text", "")
                    if isinstance(text, str) and text:
                        new_content.append(text)
                elif content_type == "table":
                    vari = table_length
                    table_length += 1
                    new_content = [gold_content] if gold_content else []
                elif content_type == "infobox":
                    vari = infobox_length
                    infobox_length += 1
                    new_content = gold_content if gold_content else {}

                new_entry = {
                    "title": title if title else None,
                    "url": gold.get("meta", {}).get("url", ""),
                    "content": new_content,
                    "used_in": [{
                        "dataset": dataset_name,
                        "id": f'{content_type}_{vari}',
                        "match": "Not found"  # Since the entry didn't exist
                    }],
                    "added_by": "gold_evidence_script",
                }
                corpus.append(new_entry)
                corpus_dict[title] = new_entry

    return text_corpus, table_corpus, infobox_corpus

if __name__ == "__main__":
    # Load data
    with open('merged_dataset.json', 'r') as f:
        dataset = json.load(f)

    with open('text_content.json', 'r') as f:
        text_corpus = json.load(f)

    with open('tables_content.json', 'r') as f:
        table_corpus = json.load(f)

    with open('infobox_content.json', 'r') as f:
        infobox_corpus = json.load(f)

    # Update corpora
    text_length =0
    infobox_length=0
    table_length=0

    text_corpus, table_corpus, infobox_corpus = update_corpus_with_gold_evidence(
        dataset, text_corpus, table_corpus, infobox_corpus
    )

    # Save updated corpora back to JSON files
    with open('text_content_updated.json', 'w') as f:
        json.dump(text_corpus, f, ensure_ascii=False, indent=4)

    with open('tables_content_updated.json', 'w') as f:
        json.dump(table_corpus, f, ensure_ascii=False, indent=4)

    with open('infobox_content_updated.json', 'w') as f:
        json.dump(infobox_corpus, f, ensure_ascii=False, indent=4)

    print("Corpus update completed.")

