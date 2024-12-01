import json
from collections import OrderedDict
import os

# Function to read a JSON file (containing an array of JSON objects)
def read_json_file(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Function to read JSON content from a specified path
def read_json_content(path):
    if os.path.exists(path):
        with open(path, 'r') as file:
            return json.load(file)
    else:
        return {}
    
# Function to rearrange the format
def rearrange_data(data, request_path, table_path):
    rearranged_data = []
    id = 0
    sorted_data = sorted(data, key=lambda x: x["question_id"])
    
    global_request_counter = 1
    global_table_counter = 1

    for item in sorted_data:
        table_id = item.get("table_id", "")
        request_evidence_path = os.path.join(request_path, f"{table_id}.json")
        table_evidence_path = os.path.join(table_path, f"{table_id}.json")
        
        gold_evidence_request = read_json_content(request_evidence_path)
        gold_evidence_table = read_json_content(table_evidence_path)

        combined_formatted = []
        request_count = 0
        table_count = 0
        gold_evidence_ids = []
        
        # Process based on answer-node to extract specific sentences
        answer_nodes = item.get("answer-node", [])
        for node in answer_nodes:
            if node[3] != "passage":
                continue
            else:
                page_title = node[2]  # 提取标题
                
                evidence_id = f"text_{global_request_counter}"
                
                if page_title in gold_evidence_request:
                    # Directly get the full text content
                    full_text = {'text': gold_evidence_request[page_title]}
                    
                    # Construct the URL using the page_title
                    meta_url = f'https://en.wikipedia.org{page_title.replace(" ", "_")}'
                    
                    combined_formatted.append({
                        "id": evidence_id,
                        "title": page_title[6:],
                        "content": full_text,
                        "meta": {"url": meta_url}  # Use the constructed URL here
                    })
                    gold_evidence_ids.append(evidence_id)
                    global_request_counter += 1
                    request_count += 1
                else:
                    print(f"Warning: {page_title} not found in the request evidence for question_id {item.get('question_id')}")

        if "header" in gold_evidence_table and "data" in gold_evidence_table:
            headers = [header[0] for header in gold_evidence_table["header"]]
            rows = [[cell[0] for cell in row] for row in gold_evidence_table["data"]]
            table_content = {
                "columns": headers,
                "rows": rows
            }
            
            evidence_id = f"table_{global_table_counter}"
            combined_formatted.append({
                "id": evidence_id,
                "title": gold_evidence_table.get("title", ""),
                "content": table_content,
                "meta": {"url": gold_evidence_table.get("url", "")}
            })
            gold_evidence_ids.append(evidence_id)
            global_table_counter += 1
            table_count += 1

        rearranged_item = OrderedDict([
            ("id", id),
            ("seed_question", item.get("question", "")),
            ("seed_dataset", "HYBRID"),
            ("seed_split", "dev"),
            ("seed_answers", item.get("answer-text", "")),
            ("seed_id", item.get("question_id", "")),
            ("extended_question", ""),
            ("gold_evidence_ids", gold_evidence_ids),
            ("gold_evidence_type", {
                "request_count": request_count,
                "table_count": table_count
            }),
            ("gold_evidences", combined_formatted),
            ("temporal_reasoning", ""),
            ("numerical_operation_program", ""),
            ("difficulty", ""),
            ("meta", [item.get("question_postag", ""), table_id, item.get("answer_node", "")])
        ])
        rearranged_data.append(rearranged_item)
        id += 1
    
    return rearranged_data

# Specify the file path and the request/table paths
json_filepath = 'processed/hybrid/dev_numerical_output.json'
request_path = 'hybridqa/WikiTables-WithLinks/request_tok'
table_path = 'hybridqa/WikiTables-WithLinks/tables_tok'

# Load and rearrange the data
data = read_json_file(json_filepath)
reformatted_data = rearrange_data(data, request_path, table_path)

# Save the reformatted data to a new JSON file (not JSONL)
output_filepath = 'processed/hybrid/new_dev_hybrid.json'
with open(output_filepath, 'w') as file:
    json.dump(reformatted_data, file, ensure_ascii=False, indent=4)

print(f"Reformatted data has been saved to '{output_filepath}'")
