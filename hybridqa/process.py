import json
from collections import OrderedDict
import os

# Load the JSON file
with open('dev_step3.json', 'r') as file:
    data = json.load(file)

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
    # Sort the data by question_id to ensure order
    sorted_data = sorted(data, key=lambda x: x["question_id"])
    
    # Initialize global counters
    global_request_counter = 1
    global_table_counter = 1

    for item in sorted_data:
        table_id = item.get("table_id", "")
        request_evidence_path = os.path.join(request_path, f"{table_id}.json")
        table_evidence_path = os.path.join(table_path, f"{table_id}.json")
        
        gold_evidence_request = read_json_content(request_evidence_path)
        gold_evidence_table = read_json_content(table_evidence_path)

        # Convert request and table evidence to the specified format and combine them
        combined_formatted = []
        request_count = 0
        table_count = 0
        gold_evidence_ids = []
        
        for key, value in gold_evidence_request.items():
            evidence_id = f"text_{global_request_counter}"
            combined_formatted.append({
                "id": evidence_id,
                "title": key,
                "content": value
            })
            gold_evidence_ids.append(evidence_id)
            global_request_counter += 1
            request_count += 1
        
        for key, value in gold_evidence_table.items():
            evidence_id = f"table_{global_table_counter}"
            combined_formatted.append({
                "id": evidence_id,
                "title": key,
                "content": value
            })
            gold_evidence_ids.append(evidence_id)
            global_table_counter += 1
            table_count += 1

        rearranged_item = OrderedDict([
            ("id", id),
            ("seed_question", item.get("question", "")),
            ("seed_dataset", "HYBRID"),
            ("seed_id", item.get("question_id", "")),
            ("extended_question", ""),
            ("answers", item.get("answer-text", "")),
            ("gold_evidence_ids", gold_evidence_ids),
            ("gold_evidence_type", {
                "request_count": request_count,
                "table_count": table_count
            }),
            ("gold_evidence", combined_formatted),
            ("temporal_reasoning", ""),
            ("numerical_operation_program", ""),
            ("difficulty", ""),
            ("meta", [item.get("question_postag", ""), table_id, item.get("answer_node", "")])
        ])
        rearranged_data.append(rearranged_item)
        id += 1
    
    return rearranged_data

# Specify the request and table paths
request_path = 'request_tok'
table_path = 'tables_tok'

# Rearrange the data
reformatted_data = rearrange_data(data, request_path, table_path)

# Save the reformatted data to a new JSON file
with open('new_reformatted_questions.json', 'w') as file:
    json.dump(reformatted_data, file, ensure_ascii=False, indent=4)

print("Reformatted data has been saved to 'new_reformatted_questions.json'")
