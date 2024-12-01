import json

# Load JSON data from a file
with open('processed/tanq/tanq_reformatted.json', 'r') as file:
    data = json.load(file)

# Pretty-print JSON to a new file
with open('processed/tanq/pretty_tanq.json', 'w') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print("Pretty-printed JSON saved to 'pretty_file.json'")
