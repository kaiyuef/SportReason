import json
import os

def explore_keys(data, path=""):
    """
    Recursively explore the keys in a JSON object. 
    Prints out the keys and explores nested dictionaries and lists.
    
    Parameters:
    - data: The JSON data (dict, list, or other).
    - path: The current path of keys, for recursive tracking.
    """
    if isinstance(data, dict):
        # For each key in the dictionary, recursively explore
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            print(f"Found key: {current_path}")
            explore_keys(value, current_path)
    elif isinstance(data, list):
        # If we have a list, explore each item (assumed to be dictionaries)
        for index, item in enumerate(data):
            current_path = f"{path}[{index}]"
            print(f"Exploring list item at: {current_path}")
            explore_keys(item, current_path)
    else:
        # Reached an end value (non-dict, non-list)
        print(f"Value at {path}: {data}")

def find_keys_in_json(file_path):
    """
    Loads a JSON file and finds all keys within lists of dictionaries recursively.
    
    Parameters:
    - file_path: The path to the JSON file.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return

    print("Exploring JSON data...")
    explore_keys(data)

# Specify your JSON file path
json_file_path = 'processed/tanq/tanq_reformatted.json'  # Replace 'your_file.json' with your JSON file path

# Start exploring the keys in the JSON file
find_keys_in_json(json_file_path)
