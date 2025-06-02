import json

def count_entries_in_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        count = len(data)
        print(f"The JSON file contains {count} top-level keys.")
    elif isinstance(data, list):
        count = len(data)
        print(f"The JSON file contains {count} elements in the list.")
    else:
        print("Unsupported JSON structure.")
        count = 0

    return count

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Count entries in a JSON file.")
    parser.add_argument("--input", required=True, help="Path to the JSON file")
    args = parser.parse_args()

    count_entries_in_json(args.input)
