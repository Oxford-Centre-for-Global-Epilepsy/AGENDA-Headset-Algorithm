import os
import json
from pathlib import Path

# Set your target directory and required fields
# Set the directory of the result files
parent_dir = Path(__file__).resolve().parent.parent
TARGET_DIR = os.path.join(parent_dir, "training/results")

REQUIRED_FIELDS = {"EEGNet_dropout"}  # Set the required keys inside "params"

def file_should_be_deleted(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if "params" not in data:
            return True
        if not isinstance(data["params"], dict):
            return True
        if not REQUIRED_FIELDS.issubset(data["params"].keys()):
            return True
        return False
    except Exception as e:
        print(f"Failed to process {json_path}: {e}")
        return True  # Delete unreadable or invalid JSON files

def main():
    for filename in os.listdir(TARGET_DIR):
        if filename.endswith(".json"):
            full_path = os.path.join(TARGET_DIR, filename)
            if file_should_be_deleted(full_path):
                print(f"Deleting {filename}...")
                os.remove(full_path)

if __name__ == "__main__":
    main()
