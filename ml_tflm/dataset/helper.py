import os
from pathlib import Path

def count_eeg_files(folder_path):
    count = 0
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.eeg'):
            count += 1
    return count

if __name__ == "__main__":
    # Get the parent of the parent directory of this script
    folder = Path(__file__).resolve().parent.parent.parent
    folder = os.path.join(folder, "data/raw/south_africa/neurotypical")

    if not os.path.isdir(folder):
        print(f"Invalid folder path: {folder}")
    else:
        num_eeg_files = count_eeg_files(folder)
        print(f"Number of .EEG files: {num_eeg_files}")
