from pathlib import Path
import shutil

# Define base paths
raw_root = Path("data/raw")
edf_root = Path("data/edf")

# Move only .edf files from raw to edf, keeping directory structure
for file_path in raw_root.rglob("*.edf"):
    if file_path.is_file():
        relative_path = file_path.relative_to(raw_root)
        target_path = edf_root / relative_path

        # Ensure the target directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Move the file
        shutil.move(str(file_path), str(target_path))
        print(f"Moved: {file_path} -> {target_path}")
