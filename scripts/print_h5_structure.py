import h5py
import os

def print_hdf5_structure(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    def print_attrs(name, obj):
        print(f"🔹 {name}")
        for key, val in obj.attrs.items():
            print(f"    ⤷ attr '{key}': {val}")
        if isinstance(obj, h5py.Dataset):
            print(f"    ⤷ shape: {obj.shape}, dtype: {obj.dtype}")

    print(f"\n📂 HDF5 structure for: {file_path}\n" + "-"*60)
    with h5py.File(file_path, "r") as f:
        f.visititems(print_attrs)

# Replace this with the path to your newly combined file
DATA_PATH = os.getenv("DATA")
if not DATA_PATH:
    raise ValueError("ERROR: The $DATA environment variable is not set!")

project_folder_path = f"{DATA_PATH}/AGENDA-Headset-Algorithm/"
combined_h5_file = project_folder_path + "data/final_processed/combined_south_africa_monopolar_standard_10_20.h5"
print_hdf5_structure(combined_h5_file)