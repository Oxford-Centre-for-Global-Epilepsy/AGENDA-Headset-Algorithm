import os
import h5py
import numpy as np
import argparse
import yaml
from tqdm import tqdm

def find_all_h5_files(base_path, sites):
    matched_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".h5") and any(site in root for site in sites):
                matched_files.append(os.path.join(root, file))
    return matched_files

def combine_hdf5_files(config, output_file, project_path):
    dataset_name = config["dataset_name"]
    montage_type = config["montage_type"]
    montage_name = config["montage_name"]
    sites = config.get("sites_to_include", [])
    input_files = config.get("input_files", None)

    # Auto-discover files if not specified
    if not input_files:
        print("🔍 No input_files specified, searching using sites_to_include...")
        base_path = os.path.join(project_path, "data", "processed", montage_type, montage_name)
        input_files = find_all_h5_files(base_path, sites)
        print(f"✅ Found {len(input_files)} matching HDF5 files")

    if not input_files:
        raise ValueError("❌ No input HDF5 files found or specified. Check your config.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    max_epochs = 0

    with h5py.File(output_file, "w") as out_f:
        root_grp = out_f.create_group(dataset_name)
        out_f.attrs["montage_type"] = montage_type
        out_f.attrs["montage_name"] = montage_name
        out_f.attrs["sites_included"] = ", ".join(sites)
        out_f.attrs["max_epochs"] = 0

        for in_file in tqdm(input_files, desc="🔗 Merging files"):
            subject_id = os.path.splitext(os.path.basename(in_file))[0]
            with h5py.File(in_file, "r") as f:
                eeg_group = f["EEG"]
                subj_grp = root_grp.create_group(subject_id)

                # Always copy the main data array
                if "data" in eeg_group:
                    subj_grp.create_dataset("data", data=eeg_group["data"][()], compression="gzip")
                else:
                    print(f"⚠️ Dataset 'data' not found in {in_file}")

                # Copy the other fields as attributes
                for attr_name in ["sfreq", "total_duration", "epoch_duration", "channel_names", "class_labels"]:
                    if attr_name in eeg_group:
                        value = eeg_group[attr_name][()]
                        if isinstance(value, np.ndarray) and value.dtype.char == "S":
                            value = [v.decode("utf-8") for v in value]  # decode bytes
                        subj_grp.attrs[attr_name] = value
                    else:
                        print(f"⚠️ Metadata field {attr_name} not found in {in_file}")

                # Copy attributes
                for attr, value in eeg_group.attrs.items():
                    subj_grp.attrs[attr] = value

                 #  Track max number of epochs by checking the first dimension of the "data" array
                if "data" in eeg_group:
                    data_shape = eeg_group["data"].shape
                    n_epochs = data_shape[0]
                    if n_epochs > max_epochs:
                        max_epochs = n_epochs

                # Record provenance
                subj_grp.attrs["source_file"] = os.path.relpath(in_file, project_path)
                subj_grp.attrs["site"] = next((site for site in sites if site in in_file), "unknown")

        out_f.attrs["max_epochs"] = max_epochs

    print(f"✅ Combined dataset saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--output", required=True, help="Path to output HDF5 file")
    parser.add_argument("--source_filepath", required=True, help="Top-level path to project on HPC cluster")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    combine_hdf5_files(config, args.output, args.source_filepath)

if __name__ == "__main__":
    main()
