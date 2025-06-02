import os
import h5py
import numpy as np
import argparse
import yaml
import pandas as pd
from tqdm import tqdm

def find_all_h5_files(base_path, sites):
    matched_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".h5") and any(site in root for site in sites):
                matched_files.append(os.path.join(root, file))
    return matched_files

def is_valid_hdf5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            return True
    except OSError:
        return False

def combine_hdf5_files(experiment_name, site, montage_type, montage_name, output_file, project_path):
    
    dataset_name = "EEG"
    montage_type = montage_type
    montage_name = montage_name
    sites = [site]
    input_files = None

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
    n_channels = 0
    summary_list = []
    duplicate_subjects = []

    with h5py.File(output_file, "w") as out_f:
        root_grp = out_f.create_group(dataset_name)
        out_f.attrs["montage_type"] = montage_type
        out_f.attrs["montage_name"] = montage_name
        out_f.attrs["sites_included"] = ", ".join(sites)
        out_f.attrs["max_epochs"] = 0
        out_f.attrs["n_channels"] = 0
        
        file_id_counter=0
        for in_file in tqdm(input_files, desc="🔗 Merging files"):
            file_id_counter = file_id_counter + 1
            subject_id = os.path.splitext(os.path.basename(in_file))[0]

            # Debug Print statement
            #print(f"{file_id_counter} / {len(input_files)}: {in_file}")
            
            # Check that the file is valid before trying to add it to the h5 file
            if not is_valid_hdf5(in_file):
                print(f"⚠️ Skipping invalid HDF5 file: {in_file}")
                continue

            # Skip duplicate subject IDs
            if subject_id in root_grp:
                print(f"⚠️ Skipping duplicate subject ID: {subject_id}")
                duplicate_subjects.append({
                    "subject_id": subject_id,
                    "file": in_file
                })
                continue

            with h5py.File(in_file, "r") as f:
                eeg_group = f["EEG"]
                subj_grp = root_grp.create_group(subject_id)

                # Always copy the main data array
                if "data" in eeg_group:
                    data = eeg_group["data"][()]
                    subj_grp.create_dataset("data", data=data, compression="gzip")
                    
                    # Get the number of channels and epochs
                    n_epochs = data.shape[0]
                    if n_epochs > max_epochs:
                        max_epochs = n_epochs
                    if n_channels == 0:
                        n_channels = data.shape[1]
                else:
                    print(f"⚠️ Dataset 'data' not found in {in_file}")
                    continue

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

                # Record provenance
                source_file = os.path.relpath(in_file, project_path)
                site = next((site for site in sites if site in in_file), "unknown")
                subj_grp.attrs["source_file"] = source_file
                subj_grp.attrs["site"] = site

                # Append to summary
                summary_list.append({
                    "subject_id": subject_id,
                    "source_file": source_file,
                    "site": site,
                    "n_channels": n_channels,
                    "n_epochs": n_epochs,
                    "epoch_duration": subj_grp.attrs.get("epoch_duration", None),
                    "total_duration": subj_grp.attrs.get("total_duration", None),
                    "class_labels": subj_grp.attrs.get("class_labels", None)
                })

        out_f.attrs["max_epochs"] = max_epochs
        out_f.attrs["n_channels"] = n_channels

        # Close the combined h5 file
        out_f.close()

    print(f"✅ Combined dataset saved to: {output_file}")

    # Write summary files
    base_path = os.path.splitext(output_file)[0]
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(base_path + "_summary.csv", index=False)
    summary_df.to_json(base_path + "_summary.json", orient="records", indent=2)
    print(f"📝 Summary saved to:\n- {base_path}_summary.csv\n- {base_path}_summary.json")

    # Write duplicate subject log
    if duplicate_subjects:
        pd.DataFrame(duplicate_subjects).to_csv(base_path + "_duplicates.csv", index=False)
        print(f"📝 Duplicate subjects saved to: {base_path}_duplicates.csv")
    else:
        print("✅ No duplicate subjects detected.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True, help="The name of the experiment")
    parser.add_argument("--site", required=True, help="The site to include in the dataset")
    parser.add_argument("--montage_type", required=True, help="The type of montage (e.g., 'monopolar', 'bipolar')")
    parser.add_argument("--montage_name", required=True, help="The name of the montage")
    parser.add_argument("--output", required=True, help="Path to output HDF5 file")
    parser.add_argument("--source_filepath", required=True, help="Top-level path to project on HPC cluster")
    args = parser.parse_args()

    combine_hdf5_files(args.experiment_name, args.site, args.montage_type, args.montage_name, args.output, args.source_filepath)

if __name__ == "__main__":
    main()
