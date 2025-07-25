import os
import subprocess
import glob

# === Hardcoded Configuration ===
base_path = "/home/anyu/projects/agenda/AGENDA-Headset-Algorithm"
edf_root = f"{base_path}/data/edf"
temp_root = f"{base_path}/data/temp"
output_root = f"{base_path}/data/processed"
montage_type = "monopolar"
montage_name = "standard_10_20"
config_path = f"{base_path}/workflow/config"

# === Helper Function ===
def run_stage(script_path, args):
    command = ["python", script_path] + args
    print("\n▶️  Running:", " ".join(command))
    subprocess.run(command, check=True)

# === Full Pipeline per File ===
def process_file(edf_path):
    # Parse file metadata
    parts = edf_path.split("/")
    site = parts[-3]
    data_label = parts[-2]
    filename = os.path.basename(edf_path)
    sample = filename.replace(".edf", "")

    # Construct paths
    temp_dir = f"{temp_root}/{montage_type}/{montage_name}/{site}/{data_label}"
    output_dir = f"{output_root}/{montage_type}/{montage_name}/{site}/{data_label}"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    filtered_fif = f"{temp_dir}/{sample}_filtered.fif"
    resampled_fif = f"{temp_dir}/{sample}_resampled.fif"
    montaged_fif = f"{temp_dir}/{sample}_headset_montage.fif"
    epoched_fif = f"{temp_dir}/{sample}_epoched.fif"
    normalised_fif = f"{temp_dir}/{sample}_normalised.fif"
    hdf5_path = f"{output_dir}/{sample}.h5"

    # Skip if final output exists
    if os.path.exists(hdf5_path):
        print(f"✅ Skipping {sample} (already processed)")
        return

    # Run pipeline
    run_stage(f"{base_path}/workflow/scripts/data_preprocessing/bandpass_filter_data.py", [
        edf_path, filtered_fif, os.path.join(config_path, "filter_settings.yaml")
    ])

    run_stage(f"{base_path}/workflow/scripts/data_preprocessing/resample_data.py", [
        filtered_fif, resampled_fif, os.path.join(config_path, "filter_settings.yaml")
    ])

    run_stage(f"{base_path}/workflow/scripts/data_preprocessing/convert_to_montage.py", [
        resampled_fif, montaged_fif, os.path.join(config_path, "spatial_montages.yaml"), montage_type, montage_name
    ])

    run_stage(f"{base_path}/workflow/scripts/data_preprocessing/epoch_data.py", [
        montaged_fif, epoched_fif, os.path.join(config_path, "epoch_settings.yaml")
    ])

    run_stage(f"{base_path}/workflow/scripts/data_preprocessing/normalise_epoched_data.py", [
        epoched_fif, normalised_fif
    ])

    run_stage(f"{base_path}/workflow/scripts/data_preprocessing/convert_to_hdf5.py", [
        normalised_fif, hdf5_path, os.path.join(config_path, "hdf5_settings.yaml"), montage_type, montage_name, site, data_label
    ])

# === Run All Files ===
def main():
    edf_files = glob.glob(f"{edf_root}/*/*/*.edf")
    print(f"🔎 Found {len(edf_files)} EDF files.")
    for i, edf in enumerate(edf_files):
        print(f"\n[{i+1}/{len(edf_files)}] Processing {edf}")
        try:
            process_file(edf)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {edf}: {e}")

if __name__ == "__main__":
    main()
