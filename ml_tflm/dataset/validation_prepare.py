import os
import numpy as np

from ml_tflm.dataset.test_prepare import sample_reshape, save_data
import ml_tflm.training.train_utils as utils

from pathlib import Path

# --- Load label config ---
label_config = utils.load_label_config("ml_tflm/training/label_map.JSON")

# --- define internal label cap ---
internal_label_cap = {
    0: 450,
    3: 260,
    4: 94,
    5: 96
}

    # --- load dataset using utils ---

train_val_sets, test_dataset, label_histograms = utils.prepare_eeg_datasets(
    h5_file_path="ml_tflm/dataset/agenda_data_23_bp45_tr05/merged_south_africa_monopolar_standard_10_20.h5",
    dataset_name="combined_south_africa_monopolar_standard_10_20",
    label_config=label_config,
    omit_channels=["A1","A2", "Fz", "Pz", "Cz"],
    val_frac=0.2,
    test_frac=0.15,
    k_fold=True,
    stratify=True,
    internal_label_cap=internal_label_cap,
    batch_size=1,
    mirror_flag=False,
    chunk_size=None
)

# Create output directory

# Resolving the directories
parent_dir = Path(__file__).resolve().parent
out_dir = os.path.join(parent_dir, "test_data")

os.makedirs(out_dir, exist_ok=True)

count = 0
for sample in test_dataset:   # no .take(N) → iterate all
    data = sample["data"].numpy()            # (E, C, T)
    data = np.squeeze(data)
    label = sample["internal_label"].numpy()
    mask = sample["attention_mask"].numpy()
    mask = np.squeeze(mask)

    print(data.shape)
    print(mask.shape)

    data = data[mask]
    print(data.shape)


    reshaped = sample_reshape(data, False)   # (E*T, C)


    file_name = f"{label_config['inverse_label_map'][label[0]][0].upper()}_{count:04d}.csv"
    file_path = os.path.join(out_dir, file_name)
    save_data(reshaped, file_path)

    count += 1
    if count % 100 == 0:
        print(f"Saved {count} validation samples...")

print(f"Done. Saved {count} validation samples to {out_dir}")
