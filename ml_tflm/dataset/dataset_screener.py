import h5py
import os
import numpy as np
from tqdm import tqdm

# ====== Hardcoded config ======
INPUT_PATH = "ml_tflm/dataset/agenda_data_05/combined_south_africa_monopolar_standard_10_20.h5"
OUTPUT_PATH = "ml_tflm/dataset/agenda_data_06/cleaned_south_africa_monopolar_standard_10_20.h5"
DATASET_NAME = "combined_south_africa_monopolar_standard_10_20"
MIN_EPOCHS = 600

# Remove old output if exists
if os.path.exists(OUTPUT_PATH):
    print(f"[!] Removing existing file: {OUTPUT_PATH}")
    os.remove(OUTPUT_PATH)

# Open input and create output
with h5py.File(INPUT_PATH, "r") as h5_in:
    in_group = h5_in[DATASET_NAME]

    with h5py.File(OUTPUT_PATH, "w") as h5_out:
        out_group = h5_out.create_group(DATASET_NAME)

        # Copy dataset-level attributes
        for attr_key in in_group.attrs:
            out_group.attrs[attr_key] = in_group.attrs[attr_key]

        # Loop through subjects
        kept = 0
        max_epochs = 0

        for sid in tqdm(in_group.keys(), desc="Filtering subjects"):
            try:
                subj = in_group[sid]
                n_epochs = subj["data"].shape[0]

                if n_epochs < MIN_EPOCHS:
                    continue

                # Copy group
                dst = out_group.create_group(sid)
                for key in subj:
                    dst.create_dataset(key, data=subj[key][()])
                for attr_key in subj.attrs:
                    dst.attrs[attr_key] = subj.attrs[attr_key]

                max_epochs = max(max_epochs, n_epochs)
                kept += 1

            except Exception as e:
                print(f"[skip] {sid}: {e}")
                continue

        # Write max_epochs to group and root
        out_group.attrs["max_epochs"] = max_epochs
        h5_out.attrs["max_epochs"] = max_epochs

        print(f"[done] Kept {kept} subjects (>= {MIN_EPOCHS} epochs)")
        print(f"[done] Max epochs written: {max_epochs}")
