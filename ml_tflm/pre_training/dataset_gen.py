import h5py
import numpy as np
import os
from tqdm import tqdm
from ml_tflm.pre_training.dataset_pretrain_aug import EEGBatchAugmentor  # Assumed to be in the same folder or installed
from collections import defaultdict
import json

# ---- CONFIG ----
H5_INPUT_PATH = "ml_tflm/dataset/agenda_data_01/combined_south_africa_monopolar_standard_10_20.h5"
DATASET_NAME = "combined_south_africa_monopolar_standard_10_20"
OUTPUT_H5_PATH = "ml_tflm/dataset/agenda_data_01/augmented_buffered.h5"
json_file = 'ml_tflm/dataset/agenda_data_01/augmented_bin_info.json'
SEGMENTS_PER_BIN = 10000
NUM_VIEWS = 8  # Total views to include diverse augmentations
LOGITS = [1.0, 1.0, 2.0, 1.0, 1.0, 2.5]  # strong augmentations last
VIEW_TO_AUGMENT_IDX = [0, 1, 2, 2, 3, 4, 5, 5]  # augment method index per view

LABEL_MAP_INTERNAL = {
    "neurotypical": 0,
    "generalized": 1,
    "left": 2,
    "right": 2,
}

LABEL_MAP_INTERNAL = {
    "neurotypical": 0,
    "generalized": 1,
    "left": 2,
    "right": 2,
}

# ---- SCRIPT ----
def generate_augmented_bins():
    augmentor = EEGBatchAugmentor(logits=LOGITS)

    with h5py.File(H5_INPUT_PATH, "r") as src_h5, h5py.File(OUTPUT_H5_PATH, "w") as dst_h5:
        subjects = list(src_h5[DATASET_NAME].keys())
        bin_counts = {0: 0, 1: 0, 2: 0}  # class -> count per current bin
        bin_ids = {0: 0, 1: 0, 2: 0}     # class -> current bin id

        for subj in tqdm(subjects):
            grp = src_h5[DATASET_NAME][subj]
            data = grp["data"][()]
            labels = grp.attrs["class_labels"]
            if isinstance(labels[0], bytes):
                labels = [l.decode("utf-8") for l in labels]

            # Assign class
            for i in reversed(range(3)):
                if i < len(labels) and labels[i] in LABEL_MAP_INTERNAL:
                    class_id = LABEL_MAP_INTERNAL[labels[i]]
                    break
            else:
                continue  # skip subject if not classifiable

            # Extract usable epochs
            mask = np.ones(data.shape[0], dtype=bool)
            if "epoch_mask" in grp:
                mask = grp["epoch_mask"][()].astype(bool)[:data.shape[0]]

            epochs = data[mask]
            if len(epochs) == 0:
                continue

            # Check if adding this subject will overflow bin
            if bin_counts[class_id] + len(epochs) > SEGMENTS_PER_BIN:
                bin_ids[class_id] += 1
                bin_counts[class_id] = 0

            # Manually apply specific augmentations per view
            augmented_views = []
            for method_idx in VIEW_TO_AUGMENT_IDX:
                method = augmentor.augment_methods[method_idx]
                view = np.stack([method(ep.copy(), label_name=class_id) for ep in epochs])
                augmented_views.append(view)

            # For each original segment, group all views together
            for seg_idx in range(len(epochs)):
                class_path = f"class_{class_id}/bin_{bin_ids[class_id]}"
                seg_name = f"seg_{bin_ids[class_id]:02d}{bin_counts[class_id]:05d}"
                seg_grp = dst_h5.require_group(class_path).create_group(seg_name)

                # Stack views
                views = np.stack([augmented_views[v][seg_idx] for v in range(NUM_VIEWS)], axis=0)
                seg_grp.create_dataset("original", data=epochs[seg_idx].astype(np.float32))
                seg_grp.create_dataset("augmented", data=views.astype(np.float32))  # shape: (NUM_VIEWS, C, T)
                seg_grp.create_dataset("patient_id", data=subj)
                seg_grp.create_dataset("segment_index", data=int(seg_idx))
                seg_grp.create_dataset("view_to_aug_idx", data=np.array(VIEW_TO_AUGMENT_IDX, dtype=np.int32))

                bin_counts[class_id] += 1

    print(f"Completed writing to {OUTPUT_H5_PATH}")

def print_h5_structure(h5_path):
    def print_attrs(name, obj):
        indent = "  " * (name.count("/") - 1)
        if isinstance(obj, h5py.Group):
            print(f"{indent}- [Group] {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}- [Dataset] {name} - shape: {obj.shape}, dtype: {obj.dtype}")
    
    with h5py.File(h5_path, "r") as f:
        print(f"Inspecting structure of: {h5_path}")
        f.visititems(print_attrs)

def summarize_bins(h5_path):
    with h5py.File(h5_path, "r") as f:
        summary = defaultdict(lambda: defaultdict(int))  # class -> bin -> count

        for class_key in f:
            class_group = f[class_key]
            for bin_key in class_group:
                bin_group = class_group[bin_key]
                count = len(bin_group)
                summary[class_key][bin_key] = count

    for class_id in sorted(summary.keys()):
        print(f"{class_id}:")
        for bin_id in sorted(summary[class_id].keys()):
            count = summary[class_id][bin_id]
            print(f"  {bin_id}: {count} segments")

def collect_patient_ids_per_bin(h5_path):
    patient_dict = defaultdict(lambda: defaultdict(set))

    with h5py.File(h5_path, "r") as f:
        for class_key in f:
            class_group = f[class_key]
            print(f"===== {class_key} ===")

            for bin_key in class_group:
                print(f" -> Investigating {class_key}: {bin_key}...")
                bin_group = class_group[bin_key]
                for segment_key in bin_group:
                    segment = bin_group[segment_key]
                    if 'patient_id' in segment:  # check dataset key, not attrs
                        pid = segment['patient_id'][()]
                        if isinstance(pid, bytes):
                            pid = pid.decode('utf-8')
                        patient_dict[class_key][bin_key].add(pid)
                    else:
                        # Handle other cases or raise warning if patient_id missing
                        print(f"Warning: 'patient_id' dataset not found in segment {segment_key} of bin {bin_key}")


    # Convert sets to lists for JSON serialization
    result = {cls: {b: list(pids) for b, pids in bins.items()} for cls, bins in patient_dict.items()}
    return result

def save_to_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # generate_augmented_bins()
    # print_h5_structure(OUTPUT_H5_PATH)
    # summarize_bins(OUTPUT_H5_PATH)
    patient_bins = collect_patient_ids_per_bin(OUTPUT_H5_PATH)
    save_to_json(patient_bins, json_file)