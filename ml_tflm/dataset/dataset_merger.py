import h5py
import numpy as np
import os
import math
from collections import defaultdict
from scipy.signal import welch

from tqdm import tqdm
import gc
import matplotlib.pyplot as plt

from ml_tflm.dataset.eeg_dataset_rewrite import EEGRecordingDatasetTF

class EEGRecordingDatasetWithShape(EEGRecordingDatasetTF):
    def build_lookup(self):
        self.sample_metadata = []

        for sid in self.subject_ids:
            subj_group = self.h5[self.dataset_name][sid]
            
            # Get label hierarchy
            labels = subj_group.attrs["class_labels"]
            if isinstance(labels[0], bytes):
                labels = [l.decode("utf-8") for l in labels]

            label_ids = np.full((3,), -1, dtype=np.int32)
            label_mask = np.zeros((3,), dtype=bool)
            for i in range(min(3, len(labels))):
                label_ids[i] = self.label_map.get(labels[i], -1)
                label_mask[i] = label_ids[i] != -1

            for i in reversed(range(3)):
                if label_mask[i]:
                    internal_label = self.label_map_internal.get(labels[i], -1)
                    break
            else:
                continue  # Skip if invalid

            # Query data shape without loading it
            data_shape = subj_group["data"].shape  # (epochs, channels, timepoints)

            self.sample_metadata.append({
                "subject_id": sid,
                "labels": label_ids,
                "label_mask": label_mask,
                "internal_label": internal_label,
                "is_mirrored": False,
                "data_shape": data_shape
            })

def find_suspected_duplicates(meta_a, meta_b):
    # Group metadata entries by (internal_label, num_epochs)
    def group_by_signature(meta_list):
        groups = defaultdict(list)
        for entry in meta_list:
            label = entry["internal_label"]
            n_epochs = entry["data_shape"][0]  # just use number of epochs
            signature = (label, n_epochs)
            groups[signature].append(entry)
        return groups

    group_a = group_by_signature(meta_a)
    group_b = group_by_signature(meta_b)

    suspected = []

    # Compare shared groups
    for signature in group_a.keys() & group_b.keys():
        for a in group_a[signature]:
            for b in group_b[signature]:
                suspected.append((a["subject_id"], b["subject_id"], signature))

    return suspected

def compute_bandpower_signature(data, fs=128, bands=((1, 4), (4, 8), (8, 13), (13, 30))):
    epochs, channels, timepoints = data.shape
    bandpowers = []

    for ch in range(channels):
        x = data[:10, ch, :].reshape(-1)  # First 3 epochs = 3 seconds
        freqs, psd = welch(x, fs=fs, nperseg=min(128, len(x)))  # short segment
        powers = [
            np.log(np.mean(psd[(freqs >= lo) & (freqs < hi)] + 1e-10))
            for lo, hi in bands
        ]
        bandpowers.append(powers)

    return np.array(bandpowers)  # shape [C, B]

def refine_duplicate_check(dups, dataset_a, dataset_b, threshold=0.001, fs=128, plot_flag=False):
    # Step 1: Collect all involved subject IDs
    ids_a = sorted(set([sid_a for sid_a, _, _ in dups]))
    ids_b = sorted(set([sid_b for _, sid_b, _ in dups]))

    # Step 2: Precompute bandpower signatures for dataset A
    print("[A] Computing bandpower signatures...")
    bp_cache_a = {}
    for sid in tqdm(ids_a):
        try:
            data = dataset_a.h5[dataset_a.dataset_name][sid]["data"][()]
            bp_cache_a[sid] = compute_bandpower_signature(data, fs=fs)
            del data  # release memory immediately
            gc.collect()
        except Exception as e:
            print(f"[Skip A] {sid}: {e}")

    # Step 3: Precompute bandpower signatures for dataset B
    print("[B] Computing bandpower signatures...")
    bp_cache_b = {}
    for sid in tqdm(ids_b):
        try:
            data = dataset_b.h5[dataset_b.dataset_name][sid]["data"][()]
            bp_cache_b[sid] = compute_bandpower_signature(data, fs=fs)
            del data
            gc.collect()
        except Exception as e:
            print(f"[Skip B] {sid}: {e}")

    # Step 4: Compare and track all MSEs
    print("[C] Comparing signatures...")
    confirmed = []
    all_mse = []

    for sid_a, sid_b, sig in tqdm(dups):
        bp_a = bp_cache_a.get(sid_a)
        bp_b = bp_cache_b.get(sid_b)

        if bp_a is None or bp_b is None:
            continue
        if bp_a.shape != bp_b.shape:
            continue

        mse = np.mean((bp_a - bp_b) ** 2)
        all_mse.append(mse)

        if mse < threshold:
            confirmed.append((sid_a, sid_b, sig, mse))

    # Step 5: Plot histogram of all MSEs
    if all_mse and plot_flag:
        plt.figure(figsize=(8, 4))
        plt.hist(all_mse, bins=30, color='skyblue', edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold}")
        plt.xlabel("MSE between bandpower signatures")
        plt.ylabel("Count")
        plt.title("Distribution of MSEs for Suspected Duplicate Pairs")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return confirmed

def plot_epoch_count_histogram(meta, max_bin=5000):
    # Step 1: Auto-detect unique internal labels
    from collections import defaultdict
    label_to_counts = defaultdict(list)
    for entry in meta:
        label = entry.get("internal_label", None)
        if label is not None:
            label_to_counts[label].append(entry["data_shape"][0])

    labels = sorted(label_to_counts.keys())
    n_labels = len(labels)

    # Step 2: Set up dynamic grid size
    ncols = 2
    nrows = math.ceil(n_labels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()

    # Step 3: Plot per label
    for i, label in enumerate(labels):
        ax = axes[i]
        counts = label_to_counts[label]

        ax.hist(counts, bins=100, color='cornflowerblue', edgecolor='black')
        ax.axvline(max_bin, color='red', linestyle='--', label=f"Threshold = {max_bin}")
        ax.set_title(f"Label {label} ({len(counts)} patients)")
        ax.set_xlabel("Epochs per Patient")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    # Hide unused subplots
    for j in range(len(labels), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Step 4: Print top 5 longest recordings
    sorted_meta = sorted(meta, key=lambda x: -x["data_shape"][0])
    print("\nTop 5 longest recordings:")
    for entry in sorted_meta[:5]:
        n_epochs = entry["data_shape"][0]
        label = entry.get("internal_label", "N/A")
        print(f"  {entry['subject_id']:>12} - {n_epochs:>5} epochs | label: {label}")

def merge_datasets_without_duplicates(
    h5_path_a, dataset_name_a,
    h5_path_b, dataset_name_b,
    confirmed_duplicates,
    output_path,
    output_dataset_name="merged",
    min_epochs=600,
    max_epochs=np.inf,
    internal_label_cap=None,
    label_config=None
):
    import collections

    if label_config is None:
        raise ValueError("label_config must be provided.")

    label_map = label_config["label_map"]
    label_map_internal = {key: i for i, key in enumerate(label_map.keys())}

    dup_b_ids = {sid_b for _, sid_b, _, _ in confirmed_duplicates}

    with h5py.File(h5_path_a, "r") as h5_a, h5py.File(h5_path_b, "r") as h5_b:
        group_a = h5_a[dataset_name_a]
        group_b = h5_b[dataset_name_b]

        all_ids_a = set(group_a.keys())
        all_ids_b = set(group_b.keys()) - dup_b_ids
        all_ids = sorted(all_ids_a | all_ids_b)

        if os.path.exists(output_path):
            print(f"[!] Removing existing file: {output_path}")
            os.remove(output_path)

        with h5py.File(output_path, "w") as h5_out:
            merged_group = h5_out.create_group(output_dataset_name)

            for attr_key in group_a.attrs:
                merged_group.attrs[attr_key] = group_a.attrs[attr_key]

            max_epochs_found = 0
            label_counter = collections.defaultdict(int)

            for sid in tqdm(all_ids, desc="Merging subjects"):
                src = None
                if sid in group_a:
                    src = group_a[sid]
                elif sid in group_b:
                    src = group_b[sid]
                else:
                    print(f"[Warning] Skipped {sid}: not found in either A or B")
                    continue

                try:
                    epoch_count = src["data"].shape[0]
                except Exception as e:
                    print(f"[Error] Could not read data for {sid}: {e}")
                    continue

                if not (min_epochs <= epoch_count <= max_epochs):
                    print(f"[Skip] {sid}: {epoch_count} epochs (outside range {min_epochs}–{max_epochs})")
                    continue

                # Get internal label via your dataset logic
                try:
                    labels = src.attrs["class_labels"]
                    if isinstance(labels[0], bytes):
                        labels = [l.decode("utf-8") for l in labels]

                    internal_label = -1
                    for i in reversed(range(min(3, len(labels)))):
                        if label_map.get(labels[i], -1) != -1:
                            internal_label = label_map_internal.get(labels[i], -1)
                            break

                    if internal_label == -1:
                        raise ValueError("Could not resolve internal label")
                except Exception as e:
                    print(f"[Skip] {sid}: failed to parse internal label from 'class_labels': {e}")
                    continue

                # Enforce label cap (if provided)
                if internal_label_cap is not None:
                    if label_counter[internal_label] >= internal_label_cap.get(internal_label, float('inf')):
                        print(f"[Cap] {sid}: label {internal_label} cap reached")
                        continue

                dst = merged_group.create_group(sid)

                for k in src.keys():
                    dst.create_dataset(k, data=src[k][()])

                for attr_key in src.attrs:
                    dst.attrs[attr_key] = src.attrs[attr_key]

                label_counter[internal_label] += 1
                max_epochs_found = max(max_epochs_found, epoch_count)

            h5_out.attrs["max_epochs"] = max_epochs_found
            print(f"max_epochs attribute set to {max_epochs_found}")
            print(f"Merged dataset written to: {output_path} under group '{output_dataset_name}'")

if __name__ == "__main__":
    label_config = {
        "label_map": {"neurotypical": 0, "epileptic": 1, "focal": 2, "generalized": 3, "left": 4, "right": 5},
        "inverse_label_map": {0: "neurotypical", 1: "epileptic", 2: "focal", 3: "generalized", 4: "left", 5: "right"},
    }

    dir_dsa = "ml_tflm/dataset/agenda_data_01/combined_south_africa_monopolar_standard_10_20.h5"
    dir_dsb = "ml_tflm/dataset/agenda_data_02/combined_south_africa_monopolar_standard_10_20.h5"
    dir_dsm = "ml_tflm/dataset/agenda_data_03/balanced_south_africa_monopolar_standard_10_20.h5"

    name_dsa = "combined_south_africa_monopolar_standard_10_20"
    name_dsb = "combined_south_africa_monopolar_standard_10_20"
    name_dsm = "combined_south_africa_monopolar_standard_10_20"

    dataset_a = EEGRecordingDatasetWithShape(
        h5_file_path=dir_dsa,
        dataset_name=name_dsa,
        label_config=label_config, 
        mirror_flag=False
        )
    
    dataset_b = EEGRecordingDatasetWithShape(
        h5_file_path=dir_dsb,
        dataset_name=name_dsb,
        label_config=label_config, 
        mirror_flag=False
        )

    suspected_dups = find_suspected_duplicates(dataset_a.sample_metadata, dataset_b.sample_metadata)

    for sid_a, sid_b, sig in suspected_dups:
        print(f"Suspected dup: A:{sid_a} <-> B:{sid_b}  | Label: {sig[0]}  | Epochs: {sig[1]}")

    confirmed_dups = refine_duplicate_check(suspected_dups, dataset_a, dataset_b)

    print("\n=== Refined Matches ===")
    for sid_a, sid_b, sig, mse in confirmed_dups:
        print(f"Probable dup: A:{sid_a} <-> B:{sid_b} | Label: {sig[0]} | Epochs: {sig[1]} | MSE: {mse:.4f}")

    dataset_a.close()
    dataset_b.close()

    merge_datasets_without_duplicates(
        h5_path_a=dir_dsa,
        dataset_name_a=name_dsa,
        h5_path_b=dir_dsb,
        dataset_name_b=name_dsb,
        confirmed_duplicates=confirmed_dups,
        output_path=dir_dsm,
        output_dataset_name=name_dsm,
        internal_label_cap={0: 120, 3: 120, 4: 100, 5: 100},
        label_config=label_config
    )

    dataset_m = EEGRecordingDatasetWithShape(
        h5_file_path=dir_dsm,
        dataset_name=name_dsm,
        label_config=label_config,
        mirror_flag=False
    )

    print(dataset_m.get_label_histogram())

    dataset_m.close()