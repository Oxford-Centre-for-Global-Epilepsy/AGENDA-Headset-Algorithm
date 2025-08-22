import numpy as np
import matplotlib.pyplot as plt
import random

from pathlib import Path
import os

from ml_tflm.dataset.eeg_dataset_rewrite import EEGRecordingDatasetTF

def sample_reshape(sample, plot=False):
    assert sample.ndim == 3, "Expected sample with shape (E, C, T)"
    E, C, T = sample.shape

    # Left: unreshaped first epoch -> (C, T) -> transpose to (T, C) for plotting over time
    epoch0_TC = sample[0].T  # shape (T, C)

    # Right: reshaped first T rows -> (E, C, T) -> (E*T, C), then take first T rows
    sample_flat = sample.transpose(0, 2, 1).reshape(E * T, C)  # (E*T, C)
    reshaped_firstT = sample_flat[:T]  # (T, C)

    # Plot
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        offset = 3

        # Left subplot: epoch 0 (unreshaped)
        for ch in range(C):
            ax1.plot(epoch0_TC[:, ch] + ch * offset, label=f"Ch {ch}")
        ax1.set_title("Epoch 0 (unreshaped)")
        ax1.set_xlabel("Time index (0..T-1)")
        ax1.set_ylabel("Value + offset")
        ax1.legend(loc="upper right", fontsize="small")

        # Right subplot: reshaped first T rows
        for ch in range(C):
            ax2.plot(reshaped_firstT[:, ch] + ch * offset, label=f"Ch {ch}")
        ax2.set_title("Reshaped first T rows")
        ax2.set_xlabel("Flattened time index (0..T-1)")
        ax2.legend(loc="upper right", fontsize="small")

        fig.suptitle(f"Compare epoch 0 vs. reshaped view (offset={offset})")
        fig.tight_layout()

        out_path = "epoch_vs_reshaped.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")

    return sample_flat

def save_data(sample, path):
    # save to CSV in the current directory
    header = ",".join(f"CH{i:02d}" for i in range(1, sample.shape[1]+1))
    np.savetxt(
        path,  # file name or path
        sample,        # 2-D array to save
        delimiter=",", # comma-separated values
        fmt="%.6f",    # format each number; here 6 decimal places
        header=header, # provide a dummy header
        comments=""    # avoid '#' before the header
    )


if __name__ == "__main__":
    # Resolving the directories
    parent_dir = Path(__file__).resolve().parent
    plot_dir = os.path.join(parent_dir, "test_checker.png")
    data_dir = os.path.join(parent_dir, "teensy_data")

    # Get the dataset
    label_config = {
        "label_map": {"neurotypical": 0, "epileptic": 1, "focal": 2, "generalized": 3, "left": 4, "right": 5},
        "inverse_label_map": {0: "neurotypical", 1: "epileptic", 2: "focal", 3: "generalized", 4: "left", 5: "right"},
    }
    dataset = EEGRecordingDatasetTF(
        h5_file_path="ml_tflm/dataset/agenda_data_23_bp45_tr05/merged_south_africa_monopolar_standard_10_20.h5",
        dataset_name="combined_south_africa_monopolar_standard_10_20",
        label_config=label_config, 
        mirror_flag=False,
        chunk_size=256,
        omit_channels=["A1","A2","Fz","Pz","Cz"]
        )
    
    # Prepare the data containers
    labelwise_target = {
        0: 16,
        3: 8,
        4: 4,
        5: 4
    }
    labelwise_container = {
        0: [],
        3: [],
        4: [],
        5: []
    }

    idx_lookup = list(range(len(dataset)))
    random.shuffle(idx_lookup)

    # Fill up the data container
    for idx in idx_lookup:
        subject = dataset.sample_metadata[idx]
        label = subject["internal_label"]

        if len(labelwise_container[label]) < labelwise_target[label]:
            labelwise_container[label].append(dataset[idx]["data"])

    for key, value in labelwise_container.items():
        count = 0
        for idx in range(len(value)):
            file_name = label_config["inverse_label_map"][key][0].upper() + f"{count:04d}.csv" 
            file_path = os.path.join(data_dir, file_name)
            
            save_data(sample_reshape(value[idx], False), file_path)

            print(f"One File Saved To {str(file_path)}")
            count += 1
    