import h5py
import numpy as np
import atexit

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

if __name__ == "__main__":
    # === Example usage ===

    label_config = {
        "label_map": {"neurotypical": 0, "epileptic": 1, "focal": 2, "generalized": 3, "left": 4, "right": 5},
        "inverse_label_map": {0: "neurotypical", 1: "epileptic", 2: "focal", 3: "generalized", 4: "left", 5: "right"},
    }

    dataset = EEGRecordingDatasetWithShape(
        h5_file_path="ml_tflm/dataset/agenda_data_01/combined_south_africa_monopolar_standard_10_20.h5",
        dataset_name="combined_south_africa_monopolar_standard_10_20",
        label_config=label_config, 
        mirror_flag=False
        )
    
    for entry in dataset.sample_metadata[:5]:
        print(entry)

    dataset.close()
