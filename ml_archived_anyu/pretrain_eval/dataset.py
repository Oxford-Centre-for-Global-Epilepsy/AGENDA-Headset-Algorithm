from ml_tflm.pre_training.dataset_pretrain_precompute import EEGAugmentedBufferDatasetTF, split_bins
from ml_tflm.training.train_utils import _make_tf_dataset, load_label_config
import json

pre_h5_path = "ml_tflm/dataset/agenda_data_01/augmented_buffered.h5"
cls_h5_path = "ml_tflm/dataset/agenda_data_01/combined_south_africa_monopolar_standard_10_20.h5"
cls_h5_name = "combined_south_africa_monopolar_standard_10_20"
label_json_file = "ml_tflm/training/label_map.JSON"
aug_json_file = 'ml_tflm/dataset/agenda_data_01/augmented_bin_info.json'

def build_datasets(pre_h5_path, num_classes, 
                   label_json_file, 
                   cls_h5_path, cls_h5_name, 
                   patient_bin_json_path,
                   buffer_size_per_class=3, batch_size=16, num_views=2,
                   load_pretrain=True,
                   verbose=1):
    # Split bins by class
    train_bins, val_bins = split_bins(pre_h5_path)

    # Flatten allowed bins lists for dataset initialization
    train_allowed = [bin_id for bins in train_bins.values() for bin_id in bins]
    val_allowed = [bin_id for bins in val_bins.values() for bin_id in bins]

    if verbose >= 3:
        print("[DEBUG]: train_allowed")
        print(train_allowed)
        print("[DEBUG]: val_allowed")
        print(val_allowed)

    # Load patient_ids_per_bin JSON file
    with open(patient_bin_json_path, 'r') as f:
        patient_ids_per_bin = json.load(f)

    # Helper to flatten patient lists from bins, ignoring duplicates
    def gather_patients(patient_ids_per_bin, bins):
        patients = set()
        for cls_int, bin_list in bins.items():
            cls_key = f"class_{cls_int}"  # Convert int to string key format
            bin_patient_map = patient_ids_per_bin.get(cls_key, {})
            for bin_id in bin_list:
                bin_patients = bin_patient_map.get(bin_id, [])
                patients.update(bin_patients)
        return list(patients)

    train_patient_ids = gather_patients(patient_ids_per_bin, train_bins)
    val_patient_ids = gather_patients(patient_ids_per_bin, val_bins)

    if verbose >= 3:
        print("[DEBUG]: train_patient_ids")
        print(train_patient_ids)
        print("[DEBUG]: val_patient_ids")
        print(val_patient_ids)

    if load_pretrain:
        # Initialize pretraining datasets
        pre_train_ds = EEGAugmentedBufferDatasetTF(
            h5_path=pre_h5_path,
            allowed_bins=train_allowed,
            num_classes=num_classes,
            buffer_size_per_class=buffer_size_per_class,
            batch_size=batch_size,
            num_views=num_views
        )

        if verbose >= 2:
            print("  -> Pretrain Training Dataset Loaded")
    else:
        pre_train_ds = None

    """
    pre_val_ds = EEGAugmentedBufferDatasetTF(
        h5_path=pre_h5_path,
        allowed_bins=val_allowed,
        num_classes=num_classes,
        buffer_size_per_class=buffer_size_per_class,
        batch_size=batch_size,
        num_views=num_views
    )

    if verbose >= 2:
        print("  -> Pretrain Validation Dataset Loaded")
    """
        
    label_config = load_label_config(label_json_file)

    cls_train_dataset = _make_tf_dataset(train_patient_ids, cls_h5_path, cls_h5_name, label_config=label_config, omit_channels=None, batch_size=1, shuffle=True)
    
    if verbose >= 2:
        print("  -> Classification Training Dataset Loaded")

    cls_val_dataset = _make_tf_dataset(val_patient_ids, cls_h5_path, cls_h5_name, label_config=label_config, omit_channels=None, batch_size=1, shuffle=True)
    
    if verbose >= 2:
        print("  -> Classification Validation Dataset Loaded")

    if verbose >= 1:
        print("-> All Dataset Loaded")

    # Return datasets and patient ID lists
    return pre_train_ds, cls_train_dataset, cls_val_dataset

if __name__ == "__main__":
    pre_train_ds, cls_train_dataset, cls_val_dataset = build_datasets(pre_h5_path=pre_h5_path, num_classes=3, 
                                                                                  label_json_file=label_json_file, cls_h5_path=cls_h5_path, cls_h5_name=cls_h5_name, 
                                                                                  patient_bin_json_path=aug_json_file, 
                                                                                  verbose=2)

    del pre_train_ds
    del cls_train_dataset
    del cls_val_dataset

    print(0)